"""Pick the Ollama model and generation options for chat, from config/lora.yaml.

Supports:
- Manual model selection (`manual_model` override from UI or voice/text command)
- Automatic complexity & modality routing (`writer` -> 4B fast, `complex`/`agent` -> 8B/9B, `vision` -> multimodal)
- Runtime auto-escalation (`pick_escalation_model`) when a task is too complex for the current model
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lora.models.ollama_client import OllamaClient, match_models

logger = logging.getLogger(__name__)

# Smaller models first for quick turns: an 8B+ model spills out of 6 GB VRAM and is slower.
PREFERRED_MODEL = "spark-x25-4b"  # Spark-X2.5-4B is the base fast model
DEFAULT_FALLBACKS: tuple[str, ...] = (
    "gemma4:e4b",
    "granite4.2:8b",
    "ornith-1.5:9b",
)
COMPLEX_PREFERENCE: tuple[str, ...] = (
    "granite4.2:8b",
    "ornith-1.5:9b",
    "gemma4:e4b",
    PREFERRED_MODEL,
)
VISION_PREFERENCE: tuple[str, ...] = (
    "gemma3:4b",
    "gemma3",
    "llava",
    "qwen2.5vl",
    "minicpm-v",
    "llama3.2-vision",
    "moondream",
    "gemma4:e4b",
    "granite4.2:8b",
    PREFERRED_MODEL,
)
MAX_INTERACTIVE_CTX = 8192
KEEP_ALIVE = "24h"


@dataclass(frozen=True)
class ChatPlan:
    model: str
    options: dict[str, Any] = field(default_factory=dict)
    think: bool | str | None = None
    keep_alive: str = KEEP_ALIVE
    note: str = ""
    role: str = "writer"
    escalated: bool = False


def find_config_file(name: str = "lora.yaml") -> Path | None:
    for base in (*Path(__file__).resolve().parents, Path.cwd()):
        cand = base / "config" / name
        if cand.is_file():
            return cand
    return None


def load_role(role: str = "writer", config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or find_config_file()
    if path is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}
    role_cfg = (data.get("models") or {}).get(role) or {}
    return role_cfg if isinstance(role_cfg, dict) else {}


_COMPLEX_QUERY_RE = re.compile(
    r"\b(in[- ]?depth|deep(?:ly)?|thorough(?:ly)?|detailed|comprehensive(?:ly)?|compare|contrast|"
    r"architecture|mechanism|derive|derivation|prove|proof|algorithm|implement|debug|refactor|"
    r"step[- ]by[- ]step|analyze|analysis|evaluate|trade[- ]?offs?|multi[- ]?step|synthesize|"
    r"critique|benchmark|formal|mathematical|system design)\b",
    re.IGNORECASE,
)


def is_complex_query(query: str) -> bool:
    """True when the user request warrants scaling up from the 4B writer to an 8B/9B or 32k model."""
    q = (query or "").strip()
    if len(q.split()) >= 18:
        return True
    if q.count("\n") >= 3 or "```" in q:
        return True
    return bool(_COMPLEX_QUERY_RE.search(q))


def is_vision_Capable(model_name: str) -> bool:
    """True if the Ollama model tag is a known multimodal vision model."""
    low = (model_name or "").lower()
    return any(v in low for v in ("llava", "gemma3", "vl", "vision", "minicpm-v", "moondream", "bakllava", "pixtral"))


def pick_escalation_model(installed: list[str], current_model: str) -> str | None:
    """Choose a stronger 8B/9B model when the current model is 4B or failed a complex step."""
    cur_low = (current_model or "").lower()
    for pref in ("granite4.2:8b", "ornith-1.5:9b", "gemma4:e4b"):
        for found in match_models(pref, installed):
            if found.lower() != cur_low:
                return found
    for cand in installed:
        if cand.lower() != cur_low and any(k in cand.lower() for k in ("8b", "9b", "7b", "14b", "12b", "granite", "ornith")):
            return cand
    return None


def candidate_names(
    user_default: str | None,
    role_cfg: dict[str, Any],
    role: str = "writer",
    manual_model: str | None = None,
) -> list[str]:
    names: list[str] = []
    if manual_model and manual_model.strip().lower() not in ("", "auto"):
        names.append(manual_model.strip())

    env_override = os.environ.get("LORA_MODEL")
    if env_override and env_override not in names:
        names.append(env_override)

    if role == "vision":
        for name in (role_cfg.get("name"), *(role_cfg.get("fallback") or []), *VISION_PREFERENCE):
            if name and name not in names:
                names.append(name)
        return names

    if role in ("complex", "agent"):
        for name in (role_cfg.get("name"), *COMPLEX_PREFERENCE, *(role_cfg.get("fallback") or []), user_default):
            if name and name not in names:
                names.append(name)
        return names

    if role != "writer":  # planner / verifier / supervisor / reader: that role's own model first
        for name in (role_cfg.get("name"), *(role_cfg.get("fallback") or [])):
            if name and name not in names:
                names.append(name)
    # Spark first for standard fast writer. LORA_MODEL overrides; a stale default in the user config comes after it.
    for name in (
        PREFERRED_MODEL,
        role_cfg.get("name"),
        user_default,
        *(role_cfg.get("fallback") or []),
        *DEFAULT_FALLBACKS,
    ):
        if name and name not in names:
            names.append(name)
    return names


def build_options(role_cfg: dict[str, Any], role: str = "writer") -> tuple[dict[str, Any], bool | str | None]:
    options: dict[str, Any] = {}
    if "temperature" in role_cfg:
        options["temperature"] = float(role_cfg["temperature"])
    num_ctx = int(role_cfg.get("num_ctx") or 4096)
    ctx_cap = 32768 if role == "reader" else MAX_INTERACTIVE_CTX
    options["num_ctx"] = min(num_ctx, ctx_cap)
    think = role_cfg.get("think", False)
    if isinstance(think, bool) or isinstance(think, str):
        return options, think
    return options, None


def choose_model(candidates: list[str], installed: list[str]) -> tuple[str | None, bool]:
    """Return (model, was_configured). Falls back to the first installed model."""
    for cand in candidates:
        found = match_models(cand, installed)
        if found:
            return min(found, key=_variant_rank), True
    return (installed[0], False) if installed else (None, False)


def _variant_rank(name: str) -> tuple[bool, bool, int]:
    """Among several tags of one model prefer the small-context build (faster, less VRAM)."""
    low = name.lower()
    return (any(k in low for k in ("32k", "16k", "128k")), "8k" not in low, len(low))


async def plan_chat(
    client: OllamaClient,
    user_default: str | None = None,
    role: str = "writer",
    config_path: Path | None = None,
    manual_model: str | None = None,
) -> ChatPlan | None:
    """Resolve model + options for an interactive reply. None if Ollama has no models."""
    installed = await client.installed_models()
    role_cfg = load_role(role, config_path)
    cands = candidate_names(user_default, role_cfg, role, manual_model=manual_model)
    model, configured = choose_model(cands, installed)
    if model is None:
        return None
    options, think = build_options(role_cfg, role=role)
    note = "" if configured else "No configured model is installed; using the first available one."
    if note:
        logger.warning("%s (%s)", note, model)
    return ChatPlan(
        model=model,
        options=options,
        think=think,
        note=note,
        role=role,
        escalated=(role in ("complex", "agent", "vision")),
    )
