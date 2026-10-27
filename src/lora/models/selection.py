"""Pick the Ollama model and generation options for chat, from config/lora.yaml.

Supports:
- Manual model selection (`manual_model` override from UI or voice/text command)
- Automatic complexity & modality routing tailored to installed local Ollama models:
  - `writer` / `reader` / `planner` -> Spark-X2.6-4B Q4_K_M, sharing an 8192-token profile
  - `controller` / `verifier` -> Granite 3B
  - `complex` / `supervisor` -> Granite 8B, with configured specialist fallbacks
  - `vision` / `optimizer` -> `gemma4:e4b`
- Hard-excludes oversized models (`gemma4:26b`, 18 GB) from automatic selection on 16 GB RAM
- Strict v1.5 Plan rule: ONE fixed `num_ctx` per model tag (8192 for interactive models, 32768 only for `-32k` reader)
  so Ollama never unloads/reloads the same model between steps.
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
from lora.models.profiles import (
    KEEP_ALIVE as KEEP_ALIVE,
)
from lora.models.profiles import (
    MAX_INTERACTIVE_CTX as MAX_INTERACTIVE_CTX,
)
from lora.models.profiles import (
    PREFERRED_MODEL as PREFERRED_MODEL,
)
from lora.models.profiles import (
    QWYTHOS_MODEL as QWYTHOS_MODEL,
)

logger = logging.getLogger(__name__)

# Use the installed quantized tag for startup, chat, planning, and document reading.

# Oversized models (>12 GB) that must never be auto-selected on a 6 GB VRAM / 16 GB RAM laptop
EXCLUDED_MODELS: tuple[str, ...] = (
    "gemma4:26b",
    "26b",
    "27b",
    "32b",
    "70b",
)

DEFAULT_FALLBACKS: tuple[str, ...] = (
    "spark-x26-4b",
    "granite4.2:3b",
    "gemma4:e4b",
    "granite4.2:8b",
    QWYTHOS_MODEL,
    "ornith-1.5:9b",
)
COMPLEX_PREFERENCE: tuple[str, ...] = (
    "granite4.2:8b",
    QWYTHOS_MODEL,
    "ornith-1.5:9b",
    "gemma4:e4b",
    PREFERRED_MODEL,
)
VISION_PREFERENCE: tuple[str, ...] = (
    "gemma4:e4b",
    "gemma3:4b",
    "gemma3",
    "llava",
    "qwen2.5vl",
    "minicpm-v",
    "llama3.2-vision",
    "moondream",
    "granite4.2:8b",
    PREFERRED_MODEL,
)


@dataclass(frozen=True)
class ChatPlan:
    model: str
    options: dict[str, Any] = field(default_factory=dict)
    think: bool | str | None = None
    keep_alive: str = KEEP_ALIVE
    note: str = ""
    role: str = "writer"
    escalated: bool = False


def is_excluded_model(model_name: str) -> bool:
    """True if `model_name` is too large for 6 GB VRAM / 16 GB RAM automatic selection (e.g., gemma4:26b)."""
    low = (model_name or "").strip().lower()
    return any(tok in low for tok in EXCLUDED_MODELS)


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
    r"critique|benchmark|formal|mathematical|system design|why\s+(?:does|do|is|are|did|would)|"
    r"how\s+(?:does|do|would|can\s+we)|explain\s+how|what\s+if|reason(?:ing)?|mythos)\b",
    re.IGNORECASE,
)


def is_complex_query(query: str) -> bool:
    """True when the user request warrants scaling up from the 4B base writer to Granite-8B / Qwythos-9B."""
    q = (query or "").strip()
    if len(q.split()) >= 14:
        return True
    if q.count("\n") >= 2 or "```" in q:
        return True
    return bool(_COMPLEX_QUERY_RE.search(q))


def is_vision_capable(model_name: str) -> bool:
    """True if the Ollama model tag is a known multimodal vision model."""
    low = (model_name or "").lower()
    return any(v in low for v in ("gemma4", "llava", "gemma3", "vl", "vision", "minicpm-v", "moondream", "bakllava", "pixtral"))


def pick_escalation_model(installed: list[str], current_model: str) -> str | None:
    """Choose a stronger 8B/9B model (Granite-8B, Qwythos-9B, Ornith-9B) when the 4B model is insufficient."""
    cur_low = (current_model or "").lower()
    safe_installed = [m for m in installed if not is_excluded_model(m)]
    for pref in ("granite4.2:8b", QWYTHOS_MODEL, "qwythos", "ornith-1.5:9b", "gemma4:e4b"):
        for found in match_models(pref, safe_installed):
            if found.lower() != cur_low:
                return found
    for cand in safe_installed:
        if cand.lower() != cur_low and any(k in cand.lower() for k in ("8b", "9b", "qwythos", "granite", "ornith")):
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

    if role != "writer":  # planner / verifier / supervisor / controller / reader: that role's own model first
        for name in (role_cfg.get("name"), *(role_cfg.get("fallback") or [])):
            if name and name not in names:
                names.append(name)

    # Spark 4B first for fast base writer; LORA_MODEL or manual_model overrides.
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


def fixed_num_ctx_for_model(model_name: str, role: str = "writer", configured_ctx: int | None = None) -> int:
    """v1.5 Plan §3.3: Return ONE fixed num_ctx per model tag so Ollama never reloads on role transitions."""
    m_low = (model_name or "").lower()
    if "granite" in m_low:
        return 6144
    if "32k" in m_low:
        return 32768 if role == "reader" else MAX_INTERACTIVE_CTX
    if configured_ctx is not None and not model_name:
        return min(int(configured_ctx), MAX_INTERACTIVE_CTX)
    return MAX_INTERACTIVE_CTX


def build_options(
    role_cfg: dict[str, Any],
    role: str = "writer",
    model_name: str = "",
) -> tuple[dict[str, Any], bool | str | None]:
    options: dict[str, Any] = {}
    if "temperature" in role_cfg:
        options["temperature"] = float(role_cfg["temperature"])
    raw_ctx = role_cfg.get("num_ctx")
    if model_name:
        options["num_ctx"] = fixed_num_ctx_for_model(model_name, role=role, configured_ctx=raw_ctx)
    elif raw_ctx:
        options["num_ctx"] = min(int(raw_ctx), 32768 if role == "reader" else MAX_INTERACTIVE_CTX)
    else:
        options["num_ctx"] = 4096
    think = role_cfg.get("think", False)
    if isinstance(think, bool) or isinstance(think, str):
        return options, think
    return options, None


def choose_model(candidates: list[str], installed: list[str], allow_excluded: bool = False) -> tuple[str | None, bool]:
    """Return (model, was_configured). Excludes oversized models (e.g. gemma4:26b) unless explicitly requested."""
    pool = list(installed) if allow_excluded else [m for m in installed if not is_excluded_model(m)]
    for cand in candidates:
        search_pool = pool
        found = match_models(cand, search_pool)
        if found:
            return min(found, key=_variant_rank), True
    return (pool[0], False) if pool else (None, False)


def _variant_rank(name: str) -> tuple[bool, bool, bool, int]:
    """Among several tags of one model prefer the 8k / native tag over 32k or duplicate HF GGUF builds."""
    low = name.lower()
    is_long_ctx = any(k in low for k in ("32k", "16k", "128k"))
    is_dup_spark_hf = "hf.co/xhtoken/spark" in low
    return (is_long_ctx, is_dup_spark_hf, "8k" not in low, len(low))


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
    allow_excl = bool(manual_model and is_excluded_model(manual_model))
    model, configured = choose_model(cands, installed, allow_excluded=allow_excl)
    if model is None:
        return None
    options, think = build_options(role_cfg, role=role, model_name=model)
    note = "" if configured else "No configured model is installed; using the first available one."
    if note:
        logger.warning("%s (%s)", note, model)
    return ChatPlan(
        model=model,
        options=options,
        think=think,
        note=note,
        role=role,
        escalated=(role in ("complex", "agent", "vision", "supervisor")),
    )
