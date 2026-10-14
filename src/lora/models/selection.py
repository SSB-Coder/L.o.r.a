"""Pick the Ollama model and generation options for chat, from config/lora.yaml.

main.py used to ignore config/lora.yaml entirely and guess model names. This
resolves the configured role (default: ``writer``) against what is actually
installed, falls back through the role's fallback list and then a small-model
first default list (the target machine has a 6 GB GPU), and returns the options
(``num_ctx``, ``temperature``, ``think``) from the same file.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lora.models.ollama_client import OllamaClient, match_models

logger = logging.getLogger(__name__)

# Smaller models first: an 8B+ model spills out of 6 GB VRAM and is several times slower.
PREFERRED_MODEL = "spark-x25-4b"  # Spark-X2.5-4B is the base model
DEFAULT_FALLBACKS: tuple[str, ...] = (
    "gemma4:e4b",
    "granite4.2:8b",
    "ornith-1.5:9b",
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


def candidate_names(user_default: str | None, role_cfg: dict[str, Any], role: str = "writer") -> list[str]:
    names: list[str] = []
    if role != "writer":  # planner / verifier / supervisor: that role's own model first, then the usual order
        for name in (role_cfg.get("name"), *(role_cfg.get("fallback") or [])):
            if name and name not in names:
                names.append(name)
    # Spark first, always. LORA_MODEL overrides; a stale default in the user config comes after it.
    for name in (
        os.environ.get("LORA_MODEL"),
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
) -> ChatPlan | None:
    """Resolve model + options for an interactive reply. None if Ollama has no models."""
    installed = await client.installed_models()
    role_cfg = load_role(role, config_path)
    model, configured = choose_model(candidate_names(user_default, role_cfg, role), installed)
    if model is None:
        return None
    options, think = build_options(role_cfg, role=role)
    note = "" if configured else "No configured model is installed; using the first available one."
    if note:
        logger.warning("%s (%s)", note, model)
    return ChatPlan(model=model, options=options, think=think, note=note)
