"""Forgiving name matching for spoken app/site names ("VSCORE" -> VS Code, "crome" -> Chrome)."""

from __future__ import annotations

import difflib
import re
from typing import Any

_SOUNDS_LIKE = (("vscore", "vscode"), ("vscored", "vscode"), ("viscode", "vscode"), ("visualstudiocore", "visualstudiocode"),
                ("notepat", "notepad"), ("crom", "chrome"), ("chrom", "chrome"), ("spotifye", "spotify"))


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _labels(key: str, info: dict[str, Any]) -> list[str]:
    return [key, str(info.get("name", "")), *[str(a) for a in info.get("aliases", [])]]


def best_key(name: str, table: dict[str, dict[str, Any]], cutoff: float = 0.8) -> str | None:
    """Key in ``table`` that ``name`` most plausibly means, or None. Exact (ignoring spaces/punctuation) first."""
    target = squash(name)
    if len(target) < 3:
        return None
    for wrong, right in _SOUNDS_LIKE:
        if target == wrong:
            target = right
    owners: dict[str, str] = {}
    for key, info in table.items():
        for label in _labels(key, info):
            sq = squash(label)
            if sq:
                owners.setdefault(sq, key)
    if target in owners:
        return owners[target]
    if len(target) < 4:
        return None
    close = difflib.get_close_matches(target, list(owners), n=1, cutoff=cutoff)
    return owners[close[0]] if close else None
