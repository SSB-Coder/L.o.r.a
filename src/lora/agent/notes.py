"""create_note: the only file-writing tool. New .md/.txt files only, never overwrites, never deletes."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ALLOWED_EXTENSIONS = {"md", "txt"}
MAX_CHARS = 200_000


def default_notes_dir() -> Path:
    env = os.environ.get("LORA_NOTES_DIR")
    return Path(env).expanduser() if env else Path.home() / "Documents" / "LORA"


def _safe_stem(title: str) -> str:
    stem = re.sub(r"[^\w\- ]+", "", title, flags=re.UNICODE).strip().replace("  ", " ")
    stem = re.sub(r"\s+", " ", stem)[:60].strip(" .-_")
    return stem or "Note " + datetime.now().strftime("%Y-%m-%d %H%M")


def create_note(
    title: str,
    content: str,
    ext: str = "md",
    folder: str | Path | None = None,
    policy: Any = None,
) -> dict[str, Any]:
    """Create a new note file. ``policy`` (a PolicyEngine) is consulted when given."""
    ext = (ext or "md").lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        return {"status": "error", "message": f"I can only create .md or .txt files, not .{ext}."}
    body = content or ""
    if not body.strip():
        return {"status": "error", "message": "There is nothing to save yet."}
    if len(body) > MAX_CHARS:
        return {"status": "error", "message": "That note is too large to save."}

    directory = Path(folder).expanduser() if folder else default_notes_dir()
    stem = _safe_stem(title)
    target = directory / f"{stem}.{ext}"
    n = 2
    while target.exists():  # never overwrite
        target = directory / f"{stem} ({n}).{ext}"
        n += 1

    if policy is not None:
        verdict = policy.check_path(str(target), origin="desktop", mode="write")
        if not verdict.allowed:
            return {"status": "denied", "message": f"I'm not allowed to write there: {verdict.reason}"}

    try:
        directory.mkdir(parents=True, exist_ok=True)
        with open(target, "x", encoding="utf-8") as fh:  # "x": fail rather than overwrite
            fh.write(body if body.endswith("\n") else body + "\n")
    except Exception as exc:
        return {"status": "error", "message": f"Couldn't save the file: {exc}"}
    return {"status": "success", "path": str(target), "message": f"Saved {target.name}"}
