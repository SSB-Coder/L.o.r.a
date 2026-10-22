"""Safe file reading with PyMuPDF (or pypdf fallback) PDF extraction for L.O.R.A."""

import os
from pathlib import Path
from typing import Any

from lora.safety.policy import PolicyEngine


def read_file(
    path: str | Path,
    max_bytes: int = 10_000_000,
    policy_engine: PolicyEngine | None = None,
) -> dict[str, Any]:
    """Safely read text, markdown, CSV, JSON, or extract PDF pages."""
    engine = policy_engine or PolicyEngine()
    policy_res = engine.check_path(path, origin="desktop", mode="read")

    if not policy_res.allowed:
        return {
            "status": "error",
            "message": f"File access denied by policy: {policy_res.reason}",
            "content": "",
        }

    target = policy_res.canonical_path or Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()
    if not target.exists() or not target.is_file():
        return {
            "status": "error",
            "message": f"Target path '{path}' is not an existing file.",
            "content": "",
        }

    size = target.stat().st_size
    if size > max_bytes:
        return {
            "status": "error",
            "message": f"File size ({size} bytes) exceeds maximum allowed size ({max_bytes} bytes).",
            "content": "",
        }

    suffix = target.suffix.lower()

    if suffix in {".pdf", ".docx"}:
        from lora.safety.parsers import parse_document
        try:
            result = parse_document(suffix[1:], target.read_bytes())
            return {"status": "success", "file_type": suffix[1:], "path": str(target),
                    "total_pages": len(result["pages"]), "pages": result["pages"], "content": result["text"]}
        except Exception as exc:
            return {"status": "error", "message": f"Isolated parser failed: {exc}", "content": ""}

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        return {
            "status": "success",
            "file_type": "text",
            "path": str(target),
            "size_bytes": size,
            "content": content,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to read file '{path}': {e}",
            "content": "",
        }
