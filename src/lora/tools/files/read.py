"""Safe file reading with PyMuPDF PDF extraction for L.O.R.A."""

import os
from pathlib import Path
from typing import Any

try:
    import pymupdf as fitz
except ImportError:
    import fitz

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

    # Enforce size cap
    size = target.stat().st_size
    if size > max_bytes:
        return {
            "status": "error",
            "message": f"File size ({size} bytes) exceeds maximum allowed size ({max_bytes} bytes).",
            "content": "",
        }

    suffix = target.suffix.lower()

    # 1. PDF Extraction via PyMuPDF
    if suffix == ".pdf":
        try:
            pages = []
            full_text_parts = []
            with fitz.open(str(target)) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    pages.append({
                        "page": page_num + 1,
                        "text": text,
                    })
                    full_text_parts.append(text)

            return {
                "status": "success",
                "file_type": "pdf",
                "path": str(target),
                "total_pages": len(pages),
                "pages": pages,
                "content": "\n\n".join(full_text_parts),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to extract text from PDF: {e}",
                "content": "",
            }

    # 2. Plain Text / Markdown / CSV / JSON
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
