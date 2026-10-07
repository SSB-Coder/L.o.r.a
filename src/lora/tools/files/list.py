import os
from datetime import datetime
from pathlib import Path
from typing import Any

from lora.safety.policy import PolicyEngine


def list_dir(path: str | Path, policy_engine: PolicyEngine | None = None) -> dict[str, Any]:
    """List directory contents within allowed roots."""
    engine = policy_engine or PolicyEngine()
    policy_res = engine.check_path(path, origin="desktop", mode="read")

    if not policy_res.allowed:
        return {
            "status": "error",
            "message": f"Directory access denied by policy: {policy_res.reason}",
            "items": [],
        }

    target = policy_res.canonical_path or Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()
    if not target.exists() or not target.is_dir():
        return {
            "status": "error",
            "message": f"Target path '{path}' is not an existing directory.",
            "items": [],
        }

    try:
        items = []
        for entry in target.iterdir():
            stat = entry.stat()
            items.append({
                "name": entry.name,
                "is_dir": entry.is_dir(),
                "size_bytes": stat.st_size if entry.is_file() else 0,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        return {
            "status": "success",
            "path": str(target),
            "count": len(items),
            "items": sorted(items, key=lambda x: (not x["is_dir"], x["name"].lower())),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list directory '{path}': {e}",
            "items": [],
        }
