"""Open path in default system handler for L.O.R.A."""

import os
from pathlib import Path
from typing import Any

from lora.safety.policy import PolicyEngine


def open_path(path: str | Path, policy_engine: PolicyEngine | None = None) -> dict[str, Any]:
    """Validate path safety and launch the file or folder in Windows Explorer or default viewer."""
    engine = policy_engine or PolicyEngine()
    policy_res = engine.check_path(path, origin="desktop", mode="read")

    if not policy_res.allowed:
        return {
            "status": "error",
            "message": f"Path access denied by policy: {policy_res.reason}",
        }

    target = policy_res.canonical_path or Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()
    if not target.exists():
        return {
            "status": "error",
            "message": f"Target path '{path}' does not exist.",
        }

    # Plan §4.2.1: Guard against spreadsheet formula/DDE payload auto-execution in .csv/.tsv files
    if target.is_file() and target.suffix.lower() in (".csv", ".tsv"):
        try:
            sample = target.read_text(encoding="utf-8", errors="ignore")[:8192]
            import re as _re

            if _re.search(r"(?:^|[,\t\r\n])\s*[=+\-@](?:cmd|powershell|wscript|mshta|rundll32|hyperlink)\b", sample, _re.IGNORECASE):
                return {
                    "status": "error",
                    "message": f"Blocked opening '{target.name}' in external spreadsheet viewer: potentially unsafe formula/DDE payload detected.",
                }
        except Exception:
            pass

    try:
        # os.startfile opens the file or directory in the default Windows application
        if hasattr(os, "startfile"):
            os.startfile(str(target))
        else:
            # Fallback for cross-platform dev
            import subprocess
            subprocess.run(["xdg-open", str(target)], check=False)

        return {
            "status": "success",
            "path": str(target),
            "is_dir": target.is_dir(),
            "message": f"Successfully opened '{target.name}'.",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to open path '{path}': {e}",
        }
