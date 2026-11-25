"""Open path in default system handler for L.O.R.A."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from lora.safety.policy import PolicyEngine


def open_path(path: str | Path, policy_engine: PolicyEngine | None = None, *, wait: bool = True) -> dict[str, Any]:
    """Validate path safety and launch the file or folder in Windows Explorer or default viewer."""
    engine = policy_engine or PolicyEngine()
    policy_res = engine.check_path(path, origin="desktop", mode="open")

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

    try:
        final = engine.check_path(path, origin="desktop", mode="open")
        if not final.allowed or final.canonical_path != target:
            return {"status": "error", "message": "Path changed or was denied before opening"}
        if hasattr(os, "startfile"):
            os.startfile(str(target))
        else:
            opener = "/usr/bin/open" if sys.platform == "darwin" else "xdg-open"
            if wait:
                subprocess.run([opener, str(target)], check=True, timeout=15)
            else:
                subprocess.Popen([opener, str(target)])

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
