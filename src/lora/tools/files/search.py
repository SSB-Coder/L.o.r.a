import os
from pathlib import Path
from typing import Any

from lora.safety.policy import PolicyEngine


def search_files(
    query: str,
    root: str | Path = ".",
    max_results: int = 50,
    policy_engine: PolicyEngine | None = None,
) -> dict[str, Any]:
    """Search filenames and text contents within allowed roots."""
    engine = policy_engine or PolicyEngine()
    policy_res = engine.check_path(root, origin="desktop", mode="read")

    if not policy_res.allowed:
        return {
            "status": "error",
            "message": f"Search root access denied by policy: {policy_res.reason}",
            "matches": [],
        }

    search_root = policy_res.canonical_path or Path(os.path.expandvars(os.path.expanduser(str(root)))).resolve()
    if not search_root.exists() or not search_root.is_dir():
        return {
            "status": "error",
            "message": f"Search root '{root}' is not an existing directory.",
            "matches": [],
        }

    query_lower = query.strip().lower()
    matches = []

    # Directories to ignore
    ignore_dirs = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".idea", ".vscode"}
    scanned_files = 0
    max_scanned_files = 10_000

    try:
        for current_dir, dirnames, filenames in os.walk(search_root):
            # Prune ignored directories in-place to stop os.walk from descending into them
            dirnames[:] = [d for d in dirnames if d not in ignore_dirs and not d.startswith(".")]

            for fname in filenames:
                scanned_files += 1
                if scanned_files > max_scanned_files:
                    break

                p = Path(current_dir) / fname

                # Check if filename matches
                if query_lower in p.name.lower():
                    matches.append({
                        "path": str(p),
                        "filename": p.name,
                        "match_type": "filename",
                        "snippet": "",
                    })
                    if len(matches) >= max_results:
                        break
                    continue

                # Check if small text file matches content
                if p.suffix.lower() in [".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml"]:
                    try:
                        if p.stat().st_size <= 2_000_000:  # skip files > 2 MB for search
                            content = p.read_text(encoding="utf-8", errors="ignore")
                            if query_lower in content.lower():
                                # Find first line containing query
                                snippet = ""
                                for line in content.splitlines():
                                    if query_lower in line.lower():
                                        snippet = line.strip()[:150]
                                        break
                                matches.append({
                                    "path": str(p),
                                    "filename": p.name,
                                    "match_type": "content",
                                    "snippet": snippet,
                                })
                                if len(matches) >= max_results:
                                    break
                    except Exception:
                        continue

            if len(matches) >= max_results or scanned_files > max_scanned_files:
                break

        return {
            "status": "success",
            "query": query,
            "root": str(search_root),
            "count": len(matches),
            "matches": matches,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Search failed: {e}",
            "matches": [],
        }
