"""Preserve a legacy notebook store when moving to the plan's application path."""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def migrate_notebooks(base_dir: Path, repository: Path | None = None) -> Path:
    target = base_dir / "notebooks" / "notebooks.db"
    if target.exists():
        return target
    candidates = [base_dir / "memory" / "notebooks.db"]
    if repository is not None:
        candidates.append(repository / "data" / "notebooks" / "notebooks.db")
    candidates = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if not candidates:
        return target
    if len(candidates) > 1:
        raise RuntimeError("Multiple legacy notebook stores exist; select a store before migration")
    source = candidates[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb"):
            pass
    except FileExistsError:
        return target
    try:
        with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True, timeout=2) as old:
            with sqlite3.connect(target, timeout=2) as new:
                old.backup(new)
        logger.info("Copied legacy notebooks; original store preserved")
    except Exception:
        # This empty/new target belongs to the migration, never the user's source.
        target.unlink(missing_ok=True)
        raise
    return target
