"""Local shutdown IPC inside the ACL-protected LORA application directory."""

import logging
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def request_shutdown(base_dir: Path | None = None) -> Path:
    root = (base_dir or Path.home() / ".lora").resolve()
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    if control.is_symlink() or not control.resolve().is_relative_to(root):
        raise ValueError("Invalid lifecycle control directory")
    target = control / f"shutdown-{uuid.uuid4().hex}.request"
    with target.open("x", encoding="ascii") as file:
        file.write(str(time.time()))
    return target


def consume_shutdown_request(base_dir: Path, started_at: float) -> bool:
    root = base_dir.resolve()
    control = root / "control"
    if not control.exists() or control.is_symlink() or not control.resolve().is_relative_to(root):
        return False
    requested = False
    for target in list(control.glob("shutdown-*.request"))[:32]:
        if target.is_symlink() or not target.resolve().is_relative_to(control.resolve()):
            continue
        try:
            timestamp = float(target.read_text(encoding="ascii")[:64])
            if timestamp >= started_at and time.time() - timestamp < 60:
                requested = True
            # Only our application-owned IPC messages are retired here.
            target.unlink()
            logger.info("Retired local lifecycle request")
        except (OSError, ValueError):
            logger.warning("Invalid or unreadable lifecycle request")
    return requested
