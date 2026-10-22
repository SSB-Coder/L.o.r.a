"""Durable Python/native fault evidence for unexpected process exits."""

import atexit
import faulthandler
import logging
import os
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)
_fault_file = None


def enable_fault_reporting(log_dir: Path) -> None:
    global _fault_file
    if _fault_file is not None:
        return
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        _fault_file = (log_dir / f"native-fault-{os.getpid()}.log").open("a", encoding="utf-8")
        faulthandler.enable(file=_fault_file, all_threads=True)
        atexit.register(_close_fault_file)
    except (OSError, RuntimeError):
        logger.exception("Could not enable native fault reporting")

    original = sys.excepthook

    def uncaught(exc_type, value, tb):
        logger.critical("Unhandled main-thread exception", exc_info=(exc_type, value, tb))
        original(exc_type, value, tb)

    def thread_failure(args):
        if args.exc_type is not SystemExit:
            logger.error("Unhandled worker exception in %s", args.thread.name,
                         exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = uncaught
    threading.excepthook = thread_failure
    logger.info("Fault reporting enabled; pid=%d", os.getpid())


def _close_fault_file() -> None:
    global _fault_file
    if _fault_file is not None:
        faulthandler.disable()
        _fault_file.close()
        _fault_file = None
