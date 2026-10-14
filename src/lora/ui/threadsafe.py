"""Hand callables from any thread over to the Qt GUI thread.

Why this exists: ``QTimer.singleShot(0, fn)`` called from a plain Python
thread (``threading.Thread``, the sounddevice callback, the pynput listener)
silently never fires, because that thread has no Qt event loop. Everything
in ``main.py`` that posts a result back to the HUD must go through
:class:`UiDispatcher` instead.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal, Slot

logger = logging.getLogger(__name__)


class UiDispatcher(QObject):
    """Queue callables onto the thread this object lives in (create it on the GUI thread)."""

    _call = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._call.connect(self._run, Qt.ConnectionType.QueuedConnection)

    @Slot(object)
    def _run(self, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception:
            logger.exception("UI callback failed")

    def post(self, fn: Callable[[], Any]) -> None:
        """Safe to call from any thread. ``fn`` runs later on the GUI thread."""
        self._call.emit(fn)
