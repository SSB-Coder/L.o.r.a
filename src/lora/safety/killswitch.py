"""Emergency Kill Switch for L.O.R.A.

Provides an immediate, non-bypassable emergency brake across all threads and processes.
When triggered:
- Halts all active and pending tool calls immediately.
- Cancels ongoing generation and agent state machines.
- Locks the agent runtime into a safe HALTED state until manually reset.
"""

import threading
from collections.abc import Callable


class KillSwitch:
    """Thread-safe emergency halt controller."""

    _global_instance: "KillSwitch | None" = None
    _global_lock = threading.Lock()

    def __init__(self) -> None:
        self._halted: bool = False
        self._reason: str = ""
        self._callbacks: list[Callable[[str], None]] = []
        self._lock = threading.Lock()

    @classmethod
    def get_global(cls) -> "KillSwitch":
        """Get or initialize the process-wide global KillSwitch instance."""
        with cls._global_lock:
            if cls._global_instance is None:
                cls._global_instance = KillSwitch()
            return cls._global_instance

    @property
    def halt_reason(self) -> str:
        return self._reason

    def is_halted(self) -> bool:
        """Return True if emergency kill switch has been activated."""
        with self._lock:
            return self._halted

    def trigger(self, reason: str = "Emergency kill switch activated") -> None:
        """Activate the emergency kill switch and fire all registered stop callbacks."""
        with self._lock:
            self._halted = True
            self._reason = reason
            callbacks_to_fire = list(self._callbacks)

        # Fire callbacks outside the lock to prevent deadlocks
        for cb in callbacks_to_fire:
            try:
                cb(reason)
            except Exception:
                pass

    def reset(self) -> None:
        """Reset the kill switch state back to normal active operations."""
        with self._lock:
            self._halted = False
            self._reason = ""

    def register_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback function to be invoked when the kill switch triggers."""
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
