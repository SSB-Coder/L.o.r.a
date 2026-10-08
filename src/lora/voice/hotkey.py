"""Push-to-Talk Hotkey Listener (Ctrl+Alt+Space) for L.O.R.A Voice Pipeline."""

import threading
from collections.abc import Callable
from typing import Any


class PushToTalkListener:
    """Global keyboard hotkey listener for push-to-talk activation."""

    def __init__(
        self,
        on_activate: Callable[[], None] | None = None,
        on_deactivate: Callable[[], None] | None = None,
    ) -> None:
        self.on_activate = on_activate
        self.on_deactivate = on_deactivate
        self.is_active = False

        self._current_keys: set[Any] = set()
        self._listener: Any = None
        self._lock = threading.Lock()

    def simulate_press(self) -> None:
        """Simulate hotkey activation (used for unit tests and programmatic triggering)."""
        with self._lock:
            if not self.is_active:
                self.is_active = True
                if self.on_activate:
                    try:
                        self.on_activate()
                    except Exception:
                        pass

    def simulate_release(self) -> None:
        """Simulate hotkey deactivation."""
        with self._lock:
            if self.is_active:
                self.is_active = False
                if self.on_deactivate:
                    try:
                        self.on_deactivate()
                    except Exception:
                        pass

    def _is_hotkey_pressed(self) -> bool:
        """Check if Ctrl + Alt + Space combination is currently pressed."""
        from pynput import keyboard

        has_ctrl = any(
            k in self._current_keys
            for k in [keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r]
        )
        has_alt = any(
            k in self._current_keys
            for k in [keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r]
        )
        has_space = keyboard.Key.space in self._current_keys

        return has_ctrl and has_alt and has_space

    def _on_key_press(self, key: Any) -> None:
        with self._lock:
            self._current_keys.add(key)
            if self._is_hotkey_pressed() and not self.is_active:
                self.is_active = True
                if self.on_activate:
                    try:
                        self.on_activate()
                    except Exception:
                        pass

    def _on_key_release(self, key: Any) -> None:
        with self._lock:
            self._current_keys.discard(key)
            if self.is_active and not self._is_hotkey_pressed():
                self.is_active = False
                if self.on_deactivate:
                    try:
                        self.on_deactivate()
                    except Exception:
                        pass

    def start(self) -> None:
        """Start listening for keyboard events in background thread."""
        try:
            from pynput import keyboard
            self._listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self._listener.daemon = True
            self._listener.start()
        except Exception:
            # Fallback if running headless or environment lacks GUI hooks
            self._listener = None

    def stop(self) -> None:
        """Stop keyboard listener."""
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
