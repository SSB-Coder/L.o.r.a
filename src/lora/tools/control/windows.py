"""Desktop Window Enumeration and Focus Management for L.O.R.A."""

import ctypes
import re
import sys
from typing import Any


class WindowManager:
    """Manages read-only desktop window listing and foreground switching."""

    def __init__(self) -> None:
        self.is_windows = sys.platform == "win32"

    def _enumerate_raw_windows(self) -> list[dict[str, Any]]:
        """Enumerate raw top-level windows using Win32 API."""
        if not self.is_windows:
            return []

        windows: list[dict[str, Any]] = []

        enum_windows = ctypes.windll.user32.EnumWindows
        enum_windows_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        get_window_text_w = ctypes.windll.user32.GetWindowTextW
        get_window_text_length_w = ctypes.windll.user32.GetWindowTextLengthW
        is_window_visible = ctypes.windll.user32.IsWindowVisible

        def foreach_window(hwnd: ctypes.c_void_p, _: ctypes.c_void_p) -> bool:
            if is_window_visible(hwnd):
                length = get_window_text_length_w(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    get_window_text_w(hwnd, buff, length + 1)
                    title = buff.value
                    if title and title.strip():
                        windows.append({
                            "handle": hwnd,
                            "title": title.strip(),
                        })
            return True

        enum_windows(enum_windows_proc(foreach_window), 0)
        return windows

    def _bring_to_foreground(self, hwnd: int) -> bool:
        """Call Win32 SetForegroundWindow."""
        if not self.is_windows:
            return False
        try:
            # SW_RESTORE = 9 in Win32
            ctypes.windll.user32.ShowWindow(hwnd, 9)
            return bool(ctypes.windll.user32.SetForegroundWindow(hwnd))
        except Exception:
            return False

    def list_windows(self) -> list[dict[str, Any]]:
        """Return list of active, visible desktop application windows with non-empty titles."""
        raw = self._enumerate_raw_windows()
        return [w for w in raw if w.get("title")]

    def focus_window(self, title_regex: str) -> dict[str, Any]:
        """Search open windows by regex and bring the first matching window to the foreground."""
        try:
            pattern = re.compile(title_regex, re.IGNORECASE)
        except re.error:
            pattern = re.compile(re.escape(title_regex), re.IGNORECASE)

        windows = self.list_windows()

        for win in windows:
            if pattern.search(win["title"]):
                success = self._bring_to_foreground(win["handle"])
                return {
                    "status": "success" if success else "partial",
                    "matched_window": win["title"],
                    "handle": win["handle"],
                    "message": f"Switched focus to window: {win['title']}",
                }

        return {
            "status": "error",
            "message": f"No open window found matching '{title_regex}'.",
        }
