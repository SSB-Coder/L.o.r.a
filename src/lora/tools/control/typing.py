"""Type text into a text editor window, with every guard that makes keystroke injection safe enough.

* only a short list of editors (Notepad, WordPad, VS Code, Word, OneNote);
* the caller must have the user's explicit OK for the exact text (see AgentRuntime.confirm);
* the foreground window title must match the target both before typing and between chunks;
* never into shells, terminals, the Run box, registry/security/credential/login windows;
* plain text only: no key combinations, nothing but letters, punctuation, tabs and newlines.
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable
from typing import Any

TARGETS: dict[str, tuple[str, str]] = {  # key -> (label, window-title regex)
    "notepad": ("Notepad", r"notepad"),
    "text editor": ("Notepad", r"notepad"),
    "wordpad": ("WordPad", r"wordpad"),
    "word": ("Word", r"\bword\b|\.docx? -"),
    "onenote": ("OneNote", r"onenote"),
    "vscode": ("Visual Studio Code", r"visual studio code"),
    "vs code": ("Visual Studio Code", r"visual studio code"),
    "visual studio code": ("Visual Studio Code", r"visual studio code"),
}
BLOCKED_TITLE = re.compile(
    r"powershell|cmd\.exe|command prompt|terminal|\bbash\b|\bwsl\b|regedit|registry|task manager|"
    r"user account control|windows security|credential|password|sign.?in|log.?in|bank|"
    r"computer management|device manager|\brun\b|services",
    re.IGNORECASE,
)
MAX_CHARS = 2000
CHUNK = 60
_BAD_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class TypingRefused(Exception):  # noqa: N818 - public API
    pass


def foreground_title() -> str:
    if sys.platform != "win32":
        return ""
    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    hwnd = user32.GetForegroundWindow()
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def default_keyboard() -> Any:
    from pynput.keyboard import Controller

    return Controller()


class TypeTextTool:
    def __init__(
        self,
        window_manager: Any,
        foreground: Callable[[], str] = foreground_title,
        keyboard: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.wm, self.foreground, self._keyboard, self.sleep = window_manager, foreground, keyboard, sleep

    @staticmethod
    def resolve_target(name: str) -> tuple[str, str] | None:
        return TARGETS.get(re.sub(r"\s+", " ", name.strip().lower()))

    def validate(self, text: str, target: str) -> tuple[str, str]:
        resolved = self.resolve_target(target)
        if resolved is None:
            raise TypingRefused(f"I only type into editors like Notepad, WordPad, Word or VS Code, not “{target}”.")
        if not text.strip():
            raise TypingRefused("There's nothing to type.")
        if len(text) > MAX_CHARS:
            raise TypingRefused(f"That's too long to type safely (limit {MAX_CHARS} characters).")
        if _BAD_CHARS.search(text):
            raise TypingRefused("That text contains control characters, so I won't type it.")
        return resolved

    def _check_foreground(self, pattern: str) -> None:
        title = self.foreground()
        if BLOCKED_TITLE.search(title):
            raise TypingRefused("The active window looks sensitive, so I stopped.")
        if not re.search(pattern, title, re.IGNORECASE):
            raise TypingRefused("The editor isn't the active window, so I didn't type anything.")

    def type_text(self, text: str, target: str, *, confirmed: bool = False) -> dict[str, Any]:
        if not confirmed:
            raise TypingRefused("Direct typing is deferred until an exact-text confirmation is accepted.")
        label, pattern = self.validate(text, target)
        self.wm.focus_window(pattern)
        self.sleep(0.5)
        self._check_foreground(pattern)
        kb = self._keyboard or default_keyboard()
        typed = 0
        for i in range(0, len(text), CHUNK):
            from lora.safety.killswitch import KillSwitch
            if KillSwitch.get_global().is_halted():
                raise TypingRefused("Stopped typing at the user's request.")
            self._check_foreground(pattern)  # abort if focus moved mid-way
            kb.type(text[i : i + CHUNK])
            typed += len(text[i : i + CHUNK])
            self.sleep(0.03)
        return {"status": "success", "chars": typed, "message": f"Typed {typed} characters into {label}."}
