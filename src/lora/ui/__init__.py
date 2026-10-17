"""L.O.R.A Desktop UI Package (lazy-loaded Qt widgets for headless compatibility)."""

from typing import Any

from lora.ui.audio_cues import AudioCueManager, SoundCue

__all__ = [
    "AmbientDropletWidget",
    "AudioCueManager",
    "LoraDesktopHUD",
    "LoraSystemTray",
    "SoundCue",
    "UiDispatcher",
    "WidgetState",
]


def __getattr__(name: str) -> Any:
    if name == "LoraDesktopHUD":
        from lora.ui.hud import LoraDesktopHUD

        return LoraDesktopHUD
    if name == "LoraSystemTray":
        from lora.ui.tray import LoraSystemTray

        return LoraSystemTray
    if name == "UiDispatcher":
        from lora.ui.threadsafe import UiDispatcher

        return UiDispatcher
    if name in ("AmbientDropletWidget", "WidgetState"):
        from lora.ui import widget as _w

        return getattr(_w, name)
    raise AttributeError(f"module 'lora.ui' has no attribute {name!r}")
