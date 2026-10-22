"""L.O.R.A Voice Pipeline package.

Exports are resolved lazily. ``VoiceController`` pulls in the whole tool stack
(planner, Laya, Playwright, FAISS...); importing it eagerly meant that a single
missing optional dependency made ``from lora.voice.capture import AudioCapture``
fail, and main.py swallowed that as a log warning, so mic and TTS never started.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "AudioBuffer": "lora.voice.capture",
    "AudioCapture": "lora.voice.capture",
    "VoiceController": "lora.voice.controller",
    "VoiceState": "lora.voice.controller",
    "PushToTalkListener": "lora.voice.hotkey",
    "FasterWhisperTranscriber": "lora.voice.stt",
    "TranscriptionResult": "lora.voice.stt",
    "TtsEngine": "lora.voice.tts.engine",
    "split_sentences": "lora.voice.tts.engine",
    "Qwen3TtsWorkerProcess": "lora.voice.tts.worker",
    "KokoroTtsWorkerProcess": "lora.voice.tts.worker",
    "SpeechState": "lora.voice.vad",
    "VoiceActivityDetector": "lora.voice.vad",
    "SpeakerVerifier": "lora.voice.wakeword",
    "WakeWordDetector": "lora.voice.wakeword",
    "WakeWordResult": "lora.voice.wakeword",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'lora.voice' has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)
