"""L.O.R.A Voice Pipeline package.

Exports are resolved lazily so importing capture/STT does not initialize the
whole application. Historical controllers resolve to lora.experimental.legacy;
the old Qwen worker name is a CPU Kokoro compatibility wrapper.
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
