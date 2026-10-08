"""TTS Engine with Sentence Streaming and Barge-In Audio Cancellation."""

import re
import threading
from collections.abc import Callable

import numpy as np

from lora.voice.tts.worker import Qwen3TtsWorkerProcess, TtsAudioChunk


def split_sentences(text: str) -> list[str]:
    """Split text into spoken sentence units while preserving punctuation."""
    clean = text.strip()
    if not clean:
        return []
    # Match sentence terminators followed by whitespace or end of string
    parts = re.split(r"(?<=[.!?])\s+", clean)
    return [p.strip() for p in parts if p.strip()]


class TtsEngine:
    """High-level TTS engine coordinating sentence synthesis, audio playback, and barge-in."""

    def __init__(
        self,
        worker: Qwen3TtsWorkerProcess | None = None,
        audio_playback_sink: Callable[[np.ndarray, int], None] | None = None,
    ) -> None:
        self.worker = worker or Qwen3TtsWorkerProcess()
        self.audio_playback_sink = audio_playback_sink
        self.is_speaking = False
        self._interrupt_event = threading.Event()
        self._lock = threading.Lock()

    def interrupt(self) -> None:
        """Trigger immediate barge-in interruption, stopping speech and clearing the queue."""
        with self._lock:
            self._interrupt_event.set()
            self.is_speaking = False
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass

    def synthesize_and_speak(self, text: str, voice: str = "default") -> list[TtsAudioChunk]:
        """Synthesize sentences sequentially with barge-in check between sentences."""
        sentences = split_sentences(text)
        if not sentences:
            return []

        with self._lock:
            self._interrupt_event.clear()
            self.is_speaking = True
        chunks: list[TtsAudioChunk] = []

        try:
            for idx, sentence in enumerate(sentences):
                # Check for barge-in before synthesis
                if self._interrupt_event.is_set():
                    break

                chunk = self.worker.synthesize(sentence, job_id=str(idx), voice=voice)
                chunks.append(chunk)

                # Check for barge-in before playback
                if self._interrupt_event.is_set():
                    break

                if len(chunk.samples) > 0:
                    self._play_chunk(chunk.samples, chunk.sample_rate)
        finally:
            with self._lock:
                self.is_speaking = False

        return chunks

    def _play_chunk(self, samples: np.ndarray, sample_rate: int = 24000) -> None:
        """Output audio to sounddevice or custom sink."""
        if self._interrupt_event.is_set():
            return

        if self.audio_playback_sink is not None:
            self.audio_playback_sink(samples, sample_rate)
            return

        try:
            import sounddevice as sd
            sd.play(samples, samplerate=sample_rate)
            # Check for interrupt during playback
            duration = len(samples) / sample_rate
            self._interrupt_event.wait(timeout=duration)
            if self._interrupt_event.is_set():
                sd.stop()
        except Exception:
            # Headless or test environment without audio hardware
            pass
