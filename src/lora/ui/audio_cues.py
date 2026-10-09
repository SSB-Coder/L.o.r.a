import io
import logging
import math
import struct
import sys
import threading
import wave
from enum import StrEnum

logger = logging.getLogger(__name__)


class SoundCue(StrEnum):
    WAKE = "wake"
    SLEEP = "sleep"
    COMPLETE = "complete"
    ERROR = "error"


class AudioCueManager:
    """Generates procedural harmonic sine-wave earcons and plays non-blocking acoustic cues."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._cache: dict[SoundCue, bytes] = {}

    def generate_cue_wav(self, cue: SoundCue) -> bytes:
        """Generates raw in-memory 16-bit PCM WAV audio for the specified cue."""
        if cue in self._cache:
            return self._cache[cue]

        sample_rate = 44100
        samples: list[float] = []

        if cue == SoundCue.WAKE:
            # Rising harmonic chime (440Hz -> 880Hz, 0.22s)
            duration = 0.22
            total_samples = int(sample_rate * duration)
            for i in range(total_samples):
                t = i / sample_rate
                progress = i / total_samples
                freq = 440.0 + (440.0 * progress)
                env = math.sin(math.pi * progress)  # smooth bell envelope
                val = env * (0.7 * math.sin(2 * math.pi * freq * t) + 0.3 * math.sin(4 * math.pi * freq * t))
                samples.append(val)

        elif cue == SoundCue.SLEEP:
            # Falling soft tone (660Hz -> 330Hz, 0.25s)
            duration = 0.25
            total_samples = int(sample_rate * duration)
            for i in range(total_samples):
                t = i / sample_rate
                progress = i / total_samples
                freq = 660.0 - (330.0 * progress)
                env = (1.0 - progress) * 0.8  # linear decay
                val = env * math.sin(2 * math.pi * freq * t)
                samples.append(val)

        elif cue == SoundCue.COMPLETE:
            # Pleasant two-tone chord (C5 523Hz + E5 659Hz, 0.28s)
            duration = 0.28
            total_samples = int(sample_rate * duration)
            for i in range(total_samples):
                t = i / sample_rate
                progress = i / total_samples
                env = math.sin(math.pi * math.sqrt(progress)) * (1.0 - progress)
                tone1 = math.sin(2 * math.pi * 523.25 * t)
                tone2 = math.sin(2 * math.pi * 659.25 * t)
                val = env * 0.5 * (tone1 + tone2)
                samples.append(val)

        else:  # SoundCue.ERROR
            # Low double pulse (220Hz, 0.20s)
            duration = 0.20
            total_samples = int(sample_rate * duration)
            for i in range(total_samples):
                t = i / sample_rate
                progress = i / total_samples
                pulse = math.sin(4 * math.pi * progress)  # 2 pulses
                env = max(0.0, pulse) * (1.0 - progress)
                val = env * 0.6 * math.sin(2 * math.pi * 220.0 * t)
                samples.append(val)

        # Convert float samples (-1.0 to 1.0) to 16-bit PCM bytes
        pcm_data = bytearray()
        for s in samples:
            clamped = max(-1.0, min(1.0, s))
            int_val = int(clamped * 32767.0)
            pcm_data.extend(struct.pack("<h", int_val))

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)

        wav_bytes = buffer.getvalue()
        self._cache[cue] = wav_bytes
        return wav_bytes

    def play_cue(self, cue: SoundCue) -> None:
        """Plays the audio cue in a background thread without blocking the UI."""
        if not self.enabled:
            return

        def _play() -> None:
            try:
                wav_bytes = self.generate_cue_wav(cue)
                if sys.platform == "win32":
                    import winsound

                    winsound.PlaySound(wav_bytes, winsound.SND_MEMORY | winsound.SND_ASYNC)
                else:
                    logger.debug("Sound playback on non-Windows skipped for cue: %s", cue)
            except Exception as e:
                logger.debug("Audio cue playback exception (ignored): %s", e)

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()
