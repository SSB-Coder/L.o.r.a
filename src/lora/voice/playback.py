"""Interruptible output without stopping or reopening the microphone."""

import os
import tempfile
import time
import wave
from pathlib import Path

import numpy as np


def play_audio(samples, sample_rate, stop_event, on_level, exiting):
    samples = np.asarray(samples, dtype=np.float32)
    if not len(samples) or stop_event.is_set() or exiting.is_set():
        return
    if os.name == "nt":
        import winsound

        # Python explicitly rejects SND_MEMORY | SND_ASYNC. Keep a private
        # temporary WAV alive for the supported asynchronous filename API.
        with tempfile.TemporaryDirectory(prefix="lora-output-") as directory:
            path = Path(directory) / "speech.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
            try:
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
                started = time.monotonic()
                duration = len(samples) / sample_rate
                while time.monotonic() - started < duration:
                    if stop_event.is_set() or exiting.is_set():
                        break
                    offset = int((time.monotonic() - started) * sample_rate)
                    block = samples[offset:offset + max(1, sample_rate // 25)]
                    on_level(float(np.sqrt(np.mean(block ** 2)))) if len(block) else None
                    stop_event.wait(.04)
            finally:
                winsound.PlaySound(None, 0)
    else:
        import sounddevice as sd
        # Explicit output ownership avoids sd.play()/sd.stop() global state.
        with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as output:
            for start in range(0, len(samples), max(1, sample_rate // 25)):
                if stop_event.is_set() or exiting.is_set():
                    break
                block = samples[start:start + max(1, sample_rate // 25)]
                output.write(block.reshape(-1, 1))
                on_level(float(np.sqrt(np.mean(block ** 2))))
