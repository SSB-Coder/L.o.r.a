"""Audio capture and ring buffer for L.O.R.A Voice Pipeline."""

import threading
from collections import deque
from typing import Any

import numpy as np


class AudioBuffer:
    """Thread-safe circular audio buffer for incoming microphone PCM frames."""

    def __init__(self, max_seconds: float = 10.0, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self._buffer: deque[float] = deque(maxlen=self.max_samples)
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> None:
        """Append 1D audio samples (float32, normalized between -1.0 and 1.0) into the buffer."""
        flattened = samples.flatten().astype(np.float32)
        with self._lock:
            self._buffer.extend(flattened.tolist())

    def read_last_n_seconds(self, seconds: float) -> np.ndarray:
        """Extract the most recent `seconds` worth of audio as float32 array."""
        num_samples = int(seconds * self.sample_rate)
        with self._lock:
            buf_len = len(self._buffer)
            if buf_len == 0:
                return np.zeros(0, dtype=np.float32)
            take = min(num_samples, buf_len)
            # Slice the last `take` elements from deque
            # Using list comprehension or slice from list
            recent = list(self._buffer)[-take:]
            return np.array(recent, dtype=np.float32)

    def read_all(self) -> np.ndarray:
        """Retrieve all currently buffered audio and clear the buffer."""
        with self._lock:
            arr = np.array(list(self._buffer), dtype=np.float32)
            self._buffer.clear()
            return arr

    def clear(self) -> None:
        """Clear all stored audio in the buffer."""
        with self._lock:
            self._buffer.clear()

    @property
    def duration_seconds(self) -> float:
        """Current duration of buffered audio in seconds."""
        with self._lock:
            return len(self._buffer) / self.sample_rate


class AudioCapture:
    """Manages microphone input streaming into an AudioBuffer."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,  # 32ms at 16kHz
        buffer_seconds: float = 10.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer = AudioBuffer(max_seconds=buffer_seconds, sample_rate=sample_rate)
        self.is_recording = False
        self._stream: Any = None
        self._callbacks: list[Any] = []

    def register_chunk_callback(self, callback: Any) -> None:
        """Register a callback that receives every new audio chunk (np.ndarray)."""
        self._callbacks.append(callback)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Callback invoked by sounddevice for each audio block."""
        if not self.is_recording:
            return
        # Flatten and normalize
        samples = indata[:, 0] if indata.ndim > 1 else indata
        norm_samples = samples.astype(np.float32)
        self.buffer.write(norm_samples)
        for cb in self._callbacks:
            try:
                cb(norm_samples)
            except Exception:
                pass

    def feed_samples(self, samples: np.ndarray) -> None:
        """Feed synthetic or file audio samples directly (used in tests or offline pipeline)."""
        norm_samples = samples.flatten().astype(np.float32)
        self.buffer.write(norm_samples)
        for cb in self._callbacks:
            try:
                cb(norm_samples)
            except Exception:
                pass

    def start(self) -> None:
        """Start capturing audio from default microphone."""
        if self.is_recording:
            return
        self.is_recording = True
        try:
            import sounddevice as sd
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception:
            # When running headless or without mic hardware, stream remains None
            # feed_samples can still be used for testing
            self._stream = None

    def stop(self) -> None:
        """Stop capturing audio."""
        self.is_recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
