"""Audio capture and ring buffer for L.O.R.A Voice Pipeline."""

import logging
import math
import threading
import time
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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
    """Manages microphone input streaming into an AudioBuffer.

    ``start()`` used to swallow every error, so the UI could say "listening" with
    no microphone open. It now records what happened (``last_error``,
    ``device_name``, ``is_streaming``) and returns whether a stream really opened.
    ``is_recording`` keeps its old meaning: start() was called and not yet stopped.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,  # 32ms at 16kHz
        buffer_seconds: float = 10.0,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = device
        self.buffer = AudioBuffer(max_seconds=buffer_seconds, sample_rate=sample_rate)
        self.is_recording = False
        self.device_name = ""
        self.native_rate = sample_rate
        self.last_error: str | None = None
        self.blocks_received = 0
        self._last_block_time = 0.0
        self._callback_errors = 0
        self._stream: Any = None
        self._callbacks: list[Any] = []
        self._resample: Any = None

    # ----- status -----
    @property
    def is_streaming(self) -> bool:
        """True only if a real input stream is open."""
        return self._stream is not None

    def seconds_since_audio(self) -> float | None:
        """Seconds since the last audio block arrived, or None if none ever did."""
        if self._last_block_time == 0.0:
            return None
        return time.monotonic() - self._last_block_time

    def status_text(self) -> str:
        if self.is_streaming:
            return self.device_name or "microphone"
        return self.last_error or "microphone not started"

    def register_chunk_callback(self, callback: Any) -> None:
        """Register a callback that receives every new audio chunk (np.ndarray)."""
        self._callbacks.append(callback)

    def _dispatch(self, samples: np.ndarray) -> None:
        for cb in self._callbacks:
            try:
                cb(samples)
            except Exception:
                self._callback_errors += 1
                if self._callback_errors <= 3:
                    logger.exception("Audio chunk callback failed")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Callback invoked by sounddevice for each audio block."""
        if status:
            logger.debug("Audio stream status: %s", status)
        if not self.is_recording:
            return
        samples = indata[:, 0] if indata.ndim > 1 else indata
        norm_samples = samples.astype(np.float32)
        if self._resample is not None:
            norm_samples = self._resample(norm_samples).astype(np.float32)
        self.blocks_received += 1
        self._last_block_time = time.monotonic()
        self.buffer.write(norm_samples)
        self._dispatch(norm_samples)

    def feed_samples(self, samples: np.ndarray) -> None:
        """Feed synthetic or file audio samples directly (used in tests or offline pipeline)."""
        norm_samples = samples.flatten().astype(np.float32)
        self.buffer.write(norm_samples)
        self._dispatch(norm_samples)

    def _make_resampler(self, native_rate: int) -> Any:
        from scipy.signal import resample_poly

        g = math.gcd(self.sample_rate, native_rate)
        up, down = self.sample_rate // g, native_rate // g
        return lambda x: resample_poly(x, up, down)

    def start(self) -> bool:
        """Open the default (or configured) microphone. Returns True if a stream is running."""
        if self.is_streaming:
            return True
        self.is_recording = True
        self.last_error = None
        try:
            import sounddevice as sd
        except Exception as exc:
            self.last_error = f"sounddevice unavailable ({type(exc).__name__}: {exc})"
            logger.warning("Microphone not started: %s", self.last_error)
            return False

        try:
            info = sd.query_devices(self.device, "input")
        except Exception as exc:
            self.last_error = f"no input device found ({type(exc).__name__}: {exc})"
            logger.warning("Microphone not started: %s", self.last_error)
            return False

        self.device_name = str(info.get("name", "microphone"))
        native = int(info.get("default_samplerate") or self.sample_rate)
        attempts: list[tuple[int, int]] = [(self.sample_rate, self.chunk_size)]
        if native != self.sample_rate:
            attempts.append((native, max(1, round(self.chunk_size * native / self.sample_rate))))

        for rate, block in attempts:
            try:
                stream = sd.InputStream(
                    device=self.device,
                    samplerate=rate,
                    channels=1,
                    blocksize=block,
                    dtype="float32",
                    callback=self._audio_callback,
                )
                self._resample = self._make_resampler(rate) if rate != self.sample_rate else None
                self.native_rate = rate
                stream.start()
                self._stream = stream
                logger.info(
                    "Microphone open: %s at %d Hz%s",
                    self.device_name,
                    rate,
                    f" (resampled to {self.sample_rate})" if self._resample else "",
                )
                return True
            except Exception as exc:
                self.last_error = f"{self.device_name}: {type(exc).__name__}: {exc}"
                logger.warning("Could not open microphone at %d Hz: %s", rate, self.last_error)

        self._stream = None
        self._resample = None
        return False

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
