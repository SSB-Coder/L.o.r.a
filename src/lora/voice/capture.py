"""Audio capture and ring buffer for L.O.R.A Voice Pipeline."""

import logging
import math
import queue
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AudioBuffer:
    """Fixed-size NumPy ring; the audio callback never allocates Python floats."""

    def __init__(self, max_seconds: float = 10.0, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.max_samples = max(1, int(max_seconds * sample_rate))
        self._buffer = np.empty(self.max_samples, dtype=np.float32)
        self._position = 0
        self._count = 0
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> None:
        data = np.asarray(samples, dtype=np.float32).reshape(-1)[-self.max_samples:]
        if not len(data):
            return
        with self._lock:
            first = min(len(data), self.max_samples - self._position)
            self._buffer[self._position:self._position + first] = data[:first]
            remaining = len(data) - first
            if remaining:
                self._buffer[:remaining] = data[first:]
            self._position = (self._position + len(data)) % self.max_samples
            self._count = min(self.max_samples, self._count + len(data))

    def _recent(self, take: int) -> np.ndarray:
        start = (self._position - take) % self.max_samples
        if not take:
            return np.zeros(0, dtype=np.float32)
        if start + take <= self.max_samples:
            return self._buffer[start:start + take].copy()
        return np.concatenate((self._buffer[start:], self._buffer[:take - (self.max_samples - start)]))

    def read_last_n_seconds(self, seconds: float) -> np.ndarray:
        with self._lock:
            return self._recent(min(self._count, max(0, int(seconds * self.sample_rate))))

    def read_all(self) -> np.ndarray:
        with self._lock:
            result = self._recent(self._count)
            self._count = self._position = 0
            return result

    def clear(self) -> None:
        with self._lock:
            self._count = self._position = 0

    @property
    def duration_seconds(self) -> float:
        with self._lock:
            return self._count / self.sample_rate


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
        self._lifecycle_lock = threading.RLock()
        self._pending: queue.Queue = queue.Queue(maxsize=64)  # at most ~2 seconds
        self._consumer: threading.Thread | None = None
        self._consumer_stop = threading.Event()
        self._generation = 0
        self.dropped_blocks = 0

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
        """Copy and enqueue only. Native inference must never run on PortAudio."""
        if not self.is_recording:
            return
        samples = indata[:, 0] if indata.ndim > 1 else indata
        self.blocks_received += 1
        self._last_block_time = time.monotonic()
        item = (self._generation, samples.astype(np.float32, copy=True))
        try:
            self._pending.put_nowait(item)
        except queue.Full:
            self.dropped_blocks += 1
            try:
                self._pending.get_nowait()
                self._pending.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def _consume_audio(self) -> None:
        while not self._consumer_stop.is_set():
            try:
                generation, samples = self._pending.get(timeout=.1)
            except queue.Empty:
                continue
            if generation != self._generation or not self.is_recording:
                continue
            if self._resample is not None:
                samples = self._resample(samples).astype(np.float32)
            self.buffer.write(samples)
            self._dispatch(samples)

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
        with self._lifecycle_lock:
            return self._start()

    def _start(self) -> bool:
        """Open the default (or configured) microphone. Returns True if a stream is running."""
        if self.is_streaming:
            return True
        self.is_recording = True
        self._generation += 1
        self._consumer_stop.clear()
        if self._consumer is None or not self._consumer.is_alive():
            self._consumer = threading.Thread(target=self._consume_audio, daemon=True, name="lora-mic-consumer")
            self._consumer.start()
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
            stream = None
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
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        logger.debug("Failed input stream cleanup", exc_info=True)
                self.last_error = f"{self.device_name}: {type(exc).__name__}: {exc}"
                logger.warning("Could not open microphone at %d Hz: %s", rate, self.last_error)

        self._stream = None
        self._resample = None
        return False

    def stop(self) -> None:
        """Stop capturing audio."""
        with self._lifecycle_lock:
            self._stop()

    def _stop(self) -> None:
        self.is_recording = False
        self._generation += 1
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        while True:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        self.stop()
        self._consumer_stop.set()
        if self._consumer is not None and self._consumer is not threading.current_thread():
            self._consumer.join(timeout=2)
