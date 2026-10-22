"""Recoverable CPU speech processes: a native DLL fault must not exit the HUD."""

from __future__ import annotations

import multiprocessing as mp
import threading
from typing import Any


class SpeechWorkerError(RuntimeError):
    pass


class SpeechWorkerClosedError(SpeechWorkerError):
    """The owner intentionally closed the worker; this is not a recoverable crash."""


def _backend(kind: str, settings: dict[str, Any]):
    if kind == "stt":
        from lora.voice.stt import FasterWhisperTranscriber
        return FasterWhisperTranscriber(**settings)
    if kind == "tts":
        from lora.voice.tts.worker import KokoroTtsWorkerProcess
        return KokoroTtsWorkerProcess(**settings)
    if kind == "vad":
        from lora.voice.vad import VoiceActivityDetector
        return VoiceActivityDetector(**settings)
    raise ValueError(f"Unknown speech backend: {kind}")


def _serve(connection, kind, settings, factory):
    import faulthandler
    faulthandler.enable()
    from lora.core.diagnostics import enable_fault_reporting
    from lora.core.paths import PROJECT_ROOT
    enable_fault_reporting(PROJECT_ROOT / "logs")
    try:
        engine = factory(kind, settings)
        while True:
            request = connection.recv()
            if request is None:
                return
            method, args, kwargs = request
            try:
                if method == "describe":
                    value = {"backend": kind, "model": getattr(engine, "model_size_or_path", kind), "device": "cpu"}
                elif method == "process_after_reset":
                    reset, chunk = args
                    if reset:
                        engine.reset()
                    value = engine.process_chunk(chunk)
                else:
                    value = getattr(engine, method)(*args, **kwargs)
                connection.send((True, value))
            except Exception as exc:
                connection.send((False, f"{type(exc).__name__}: {exc}"))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class SpeechProcess:
    """One serialized native backend; failed calls report an error, next call restarts."""

    def __init__(self, kind: str, settings: dict[str, Any], timeout: float = 60,
                 factory=_backend):
        self.kind, self.settings, self.timeout = kind, settings, timeout
        self.factory = factory
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._process = self._connection = None
        self._closed = False
        self.restarts = 0

    def _dispose(self):
        with self._lifecycle_lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if self._process is not None:
                if self._process.is_alive():
                    self._process.terminate()
                self._process.join(timeout=2)
                self._process.close()
                self._process = None

    def call(self, method, *args, **kwargs):
        with self._lock:
            # Publish startup atomically with respect to close(). The protocol
            # lock alone cannot do this because close must interrupt a recv.
            with self._lifecycle_lock:
                if self._closed:
                    raise SpeechWorkerClosedError(f"{self.kind} worker is closed")
                if self._process is None or not self._process.is_alive():
                    self._dispose()
                    context = mp.get_context("spawn")
                    parent, child = context.Pipe()
                    self._process = context.Process(target=_serve,
                        args=(child, self.kind, self.settings, self.factory), daemon=True,
                        name=f"lora-{self.kind}-cpu")
                    self._process.start()
                    child.close()
                    self._connection = parent
                    self.restarts += 1
            try:
                self._connection.send((method, args, kwargs))
                if not self._connection.poll(self.timeout):
                    raise TimeoutError(f"{self.kind} CPU worker exceeded {self.timeout:g}s")
                ok, value = self._connection.recv()
            except (EOFError, OSError, TimeoutError) as exc:
                code = self._process.exitcode if self._process is not None else None
                self._dispose()
                if self._closed:
                    raise SpeechWorkerClosedError(f"{self.kind} worker closed during {method}") from exc
                raise SpeechWorkerError(f"{self.kind} worker stopped (exit={code}): {exc}; next request will restart it") from exc
            if not ok:
                raise SpeechWorkerError(f"{self.kind}: {value}")
            return value

    def close(self):
        with self._lifecycle_lock:
            self._closed = True
            # Unblock a pending native call before acquiring its protocol lock.
            # Startup and disposal cannot replace this handle while it closes.
            process = self._process
            try:
                if process is not None and process.is_alive():
                    process.terminate()
            except (ValueError, OSError):
                pass  # A simultaneous completed call already retired this child.
        with self._lock:
            self._dispose()


class ProcessTranscriber:
    def __init__(self, **settings):
        self.worker = SpeechProcess("stt", settings, timeout=90)

    def preload(self):
        return self.worker.call("preload")

    def transcribe(self, audio, sample_rate=16000):
        return self.worker.call("transcribe", audio, sample_rate)

    def shutdown(self):
        self.worker.close()


class ProcessKokoro:
    def __init__(self, **settings):
        self.worker = SpeechProcess("tts", settings)

    def preload(self):
        return self.worker.call("preload")

    def diagnosis(self):
        return self.worker.call("diagnosis")

    def synthesize(self, text, **kwargs):
        return self.worker.call("synthesize", text, **kwargs)

    def shutdown(self):
        self.worker.close()


class ProcessVad:
    def __init__(self, **settings):
        self.worker = SpeechProcess("vad", settings, timeout=15)
        self._reset = threading.Event()

    def process_chunk(self, chunk):
        reset = self._reset.is_set()
        self._reset.clear()
        return self.worker.call("process_after_reset", reset, chunk)

    def get_and_reset_utterance(self):
        return self.worker.call("get_and_reset_utterance")

    def reset(self):
        # GUI callers never wait for a native model or a blocked worker.
        self._reset.set()

    def shutdown(self):
        self.worker.close()
