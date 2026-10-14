"""Client for the standalone voice server (see server_main.py)."""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
SERVER_SCRIPT = Path(__file__).with_name("server_main.py")


def find_tts_python() -> str | None:
    """Python of the separate voice environment: LORA_TTS_PYTHON, else <project>/.venv-tts."""
    env = os.environ.get("LORA_TTS_PYTHON")
    if env and Path(env).exists():
        return env
    for base in list(Path(__file__).resolve().parents)[:8]:
        for rel in (Path(".venv-tts") / "Scripts" / "python.exe", Path(".venv-tts") / "bin" / "python"):
            if (base / rel).exists():
                return str(base / rel)
    return None


class ExternalVoiceServer:
    def __init__(self, python_exe: str, models_dir: str, script: Path = SERVER_SCRIPT, env: dict[str, str] | None = None) -> None:
        self.python_exe, self.models_dir, self.script = python_exe, models_dir, script
        self.env = {**os.environ, **(env or {})}
        self.state = "unstarted"  # unstarted | ready | failed
        self.reason, self.fix, self.dir, self.sample_rate = "", "", "", 0
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self, timeout: float = 300.0) -> bool:
        if self.state in ("ready", "failed"):
            return self.state == "ready"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        try:
            self._proc = subprocess.Popen(
                [self.python_exe, str(self.script), "--models-dir", self.models_dir],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", bufsize=1, env=self.env, creationflags=flags,
            )
        except Exception as exc:
            return self._fail(f"could not start the voice environment ({exc})", r"run scripts\install_tts.ps1")
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        try:
            msg = self._lines.get(timeout=timeout)
        except queue.Empty:
            return self._fail("the voice environment took too long to load", "check the terminal for errors")
        if msg.get("event") == "ready":
            self.state, self.dir, self.sample_rate = "ready", str(msg.get("dir", "")), int(msg.get("sr", 24000))
            logger.info("Qwen3-TTS voice server ready (%s).", self.dir)
            return True
        return self._fail(str(msg.get("reason", "unknown error")), str(msg.get("fix", "")))

    def _fail(self, reason: str, fix: str) -> bool:
        self.state, self.reason, self.fix = "failed", reason, fix
        logger.warning("Qwen3-TTS voice server unavailable: %s. Fix: %s", reason, fix or "see the terminal")
        self.stop()
        return False

    def _read_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            try:
                self._lines.put(json.loads(line))
            except Exception:
                logger.info("tts-server: %s", line.strip())
        self._lines.put({"event": "closed"})

    def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            if line.strip():
                logger.debug("tts-server: %s", line.strip())

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            try:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------ synthesis
    def synthesize(self, text: str, voice: str = "vivian", timeout: float = 120.0) -> tuple[np.ndarray, int] | None:
        if self.state != "ready" or not self.alive:
            return None
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            try:
                self._proc.stdin.write(json.dumps({"id": rid, "text": text, "voice": voice}) + "\n")  # type: ignore[union-attr]
                self._proc.stdin.flush()  # type: ignore[union-attr]
            except Exception as exc:
                self._fail(f"lost the voice process ({exc})", "restart LORA")
                return None
            while True:
                try:
                    msg = self._lines.get(timeout=timeout)
                except queue.Empty:
                    logger.warning("Voice server timed out on %r", text[:40])
                    return None
                if msg.get("event") == "closed":
                    self._fail("the voice process exited", "restart LORA")
                    return None
                if msg.get("id") != rid:
                    continue
                if not msg.get("ok"):
                    logger.warning("Voice server could not speak: %s", msg.get("error"))
                    return None
                return self._load(str(msg["path"]), int(msg.get("sr", self.sample_rate)))

    @staticmethod
    def _load(path: str, sr: int) -> tuple[np.ndarray, int] | None:
        try:
            with wave.open(path, "rb") as w:
                raw = w.readframes(w.getnframes())
                sr = w.getframerate() or sr
            return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, sr
        except Exception as exc:
            logger.warning("Could not read the voice audio: %s", exc)
            return None
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
