"""Standalone Qwen3-TTS voice server. Runs in its OWN Python environment (.venv-tts) with GPU PyTorch.

LORA's main environment (managed by uv) ships the CPU build of PyTorch and `uv run` would undo any GPU build
installed there, so the natural voice lives in a separate environment and is driven over stdin/stdout.

Protocol (one JSON object per line):
  server -> {"event": "ready", "dir": "...", "sr": 24000}  or  {"event": "error", "reason": "...", "fix": "..."}
  client -> {"id": 1, "text": "Hello.", "voice": "vivian"}
  server -> {"id": 1, "ok": true, "path": "C:/.../lora_tts_x.wav", "sr": 24000}  (16-bit mono WAV; client deletes it)
  client -> {"cmd": "quit"}
This file imports nothing from `lora` on purpose. LORA_TTS_FAKE=1 swaps the model for a test tone (used by tests).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path


def _patch_qwen_tts_rope() -> None:
    """Shim Qwen3TTSTalkerRotaryEmbedding so self.config always exists across all transformers versions."""
    try:
        import sys
        import types
        for mod in list(sys.modules.values()):
            cls = getattr(mod, "Qwen3TTSTalkerRotaryEmbedding", None)
            if cls is not None and not getattr(cls, "_lora_patched", False):
                orig_init = cls.__init__
                def _wrapped_init(self, *args, **kwargs):
                    self.__dict__["config"] = kwargs.get("config") or types.SimpleNamespace(
                        rope_theta=kwargs.get("base", 10000.0),
                        partial_rotary_factor=1.0,
                        head_dim=kwargs.get("dim", 64),
                        max_position_embeddings=kwargs.get("max_position_embeddings", 2048),
                        rope_scaling=None,
                    )
                    return orig_init(self, *args, **kwargs)
                cls.__init__ = _wrapped_init
                cls._lora_patched = True
    except Exception:
        pass

SMALL_MODEL_NAME = "Qwen3-TTS-12Hz-0.6B-CustomVoice"
FAKE_SR = 22050


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def write_wav(samples: list[float], sr: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="lora_tts_")
    os.close(fd)
    frames = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)
    return path


def has_weights(p: Path) -> bool:
    return p.is_dir() and ((p / "model.safetensors").exists() or (p / "model.safetensors.index.json").exists() or any(p.glob("model-*.safetensors")))


def weights_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.glob("*.safetensors")) / 1e9


def need_gb(p: Path) -> float:
    return weights_gb(p) * 1.02 + 0.25


def candidate_dirs(models_dir: str) -> list[Path]:
    cands = []
    if os.environ.get("LORA_TTS_MODEL_DIR"):
        cands.append(Path(os.environ["LORA_TTS_MODEL_DIR"]))
    base = Path(models_dir)
    cands += [base, base.parent / SMALL_MODEL_NAME]
    seen, out = set(), []
    for c in cands:
        if str(c).lower() not in seen and has_weights(c):
            seen.add(str(c).lower())
            out.append(c)
    return out


class FakeModel:
    """A tone whose length follows the text, so the client path can be tested without a GPU."""

    sr = FAKE_SR

    def generate(self, text: str, voice: str) -> tuple[list[float], int]:
        n = int(self.sr * min(2.0, 0.05 * max(1, len(text))))
        return [0.2 * math.sin(2 * math.pi * 440 * i / self.sr) for i in range(n)], self.sr


class RealModel:
    def __init__(self, model, np) -> None:
        self.model, self.np = model, np

    def _speaker(self, voice: str) -> str:
        try:
            sup = self.model.get_supported_speakers()
            names = list(sup.keys()) if isinstance(sup, dict) else list(sup)
            lookup = {n.lower(): n for n in names}
            if voice.lower() in lookup:
                return lookup[voice.lower()]
            if names:
                return lookup.get("vivian", names[0])
        except Exception:
            pass
        return "_".join(part.capitalize() for part in voice.split("_"))

    def generate(self, text: str, voice: str) -> tuple[list[float], int]:
        wavs, sr = self.model.generate_custom_voice(text=text, language="English", speaker=self._speaker(voice))
        wav = wavs[0]
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().float().numpy()
        return self.np.asarray(wav, dtype="float32").reshape(-1).tolist(), int(sr)


def load_model(models_dir: str):
    """Returns (model, dir, sr) or raises RuntimeError(reason, fix)."""
    if os.environ.get("LORA_TTS_FAKE") == "1":
        return FakeModel(), "fake", FAKE_SR
    try:
        import numpy as np
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch is missing in the voice environment ({exc})|run scripts\\install_tts.ps1 again") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(f"this PyTorch ({torch.__version__}) has no GPU support|run scripts\\install_tts.ps1 again, or try -Cuda cu126")
    try:
        from qwen_tts import Qwen3TTSModel
        _patch_qwen_tts_rope()
    except Exception as exc:
        raise RuntimeError(f"qwen-tts is missing in the voice environment ({exc})|run scripts\\install_tts.ps1 again") from exc
    cands = candidate_dirs(models_dir)
    if not cands:
        raise RuntimeError(f"no Qwen3-TTS model files found in {models_dir}|download the model there (the smaller {SMALL_MODEL_NAME} also works)")
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    fitting = [c for c in cands if need_gb(c) <= free_gb]
    if not fitting:
        need = min(need_gb(c) for c in cands)
        raise RuntimeError(f"not enough free GPU memory ({free_gb:.1f} GB free, {need:.1f} GB needed)|download the smaller {SMALL_MODEL_NAME} next to the other model or close other GPU programs")
    chosen = max(fitting, key=weights_gb)
    try:
        try:
            model = Qwen3TTSModel.from_pretrained(str(chosen), device_map="cuda:0", dtype=torch.bfloat16)
        except TypeError:
            model = Qwen3TTSModel.from_pretrained(str(chosen), device_map="cuda:0", torch_dtype=torch.bfloat16)
    except Exception as exc:
        torch.cuda.empty_cache()
        oom = "out of memory" in str(exc).lower()
        raise RuntimeError(
            ("ran out of GPU memory while loading" if oom else f"loading failed ({type(exc).__name__}: {exc})")
            + f"|download the smaller {SMALL_MODEL_NAME} or close other GPU programs"
        ) from exc
    return RealModel(model, np), str(chosen), 24000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=r"C:\Models\Qwen3-TTS-12Hz-1.7B-CustomVoice")
    args = ap.parse_args()
    try:
        model, chosen, sr = load_model(args.models_dir)
    except RuntimeError as exc:
        reason, _, fix = str(exc).partition("|")
        emit({"event": "error", "reason": reason, "fix": fix})
        return 1
    emit({"event": "ready", "dir": chosen, "sr": sr})
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except Exception:
            continue
        if req.get("cmd") == "quit":
            break
        rid = req.get("id")
        try:
            samples, out_sr = model.generate(str(req.get("text", "")), str(req.get("voice", "vivian")))
            emit({"id": rid, "ok": True, "path": write_wav(samples, out_sr), "sr": out_sr})
        except Exception as exc:
            emit({"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
