"""Embedding engine running on CPU for zero VRAM consumption."""

import functools
import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _serialized(method):
    @functools.wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return locked


def _resolve_default_embedding_path() -> str:
    env_path = os.getenv("LORA_EMBEDDING_MODEL_PATH")
    if env_path and Path(env_path).exists():
        return str(Path(env_path).resolve())
    repo_root = Path(__file__).resolve().parents[3]
    for cand in (
        Path("models/embeddinggemma2"),
        Path("models/embeddinggemma-2"),
        Path("models/embedding-gemma-2"),
        repo_root / "models" / "embeddinggemma2",
        repo_root / "models" / "embeddinggemma-2",
        repo_root / "models" / "embedding-gemma-2",
        Path(r"C:\Models\embeddinggemma2"),
        Path(r"C:\Models\embeddinggemma-2"),
    ):
        if cand.exists():
            return str(cand.resolve())
    return env_path or str(Path("models/embeddinggemma2").resolve())


DEFAULT_MODEL_PATH = _resolve_default_embedding_path()


class EmbeddingEngine:
    """Computes semantic vector embeddings using local sentence-transformers models on CPU."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 8,
    ) -> None:
        resolved = model_name_or_path or _resolve_default_embedding_path()
        # If caller passed legacy "C:/Models/embeddinggemma-2" that doesn't exist, resolve to local models/embeddinggemma2
        if not Path(resolved).exists():
            alt = _resolve_default_embedding_path()
            if Path(alt).exists():
                resolved = alt
        self.model_path = resolved
        self._lock = threading.RLock()
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None
        self._use_fallback: bool = False
        self._fallback_dim: int = 768
        self._cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self.last_encode_stats: dict[str, int | float] = {}

    @_serialized
    def _ensure_model(self) -> Any:
        if self._model is None and not self._use_fallback:
            path = Path(self.model_path)
            if not path.exists():
                alt = _resolve_default_embedding_path()
                if Path(alt).exists():
                    self.model_path = alt
                    path = Path(alt)
                else:
                    logger.warning(
                        "Embedding model path '%s' does not exist. Falling back to CPU embedder.",
                        self.model_path,
                    )
            try:
                import torch
                from sentence_transformers import SentenceTransformer

                torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

                logger.info("Loading embedding model from '%s' on %s", self.model_path, self.device)
                kwargs = {"trust_remote_code": True}
                if Path(self.model_path).exists():
                    kwargs["local_files_only"] = True
                self._model = SentenceTransformer(
                    str(self.model_path),
                    device=self.device,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(
                    "Failed to load sentence-transformer model '%s' (%s). Using fallback CPU embeddings.",
                    self.model_path,
                    e,
                )
                self._use_fallback = True
        return self._model

    @_serialized
    def encode(self, texts: list[str]) -> np.ndarray:
        """Generate normalized float32 embeddings for a list of strings."""
        dim = self.dimension
        if not texts:
            return np.empty((0, dim), dtype=np.float32)

        model = self._ensure_model()
        if self._use_fallback or model is None:
            vecs = []
            for t in texts:
                vec = np.zeros(dim, dtype=np.float32)
                for i, c in enumerate(t[:dim]):
                    vec[i % dim] += (ord(c) % 13) + 1.0
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
                vecs.append(vec)
            return np.asarray(vecs, dtype=np.float32)

        started = time.perf_counter()
        keys = [hashlib.sha256(t.encode("utf-8")).digest() for t in texts]
        missing = dict((key, text) for key, text in zip(keys, texts, strict=True) if key not in self._cache)
        fresh = model.encode(
            list(missing.values()),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ) if missing else []
        computed = {key: np.asarray(vector, dtype=np.float32) for key, vector in zip(missing, fresh, strict=True)}
        # Gather before evicting so a large batch cannot evict its own output.
        output = np.asarray([computed[key] if key in computed else self._cache[key] for key in keys], dtype=np.float32)
        for key in keys:
            self._cache[key] = computed[key] if key in computed else self._cache[key]
            self._cache.move_to_end(key)
        while len(self._cache) > 256:
            self._cache.popitem(last=False)
        self.last_encode_stats = {"texts": len(texts), "encoded": len(missing), "cached": len(texts) - len(missing),
                                  "seconds": round(time.perf_counter() - started, 4)}
        return output.copy()

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        model = self._ensure_model()
        if self._use_fallback or model is None:
            return self._fallback_dim
        try:
            fn = getattr(model, "get_embedding_dimension", None) or getattr(model, "get_sentence_embedding_dimension", None)
            if fn is not None:
                return int(fn())
            return self._fallback_dim
        except Exception:
            return self._fallback_dim
