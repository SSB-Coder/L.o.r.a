"""Embedding engine running on CPU for zero VRAM consumption."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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
        batch_size: int = 16,
    ) -> None:
        resolved = model_name_or_path or _resolve_default_embedding_path()
        # If caller passed legacy "C:/Models/embeddinggemma-2" that doesn't exist, resolve to local models/embeddinggemma2
        if not Path(resolved).exists():
            alt = _resolve_default_embedding_path()
            if Path(alt).exists():
                resolved = alt
        self.model_path = resolved
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None
        self._use_fallback: bool = False
        self._fallback_dim: int = 768

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
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model from '%s' on %s", self.model_path, self.device)
                self._model = SentenceTransformer(
                    str(self.model_path),
                    device=self.device,
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.warning(
                    "Failed to load sentence-transformer model '%s' (%s). Using fallback CPU embeddings.",
                    self.model_path,
                    e,
                )
                self._use_fallback = True
        return self._model

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

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        model = self._ensure_model()
        if self._use_fallback or model is None:
            return self._fallback_dim
        try:
            return int(model.get_sentence_embedding_dimension())
        except Exception:
            return self._fallback_dim
