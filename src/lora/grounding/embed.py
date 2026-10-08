"""Embedding engine running on CPU for zero VRAM consumption."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.getenv("LORA_EMBEDDING_MODEL_PATH", "C:/Models/embeddinggemma-2")


class EmbeddingEngine:
    """Computes semantic vector embeddings using local sentence-transformers models on CPU."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_PATH,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self.model_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            path = Path(self.model_path)
            if not path.exists():
                logger.warning(
                    "Embedding model path '%s' does not exist. Falling back to default or mock.",
                    self.model_path,
                )
            logger.info("Loading embedding model from '%s' on %s", self.model_path, self.device)
            self._model = SentenceTransformer(str(self.model_path), device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Generate normalized float32 embeddings for a list of strings."""
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        model = self._ensure_model()
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
        return int(model.get_sentence_embedding_dimension())
