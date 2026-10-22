"""Cross-encoder reranker for refining candidate chunks."""

import logging
import os
from pathlib import Path
from typing import Any

from lora.core.paths import project_path
from lora.grounding.chunk import DocumentChunk

logger = logging.getLogger(__name__)

LOCAL_RERANKER_PATH = project_path("models/bge-reranker-base")
DEFAULT_RERANKER_MODEL = os.getenv(
    "LORA_RERANKER_MODEL",
    str(LOCAL_RERANKER_PATH.resolve()) if LOCAL_RERANKER_PATH.is_dir() else "BAAI/bge-reranker-base",
)


class CrossEncoderReranker:
    """Refines candidate retrieved chunks using a cross-encoder on CPU."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
    ) -> None:
        if model_name is None:
            self.model_name = (
                str(LOCAL_RERANKER_PATH.resolve())
                if LOCAL_RERANKER_PATH.is_dir()
                else DEFAULT_RERANKER_MODEL
            )
        else:
            self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._model_failed: bool = False

    def _load_model(self) -> Any:
        if self._model is None and not self._model_failed:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading cross-encoder reranker '%s' on %s", self.model_name, self.device)
                is_local = Path(self.model_name).is_dir()
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    local_files_only=is_local,
                )
            except Exception as e:
                logger.warning("Failed to load cross-encoder model '%s': %s. Using lexical fallback.", self.model_name, e)
                self._model_failed = True
        return self._model

    def _fallback_score(self, query: str, text: str) -> float:
        """Token-overlap heuristic fallback when cross-encoder weights are offline."""
        q_words = set(query.lower().split())
        if not q_words:
            return 0.0
        text_words = set(text.lower().split())
        overlap = len(q_words.intersection(text_words))
        return float(overlap) / float(len(q_words))

    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        """Score candidate chunks against the query and return the top-k highest scoring chunks."""
        if not chunks:
            return []
        if len(chunks) <= top_k:
            return chunks

        try:
            model = self._load_model()
        except Exception:
            model = None
        if model is not None:
            pairs = [[query, c.text] for c in chunks]
            try:
                scores = model.predict(pairs)
                ranked = sorted(
                    zip(chunks, scores, strict=False),
                    key=lambda item: item[1],
                    reverse=True,
                )
                return [item[0] for item in ranked[:top_k]]
            except Exception as e:
                logger.warning("Inference error in cross-encoder reranker: %s. Using fallback.", e)

        # Lexical overlap fallback
        scored = [
            (c, self._fallback_score(query, c.text))
            for c in chunks
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [item[0] for item in scored[:top_k]]
