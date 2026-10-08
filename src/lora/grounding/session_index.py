"""In-memory ephemeral SessionVectorIndex using FAISS CPU."""

import logging
from typing import Any

import faiss

from lora.grounding.chunk import DocumentChunk
from lora.grounding.embed import EmbeddingEngine

logger = logging.getLogger(__name__)


class SessionVectorIndex:
    """Ephemeral in-memory vector index for web articles and PDFs browsed during a session.

    Leaves 0 disk footprint and is cleanly cleared on session termination.
    """

    def __init__(self, session_id: str, embedding_engine: EmbeddingEngine) -> None:
        self._session_id = session_id
        self._embedding_engine = embedding_engine
        self._chunks: list[DocumentChunk] = []
        self._index: Any = None
        self._dimension: int | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _init_index(self, dim: int) -> None:
        self._dimension = dim
        # IndexFlatIP calculates inner product, which equals cosine similarity for normalized vectors
        self._index = faiss.IndexFlatIP(dim)

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Embeds and indexes new document chunks in RAM."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self._embedding_engine.encode(texts)
        if len(embeddings) == 0:
            return

        dim = embeddings.shape[1]
        if self._index is None:
            self._init_index(dim)
        elif self._dimension != dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, got {dim}"
            )

        self._index.add(embeddings)
        self._chunks.extend(chunks)
        logger.debug(
            "SessionVectorIndex (%s) added %d chunks; total: %d",
            self._session_id,
            len(chunks),
            len(self._chunks),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search in-memory chunks for top-k semantically relevant passages."""
        if not query or not query.strip() or self._index is None or not self._chunks:
            return []

        q_vec = self._embedding_engine.encode([query.strip()])
        if len(q_vec) == 0:
            return []

        actual_k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(q_vec, actual_k)

        results: list[tuple[DocumentChunk, float]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._chunks):
                continue
            f_score = float(score)
            if f_score >= score_threshold:
                results.append((self._chunks[idx], f_score))

        return results

    def clear(self) -> None:
        """Immediately release all indexed vectors and text chunks from memory."""
        self._chunks.clear()
        self._index = None
        self._dimension = None
        logger.info("SessionVectorIndex (%s) wiped from memory.", self._session_id)
