"""In-memory ephemeral SessionVectorIndex using FAISS CPU + BM25 Reciprocal Rank Fusion."""

import logging
import math
import re
from collections import Counter
from typing import Any

import faiss

from lora.grounding.chunk import DocumentChunk
from lora.grounding.embed import EmbeddingEngine

logger = logging.getLogger(__name__)


def _tokenize_bm25(text: str) -> list[str]:
    return [w for w in re.sub(r"[^\w\s]", " ", (text or "").lower()).split() if len(w) > 1]


class SessionVectorIndex:
    """Ephemeral in-memory hybrid index (FAISS dense + BM25 lexical with RRF) for session articles/PDFs.

    Leaves 0 disk footprint, caps memory at `max_chunks`, and is cleanly cleared on session termination.
    """

    def __init__(
        self,
        session_id: str,
        embedding_engine: EmbeddingEngine,
        max_chunks: int = 256,
    ) -> None:
        self._session_id = session_id
        self._embedding_engine = embedding_engine
        self._max_chunks = max(16, max_chunks)
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
        """Embeds and indexes new document chunks in RAM (with automatic rolling cap)."""
        if not chunks:
            return

        # Enforce rolling session chunk cap so multi-hour desktop sessions never grow unbounded
        if len(self._chunks) + len(chunks) > self._max_chunks:
            keep_old = max(0, self._max_chunks - len(chunks))
            retained = self._chunks[-keep_old:] if keep_old > 0 else []
            self.clear()
            chunks = retained + list(chunks)[-self._max_chunks :]

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

    def _bm25_scores(self, query: str) -> list[float]:
        """Compute Okapi BM25 scores across in-memory chunks."""
        q_terms = _tokenize_bm25(query)
        n_docs = len(self._chunks)
        if not q_terms or n_docs == 0:
            return [0.0] * n_docs

        doc_tokens = [_tokenize_bm25(c.text) for c in self._chunks]
        doc_lens = [len(toks) for toks in doc_tokens]
        avgdl = sum(doc_lens) / max(1, n_docs)
        k1 = 1.5
        b = 0.75

        df: Counter[str] = Counter()
        for toks in doc_tokens:
            for term in set(toks):
                df[term] += 1

        scores = [0.0] * n_docs
        for i, toks in enumerate(doc_tokens):
            if not toks:
                continue
            tf = Counter(toks)
            dl = doc_lens[i]
            sc = 0.0
            for t in q_terms:
                if t not in tf:
                    continue
                n_q = df[t]
                idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
                freq = tf[t]
                denom = freq + k1 * (1.0 - b + b * (dl / max(1e-6, avgdl)))
                sc += idf * ((freq * (k1 + 1.0)) / denom)
            scores[i] = sc
        return scores

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

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> list[tuple[DocumentChunk, float]]:
        """Hybrid retrieval combining dense FAISS vectors and lexical BM25 via Reciprocal Rank Fusion (RRF)."""
        if not query or not query.strip() or self._index is None or not self._chunks:
            return []

        n_chunks = len(self._chunks)
        dense_hits = self.search(query, top_k=n_chunks, score_threshold=-1.0)
        dense_rank: dict[int, int] = {}
        dense_score_map: dict[int, float] = {}
        chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(self._chunks)}
        for rank, (chunk, d_sc) in enumerate(dense_hits, 1):
            c_idx = chunk_id_to_idx.get(chunk.chunk_id, -1)
            if c_idx >= 0:
                dense_rank[c_idx] = rank
                dense_score_map[c_idx] = d_sc

        bm25_raw = self._bm25_scores(query)
        bm25_order = sorted(range(n_chunks), key=lambda i: bm25_raw[i], reverse=True)
        bm25_rank = {doc_idx: rank for rank, doc_idx in enumerate(bm25_order, 1) if bm25_raw[doc_idx] > 0}

        fused: list[tuple[int, float]] = []
        for idx in range(n_chunks):
            r_dense = dense_rank.get(idx, n_chunks + 1)
            r_bm25 = bm25_rank.get(idx, n_chunks + 1)
            rrf_score = (1.0 / (rrf_k + r_dense)) + (1.0 / (rrf_k + r_bm25))
            # Blend RRF rank signal with dense cosine score for downstream threshold compatibility
            blended = 0.75 * dense_score_map.get(idx, 0.0) + 0.25 * min(1.0, rrf_score * (rrf_k / 2.0))
            fused.append((idx, float(blended)))

        fused.sort(key=lambda item: item[1], reverse=True)
        return [(self._chunks[idx], sc) for idx, sc in fused[: min(top_k, n_chunks)]]

    def clear(self) -> None:
        """Immediately release all indexed vectors and text chunks from memory."""
        self._chunks.clear()
        self._index = None
        self._dimension = None
        logger.info("SessionVectorIndex (%s) wiped from memory.", self._session_id)
