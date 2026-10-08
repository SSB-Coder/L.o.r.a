"""Unit tests for CPU EmbeddingEngine and ephemeral SessionVectorIndex."""

import numpy as np

from lora.grounding.chunk import DocumentChunk
from lora.grounding.embed import EmbeddingEngine
from lora.grounding.session_index import SessionVectorIndex


class MockEmbeddingEngine(EmbeddingEngine):
    """Deterministic mock embedding engine for fast unit tests without loading 1.5GB weights."""

    def __init__(self, dimension: int = 64):
        self._dim = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            # Deterministic hash-based normalized vector
            vec = np.zeros(self._dim, dtype=np.float32)
            for i, char in enumerate(t[: self._dim]):
                vec[i] = ord(char) % 10 + 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def test_session_vector_index_add_and_search():
    engine = MockEmbeddingEngine(dimension=32)
    index = SessionVectorIndex(session_id="test-session-1", embedding_engine=engine)

    chunk1 = DocumentChunk(
        chunk_id="c1",
        source_url="https://example.com/transformer",
        title="Attention Paper",
        text="Self-attention mechanisms allow parallel processing of sequence tokens.",
        paragraph_index=0,
        page_number=1,
        token_count=12,
        tier=1,
        content_hash="hash1",
    )
    chunk2 = DocumentChunk(
        chunk_id="c2",
        source_url="https://example.com/cooking",
        title="Recipe Guide",
        text="Baking sourdough bread requires water, flour, salt, and wild yeast starter.",
        paragraph_index=0,
        page_number=1,
        token_count=14,
        tier=3,
        content_hash="hash2",
    )

    index.add_chunks([chunk1, chunk2])
    assert index.chunk_count == 2
    assert index.session_id == "test-session-1"

    results = index.search(query="Self-attention mechanisms", top_k=2)
    assert len(results) == 2
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "c1"
    assert score > 0.0


def test_session_vector_index_clear():
    engine = MockEmbeddingEngine(dimension=32)
    index = SessionVectorIndex(session_id="test-session-2", embedding_engine=engine)

    chunk = DocumentChunk(
        chunk_id="c1",
        source_url="https://example.com",
        title="Title",
        text="Sample text content for indexing.",
        paragraph_index=0,
        page_number=1,
        token_count=6,
        tier=2,
        content_hash="hash1",
    )
    index.add_chunks([chunk])
    assert index.chunk_count == 1

    index.clear()
    assert index.chunk_count == 0
    results = index.search(query="Sample text", top_k=5)
    assert results == []


def test_session_vector_index_empty_query():
    engine = MockEmbeddingEngine(dimension=32)
    index = SessionVectorIndex(session_id="test-empty", embedding_engine=engine)
    assert index.search("empty", top_k=5) == []
    assert index.search("", top_k=5) == []


def test_embedding_engine_with_mocked_transformer(monkeypatch):
    import sys
    from unittest.mock import MagicMock

    mock_st_class = MagicMock()
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 768
    mock_instance.encode.return_value = np.ones((2, 768), dtype=np.float32)
    mock_st_class.return_value = mock_instance

    mock_module = MagicMock()
    mock_module.SentenceTransformer = mock_st_class
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_module)

    engine = EmbeddingEngine(model_name_or_path="C:/Models/embeddinggemma-2", device="cpu")
    dim = engine.dimension
    assert dim == 768

    vectors = engine.encode(["hello", "world"])
    assert vectors.shape == (2, 768)

    empty_vecs = engine.encode([])
    assert len(empty_vecs) == 0

