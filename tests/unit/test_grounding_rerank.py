"""Unit tests for CrossEncoderReranker in LORA grounding."""

from unittest.mock import MagicMock

from lora.grounding.chunk import DocumentChunk
from lora.grounding.rerank import CrossEncoderReranker


def test_reranker_empty_and_passthrough():
    reranker = CrossEncoderReranker()
    assert reranker.rerank("query", []) == []

    chunk1 = DocumentChunk(
        chunk_id="c1",
        source_url="https://example.com/1",
        title="Title 1",
        text="Sample text 1",
        paragraph_index=0,
        page_number=1,
        token_count=3,
        tier=1,
        content_hash="h1",
    )
    # len(chunks) <= top_k -> returns chunks
    res = reranker.rerank("query", [chunk1], top_k=5)
    assert len(res) == 1
    assert res[0].chunk_id == "c1"


def test_reranker_with_mock_model(monkeypatch):
    import sys

    mock_cross_encoder_class = MagicMock()
    mock_instance = MagicMock()
    # Return higher score for chunk 2
    mock_instance.predict.return_value = [0.1, 0.95]
    mock_cross_encoder_class.return_value = mock_instance

    mock_module = MagicMock()
    mock_module.CrossEncoder = mock_cross_encoder_class
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_module)

    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base", device="cpu")

    chunk1 = DocumentChunk(
        chunk_id="c1",
        source_url="https://example.com/1",
        title="Title 1",
        text="Less relevant text",
        paragraph_index=0,
        page_number=1,
        token_count=3,
        tier=1,
        content_hash="h1",
    )
    chunk2 = DocumentChunk(
        chunk_id="c2",
        source_url="https://example.com/2",
        title="Title 2",
        text="Highly relevant text about the exact query",
        paragraph_index=0,
        page_number=1,
        token_count=7,
        tier=1,
        content_hash="h2",
    )

    reranked = reranker.rerank(query="exact query", chunks=[chunk1, chunk2], top_k=1)
    assert len(reranked) == 1
    assert reranked[0].chunk_id == "c2"


def test_reranker_fallback_on_load_failure(monkeypatch):
    reranker = CrossEncoderReranker(model_name="nonexistent-model")

    def failing_loader():
        raise RuntimeError("Model download failed in offline environment")

    reranker._load_model = failing_loader

    chunk1 = DocumentChunk(
        chunk_id="c1",
        source_url="https://example.com/1",
        title="Title 1",
        text="Completely unrelated topic about gardening flowers.",
        paragraph_index=0,
        page_number=1,
        token_count=6,
        tier=2,
        content_hash="h1",
    )
    chunk2 = DocumentChunk(
        chunk_id="c2",
        source_url="https://example.com/2",
        title="Title 2",
        text="Deep learning transformer models with multi-head attention.",
        paragraph_index=0,
        page_number=1,
        token_count=7,
        tier=1,
        content_hash="h2",
    )

    # Fallback to token overlap ranker should not crash and should prioritize chunk 2
    res = reranker.rerank(query="transformer attention", chunks=[chunk1, chunk2], top_k=1)
    assert len(res) == 1
    assert res[0].chunk_id == "c2"
