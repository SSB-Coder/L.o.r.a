"""Unit tests for semantic document chunking in LORA grounding."""

import pytest

from lora.grounding.chunk import DocumentChunk, DocumentChunker


def test_document_chunk_immutable():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        source_url="https://example.com/article",
        title="Example Article",
        text="This is a test paragraph explaining attention mechanisms.",
        paragraph_index=0,
        page_number=1,
        token_count=10,
        tier=2,
        content_hash="abc123hash",
    )
    assert chunk.chunk_id == "chunk-1"
    assert chunk.source_url == "https://example.com/article"
    with pytest.raises(Exception):
        chunk.text = "Attempted mutation"


def test_chunker_basic_text():
    chunker = DocumentChunker(target_chunk_tokens=50, overlap_tokens=10)
    text = (
        "Artificial intelligence has evolved significantly over the past decade. "
        "Modern transformer architectures utilize self-attention mechanisms to process "
        "tokens in parallel across deep layers. "
        "This approach replaced recurrent networks that processed sequence items sequentially. "
        "Furthermore, retrieval-augmented generation connects these models to external databases. "
        "By doing so, factual hallucinations are substantially decreased in domain-specific tasks."
    )
    chunks = chunker.chunk_text(
        text=text,
        source_url="https://example.com/ai",
        title="AI Evolution",
        tier=2,
    )

    assert len(chunks) >= 1
    for i, c in enumerate(chunks):
        assert c.source_url == "https://example.com/ai"
        assert c.title == "AI Evolution"
        assert c.paragraph_index == i
        assert len(c.text.strip()) > 0
        assert c.content_hash != ""


def test_chunker_empty_and_short_text():
    chunker = DocumentChunker()
    assert chunker.chunk_text("", source_url="https://example.com", title="Empty") == []
    assert chunker.chunk_text("   \n\n  ", source_url="https://example.com", title="Blank") == []

    single = chunker.chunk_text(
        "Short one-liner fact.", source_url="https://example.com/fact", title="Fact"
    )
    assert len(single) == 1
    assert single[0].text == "Short one-liner fact."


def test_chunker_preserves_page_numbers():
    chunker = DocumentChunker(target_chunk_tokens=30, overlap_tokens=5)
    pages = [
        (1, "Page 1 content discussing the initial methodology and setup of experiments."),
        (2, "Page 2 content discussing the empirical results and benchmark latency measurements."),
    ]
    chunks = chunker.chunk_pages(
        pages=pages,
        source_url="https://arxiv.org/abs/2401.00000",
        title="Research Paper",
        tier=1,
    )
    assert len(chunks) >= 2
    assert any(c.page_number == 1 for c in chunks)
    assert any(c.page_number == 2 for c in chunks)
