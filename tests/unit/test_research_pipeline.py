"""Unit tests for ResearchPipeline end-to-end flow."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lora.grounding.embed import EmbeddingEngine
from lora.grounding.modes import DepthMode
from lora.grounding.pipeline import ResearchPipeline


class DummyEmbeddingEngine(EmbeddingEngine):
    def __init__(self, dimension: int = 16):
        self._dim = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            for i, c in enumerate(t[: self._dim]):
                vec[i] = ord(c) % 5 + 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


@pytest.mark.asyncio
async def test_research_pipeline_ingest_and_query():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "pipeline_notebooks.db"
        mock_ollama = AsyncMock()
        mock_ollama.generate = AsyncMock(
            return_value="Attention mechanisms process tokens across deep transformer layers [S1 p. 1]."
        )

        pipeline = ResearchPipeline(
            embedding_engine=DummyEmbeddingEngine(dimension=16),
            ollama_client=mock_ollama,
            notebook_db_path=db_path,
        )

        # 1. Ingest web article
        sample_web_text = (
            "Modern deep learning uses transformer architectures with multi-head self-attention. "
            "These models compute interactions between all pairs of sequence tokens in parallel. "
            "This completely replaced sequential recurrence in state-of-the-art NLP models."
        )
        ingested_count = pipeline.ingest_web_content(
            url="https://arxiv.org/abs/1706.03762",
            title="Attention Is All You Need",
            text=sample_web_text,
            tier=1,
        )
        assert ingested_count >= 1
        assert pipeline.session_index.chunk_count >= 1

        # 2. Run research query
        response = await pipeline.research(
            query="How does multi-head attention process sequence tokens?",
            depth=DepthMode.STANDARD,
        )

        assert "transformer" in response.answer.lower() or "attention" in response.answer.lower()
        assert len(response.evidence_cards) >= 1
        assert len(response.citations) >= 1
        assert response.citations[0].source_url == "https://arxiv.org/abs/1706.03762"

        # 3. Promote session to permanent notebook
        notebook_id = await pipeline.promote_to_notebook(title="Transformer Self-Attention Notes")
        assert notebook_id.startswith("nb-")

        # 4. Clear ephemeral index (simulating session end)
        pipeline.clear_session()
        assert pipeline.session_index.chunk_count == 0


@pytest.mark.asyncio
async def test_research_pipeline_pdf_and_depth_modes():
    mock_ollama = AsyncMock()
    mock_ollama.generate = AsyncMock(
        return_value="Think of attention like a spotlight in a theater [S1 p. 2]."
    )

    pipeline = ResearchPipeline(
        embedding_engine=DummyEmbeddingEngine(dimension=16),
        ollama_client=mock_ollama,
    )

    # Ingest PDF pages
    pdf_pages = [
        (1, "Introduction to neural networks and recurrent sequential architectures."),
        (2, "Self-attention spotlight mechanism routing information dynamically between tokens."),
    ]
    count = pipeline.ingest_pdf_pages(
        url="https://example.com/transformer.pdf",
        title="Transformer PDF",
        pages=pdf_pages,
        tier=1,
    )
    assert count >= 2
    assert pipeline.session_index.chunk_count >= 2

    # Query in ELI5 mode
    response = await pipeline.research(
        query="Explain attention mechanism",
        depth=DepthMode.ELI5,
    )
    assert "spotlight" in response.answer.lower()
    assert response.depth == DepthMode.ELI5
    assert len(response.citations) >= 1

    # Empty text ingestion returns 0
    assert pipeline.ingest_web_content("https://example.com", "Empty", "") == 0
    assert pipeline.ingest_pdf_pages("https://example.com", "Empty", []) == 0

