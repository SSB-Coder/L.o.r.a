"""Unit tests for NotebookStore SQLite persistence."""

import tempfile
from pathlib import Path

import pytest

from lora.grounding.cards import EvidenceCard
from lora.grounding.chunk import DocumentChunk
from lora.storage.notebook import NotebookStore


@pytest.mark.asyncio
async def test_notebook_store_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_notebooks.db"
        store = NotebookStore(db_path=db_path)
        await store.init_db()

        chunks = [
            DocumentChunk(
                chunk_id="c1",
                source_url="https://example.com/p1",
                title="Paper 1",
                text="Attention mechanism summary text.",
                paragraph_index=0,
                page_number=1,
                token_count=10,
                tier=1,
                content_hash="h1",
            )
        ]
        cards = [
            EvidenceCard(
                card_id="S1",
                claim_fact="Attention allows parallel computation.",
                exact_quote="Attention allows parallel computation.",
                source_url="https://example.com/p1",
                title="Paper 1",
                page_or_section="p. 1",
                paragraph_index=0,
                tier=1,
            )
        ]

        notebook_id = await store.save_notebook(
            title="Transformer Architecture Notes",
            chunks=chunks,
            cards=cards,
            answer="Transformers process tokens concurrently [S1 p. 1].",
        )

        assert notebook_id.startswith("nb-")

        loaded = await store.get_notebook(notebook_id)
        assert loaded is not None
        assert loaded.title == "Transformer Architecture Notes"
        assert len(loaded.chunks) == 1
        assert len(loaded.cards) == 1
        assert loaded.chunks[0].chunk_id == "c1"
        assert loaded.cards[0].card_id == "S1"

        all_nbs = await store.list_notebooks()
        assert len(all_nbs) == 1
        assert all_nbs[0]["id"] == notebook_id
