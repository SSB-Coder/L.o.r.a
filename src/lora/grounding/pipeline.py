"""End-to-end grounded research engine with ephemeral web embeddings and notebook promotion."""

import logging
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from lora.grounding.cards import EvidenceCard, EvidenceCardExtractor
from lora.grounding.chunk import DocumentChunk, DocumentChunker
from lora.grounding.embed import EmbeddingEngine
from lora.grounding.modes import DepthMode
from lora.grounding.rerank import CrossEncoderReranker
from lora.grounding.session_index import SessionVectorIndex
from lora.grounding.writer import CitationItem, GroundedAnswer, ResearchWriter
from lora.storage.notebook import DEFAULT_NOTEBOOK_DB, NotebookStore

logger = logging.getLogger(__name__)


class ResearchResponse(BaseModel):
    """The complete response from a research query run."""

    model_config = ConfigDict(frozen=True)

    query: str
    answer: str
    depth: DepthMode
    citations: list[CitationItem]
    evidence_cards: list[EvidenceCard]
    overlap_ratio: float = 0.0


class ResearchPipeline:
    """Coordinates web content ingestion, ephemeral vector indexing, citation extraction,

    grounded synthesis, and notebook promotion.
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine | None = None,
        reranker: CrossEncoderReranker | None = None,
        ollama_client: Any = None,
        session_id: str | None = None,
        notebook_db_path: Path = DEFAULT_NOTEBOOK_DB,
    ) -> None:
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.session_index = SessionVectorIndex(
            session_id=self.session_id,
            embedding_engine=self.embedding_engine,
        )
        self.chunker = DocumentChunker()
        self.reranker = reranker or CrossEncoderReranker()
        self.extractor = EvidenceCardExtractor()
        self.writer = ResearchWriter(ollama_client=ollama_client)
        self.notebook_store = NotebookStore(db_path=notebook_db_path)

        self._all_ingested_chunks: list[DocumentChunk] = []
        self._last_cards: list[EvidenceCard] = []
        self._last_answer: GroundedAnswer | None = None

    def ingest_web_content(
        self,
        url: str,
        title: str,
        text: str,
        tier: int = 2,
    ) -> int:
        """Ingest and chunk web article text, adding it to the ephemeral in-memory index."""
        chunks = self.chunker.chunk_text(
            text=text,
            source_url=url,
            title=title,
            tier=tier,
        )
        if not chunks:
            return 0

        self.session_index.add_chunks(chunks)
        self._all_ingested_chunks.extend(chunks)
        logger.info(
            "Ingested web content from '%s' (%s): %d chunks indexed in RAM.",
            url,
            title,
            len(chunks),
        )
        return len(chunks)

    def ingest_pdf_pages(
        self,
        url: str,
        title: str,
        pages: list[tuple[int, str]],
        tier: int = 1,
    ) -> int:
        """Ingest multi-page PDF text into the ephemeral in-memory index."""
        chunks = self.chunker.chunk_pages(
            pages=pages,
            source_url=url,
            title=title,
            tier=tier,
        )
        if not chunks:
            return 0

        self.session_index.add_chunks(chunks)
        self._all_ingested_chunks.extend(chunks)
        logger.info(
            "Ingested PDF from '%s' (%s): %d chunks across %d pages indexed in RAM.",
            url,
            title,
            len(chunks),
            len(pages),
        )
        return len(chunks)

    async def research(
        self,
        query: str,
        depth: DepthMode = DepthMode.STANDARD,
        top_k: int = 8,
    ) -> ResearchResponse:
        """Perform semantic retrieval over visited web pages and synthesize grounded answer."""
        candidate_k = max(top_k * 2, 16)
        search_hits = self.session_index.search(query=query, top_k=candidate_k)
        retrieved_chunks = [hit[0] for hit in search_hits]
        reranked_chunks = self.reranker.rerank(query=query, chunks=retrieved_chunks, top_k=top_k)

        cards = self.extractor.extract_cards(chunks=reranked_chunks, max_cards=top_k)
        self._last_cards = cards

        answer = await self.writer.synthesize(query=query, cards=cards, depth=depth)
        self._last_answer = answer

        return ResearchResponse(
            query=query,
            answer=answer.text,
            depth=depth,
            citations=answer.citations,
            evidence_cards=cards,
            overlap_ratio=answer.overlap_ratio,
        )

    async def promote_to_notebook(self, title: str) -> str:
        """Persist current session research, source chunks, cards, and answer to SQLite."""
        answer_text = self._last_answer.text if self._last_answer else ""
        nb_id = await self.notebook_store.save_notebook(
            title=title,
            chunks=self._all_ingested_chunks,
            cards=self._last_cards,
            answer=answer_text,
        )
        logger.info("Promoted research session to persistent notebook %s ('%s')", nb_id, title)
        return nb_id

    def clear_session(self) -> None:
        """Cleanly wipes all in-memory vectors and chunks with zero disk residue."""
        self.session_index.clear()
        self._all_ingested_chunks.clear()
        self._last_cards.clear()
        self._last_answer = None
        logger.info("ResearchPipeline session %s wiped.", self.session_id)
