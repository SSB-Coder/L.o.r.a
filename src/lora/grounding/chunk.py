"""Document and web page chunking for grounded research."""

import hashlib
import math
import re
import time
import uuid

from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    """An immutable chunk of an ingested web article or document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    source_url: str
    title: str
    text: str
    paragraph_index: int = 0
    page_number: int | None = 1
    section: str = ""
    token_count: int = 0
    tier: int = 2
    content_hash: str = ""
    timestamp: float = Field(default_factory=time.time)


class DocumentChunker:
    """Splits web page and PDF text into semantic, citation-preserving chunks."""

    def __init__(self, target_chunk_tokens: int = 400, overlap_tokens: int = 60) -> None:
        self.target_chunk_tokens = max(50, target_chunk_tokens)
        self.overlap_tokens = min(overlap_tokens, self.target_chunk_tokens // 2)

    def _split_into_sentences(self, text: str) -> list[str]:
        # Split on sentence boundaries while preserving structure
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    def _estimate_tokens(self, text: str) -> int:
        # Conservative estimate, also bounded for formulae, code and text
        # without spaces. This is an estimate, never a tokenizer measurement.
        return max(1, len(text.split()), math.ceil(len(text.encode("utf-8")) / 4))

    def _bounded_units(self, text: str) -> list[str]:
        units = []
        for block in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=\S)", text.strip()):
            rest = block.strip()
            while rest:
                if self._estimate_tokens(rest) <= self.target_chunk_tokens:
                    units.append(rest)
                    break
                lo, hi = 1, min(len(rest), self.target_chunk_tokens * 4)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self._estimate_tokens(rest[:mid]) <= self.target_chunk_tokens:
                        lo = mid
                    else:
                        hi = mid - 1
                cut = lo
                boundary = max(rest.rfind(" ", 0, cut + 1), rest.rfind("\n", 0, cut + 1))
                if boundary > cut // 2:
                    cut = boundary
                units.append(rest[:cut].strip())
                rest = rest[cut:].strip()
        return units

    def chunk_text(
        self,
        text: str,
        source_url: str,
        title: str,
        tier: int = 2,
        page_number: int | None = 1,
    ) -> list[DocumentChunk]:
        """Split a single document or web page string into chunks."""
        if not text or not text.strip():
            return []

        chunks: list[DocumentChunk] = []

        def emit(body: str, section: str) -> None:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chk-{uuid.uuid4().hex[:8]}",
                    source_url=source_url,
                    title=title,
                    text=body,
                    paragraph_index=len(chunks),
                    page_number=page_number,
                    section=section,
                    token_count=self._estimate_tokens(body),
                    tier=tier,
                    content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest()[:16],
                )
            )

        # A heading starts a new section; never blend unrelated sections just
        # to fill the budget. Keep headings and line breaks in the actual text.
        sections = re.split(r"(?m)(?=^#{1,6}\s+\S)", text.strip())
        for section_text in sections:
            if not section_text.strip():
                continue
            heading = re.match(r"^#{1,6}\s+([^\n]+)", section_text)
            section = heading.group(1).strip() if heading else ""
            current = ""
            for unit in self._bounded_units(section_text):
                joined = current + "\n\n" + unit if current else unit
                if current and self._estimate_tokens(joined) > self.target_chunk_tokens:
                    emit(current, section)
                    tail = current[-self.overlap_tokens * 4 :] if self.overlap_tokens else ""
                    # Start overlap at a word boundary, and discard it when a
                    # full-sized unit needs the entire available budget.
                    tail = tail.partition(" ")[2] if " " in tail else ""
                    current = (
                        tail
                        if tail
                        and self._estimate_tokens(tail + "\n\n" + unit) <= self.target_chunk_tokens
                        else ""
                    )
                current = current + "\n\n" + unit if current else unit
            if current:
                emit(current, section)
        return chunks

    def chunk_pages(
        self,
        pages: list[tuple[int, str]],
        source_url: str,
        title: str,
        tier: int = 1,
    ) -> list[DocumentChunk]:
        """Split multi-page documents (like academic PDFs) preserving exact page numbers."""
        all_chunks: list[DocumentChunk] = []
        for page_num, page_text in pages:
            page_chunks = self.chunk_text(
                text=page_text,
                source_url=source_url,
                title=title,
                tier=tier,
                page_number=page_num,
            )
            all_chunks.extend(page_chunks)
        return all_chunks
