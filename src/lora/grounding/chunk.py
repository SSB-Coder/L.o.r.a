"""Document and web page chunking for grounded research."""

import hashlib
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
    page_number: int = 1
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
        return max(1, len(text.split()))

    def chunk_text(
        self,
        text: str,
        source_url: str,
        title: str,
        tier: int = 2,
        page_number: int = 1,
    ) -> list[DocumentChunk]:
        """Split a single document or web page string into chunks."""
        if not text or not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks: list[DocumentChunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        paragraph_idx = 0

        for s in sentences:
            s_tokens = self._estimate_tokens(s)
            if current_tokens + s_tokens > self.target_chunk_tokens and current_sentences:
                chunk_str = " ".join(current_sentences)
                c_hash = hashlib.sha256(chunk_str.encode("utf-8")).hexdigest()[:16]
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"chk-{uuid.uuid4().hex[:8]}",
                        source_url=source_url,
                        title=title,
                        text=chunk_str,
                        paragraph_index=paragraph_idx,
                        page_number=page_number,
                        token_count=current_tokens,
                        tier=tier,
                        content_hash=c_hash,
                    )
                )
                paragraph_idx += 1

                # Keep overlap sentences
                overlap_sents: list[str] = []
                overlap_toks = 0
                for prev_s in reversed(current_sentences):
                    prev_tok = self._estimate_tokens(prev_s)
                    if overlap_toks + prev_tok <= self.overlap_tokens:
                        overlap_sents.insert(0, prev_s)
                        overlap_toks += prev_tok
                    else:
                        break
                current_sentences = overlap_sents
                current_tokens = overlap_toks

            current_sentences.append(s)
            current_tokens += s_tokens

        if current_sentences:
            chunk_str = " ".join(current_sentences)
            c_hash = hashlib.sha256(chunk_str.encode("utf-8")).hexdigest()[:16]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chk-{uuid.uuid4().hex[:8]}",
                    source_url=source_url,
                    title=title,
                    text=chunk_str,
                    paragraph_index=paragraph_idx,
                    page_number=page_number,
                    token_count=current_tokens,
                    tier=tier,
                    content_hash=c_hash,
                )
            )

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
