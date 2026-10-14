"""Persistent and In-Memory FAISS Vector Database for Files & Media using EmbeddingGemma-2."""

import functools
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from lora.grounding.chunk import DocumentChunk, DocumentChunker
from lora.grounding.embed import EmbeddingEngine

logger = logging.getLogger(__name__)

DEFAULT_MEDIA_STORAGE = Path("data/media_db")


class MediaVectorDB:
    """FAISS-powered vector database for user files and media assets.

    Features:
    - Multi-modal file ingestion: PDF, TXT, MD, CSV, JSON, Code, Images, Audio.
    - Semantic chunking and vector indexing via EmbeddingGemma-2.
    - Fast sub-millisecond inner-product (cosine) search with FAISS CPU.
    - Dynamic file removal with vector index rebuilding.
    - In-memory querying and persistence to disk.
    - Agent reporting tool for executive summaries across all indexed files.
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine | None = None,
        storage_dir: Path | str | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.embedding_engine = embedding_engine or EmbeddingEngine(
            model_name_or_path=os.getenv("LORA_EMBEDDING_MODEL_PATH", "C:/Models/embeddinggemma-2")
        )
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_MEDIA_STORAGE
        self.chunker = DocumentChunker(target_chunk_tokens=350, overlap_tokens=50)

        self._files: dict[str, dict[str, Any]] = {}  # file_id -> metadata
        self._chunks: list[DocumentChunk] = []
        self._chunk_to_file: dict[str, str] = {}  # chunk_id -> file_id
        self._index: Any = None
        self._dimension: int | None = None

    @property
    def file_count(self) -> int:
        return len(self._files)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _init_index(self, dim: int) -> None:
        self._dimension = dim
        self._index = faiss.IndexFlatIP(dim)

    def _extract_file_content(self, path: Path) -> tuple[str, list[tuple[int, str]], str]:
        """Extract text content and page/segment breakdown according to file type."""
        suffix = path.suffix.lower()
        full_text = ""
        pages: list[tuple[int, str]] = []
        media_type = "document"

        # 1. PDF via PyMuPDF
        if suffix == ".pdf":
            media_type = "pdf"
            try:
                import warnings
                warnings.filterwarnings("ignore", message=".*fitz.*")
                import fitz

                with fitz.open(str(path)) as doc:
                    for i in range(len(doc)):
                        pg_text = doc[i].get_text()
                        if pg_text.strip():
                            pages.append((i + 1, pg_text))
                full_text = "\n\n".join(t for _, t in pages)
            except Exception as e:
                logger.warning("Could not read PDF with PyMuPDF: %s", e)
                full_text = f"[PDF file: {path.name}]"

        # 2. Images: extract metadata and resolution
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
            media_type = "image"
            try:
                from PIL import Image

                with Image.open(str(path)) as img:
                    w, h = img.size
                    mode = img.mode
                    fmt = img.format
                full_text = (
                    f"Image file: {path.name}\n"
                    f"Format: {fmt}, Dimensions: {w}x{h} px, Color mode: {mode}\n"
                    f"File path: {path.resolve()}"
                )
            except Exception as e:
                logger.warning("Could not read image metadata: %s", e)
                full_text = f"Image file: {path.name}"
            pages.append((1, full_text))

        # 3. Audio: transcribe if possible or record metadata
        elif suffix in (".wav", ".mp3", ".ogg", ".flac", ".m4a"):
            media_type = "audio"
            try:
                from lora.voice.stt import FasterWhisperTranscriber
                import soundfile as sf  # type: ignore

                data, sr = sf.read(str(path))
                if data.ndim > 1:
                    data = data.mean(axis=1)
                stt = FasterWhisperTranscriber(lazy_load=True)
                transcription = stt.transcribe(data.astype(np.float32), sample_rate=sr)
                audio_text = transcription.text.strip()
                full_text = (
                    f"Audio file: {path.name}\n"
                    f"Transcription:\n{audio_text if audio_text else '[No speech detected]'}"
                )
            except Exception as e:
                logger.info("Audio transcription fallback for %s: %s", path.name, e)
                full_text = f"Audio file: {path.name} (Duration/speech metadata recorded)"
            pages.append((1, full_text))

        # 4. Standard text/code/markdown/json/csv
        else:
            media_type = "text" if suffix in (".txt", ".md", ".csv") else "code"
            try:
                full_text = path.read_text(encoding="utf-8", errors="replace")
                pages.append((1, full_text))
            except Exception as e:
                logger.warning("Could not read text file: %s", e)
                full_text = f"File {path.name}"
                pages.append((1, full_text))

        return full_text, pages, media_type

    def add_file(self, file_path: str | Path, tags: list[str] | None = None) -> dict[str, Any]:
        """Ingest a file or media asset, compute EmbeddingGemma-2 vectors, and add to FAISS."""
        p = Path(file_path).resolve()
        if not p.exists() or not p.is_file():
            return {
                "status": "error",
                "message": f"File does not exist or is not a regular file: {file_path}",
            }

        # Check if already indexed
        for fid, fmeta in self._files.items():
            if Path(fmeta["path"]).resolve() == p:
                return {
                    "status": "exists",
                    "file_id": fid,
                    "message": f"File '{p.name}' is already indexed in the vector database.",
                }

        file_id = f"file-{uuid.uuid4().hex[:8]}"
        full_text, pages, media_type = self._extract_file_content(p)

        if not full_text.strip():
            return {
                "status": "error",
                "message": f"No extractable text or metadata found in '{p.name}'.",
            }

        # Chunk content
        chunks: list[DocumentChunk] = []
        if len(pages) > 1:
            chunks = self.chunker.chunk_pages(
                pages=pages,
                source_url=str(p),
                title=p.name,
                tier=1,
            )
        else:
            chunks = self.chunker.chunk_text(
                text=full_text,
                source_url=str(p),
                title=p.name,
                tier=1,
            )

        if not chunks:
            # Create at least one fallback chunk
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chk-{uuid.uuid4().hex[:8]}",
                    source_url=str(p),
                    title=p.name,
                    text=full_text[:1200],
                    paragraph_index=0,
                    page_number=1,
                    token_count=len(full_text.split()),
                    tier=1,
                    content_hash=hashlib.sha256(full_text.encode("utf-8")).hexdigest()[:16],
                )
            )

        # Compute embeddings via EmbeddingGemma-2
        texts = [c.text for c in chunks]
        embeddings = self.embedding_engine.encode(texts)

        dim = embeddings.shape[1] if len(embeddings) > 0 else 768
        if self._index is None:
            self._init_index(dim)
        elif self._dimension != dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self._dimension}, got {dim}")

        self._index.add(embeddings)

        chunk_ids = []
        for c in chunks:
            self._chunks.append(c)
            self._chunk_to_file[c.chunk_id] = file_id
            chunk_ids.append(c.chunk_id)

        self._files[file_id] = {
            "file_id": file_id,
            "filename": p.name,
            "path": str(p),
            "media_type": media_type,
            "size_bytes": p.stat().st_size,
            "chunk_count": len(chunks),
            "chunk_ids": chunk_ids,
            "tags": tags or [],
            "added_at": time.time(),
        }

        logger.info(
            "Indexed '%s' (%s) -> %d chunks into FAISS (Total chunks: %d)",
            p.name,
            media_type,
            len(chunks),
            len(self._chunks),
        )

        return {
            "status": "success",
            "file_id": file_id,
            "filename": p.name,
            "media_type": media_type,
            "chunks_added": len(chunks),
            "total_chunks": len(self._chunks),
            "total_files": len(self._files),
            "vector_dimension": dim,
        }

    def remove_file(self, file_id_or_path: str) -> dict[str, Any]:
        """Remove a file and its vectors from the FAISS database."""
        target_id = None
        target_meta = None

        if file_id_or_path in self._files:
            target_id = file_id_or_path
            target_meta = self._files[target_id]
        else:
            raw_norm = file_id_or_path.strip().lower()
            try:
                clean_str = str(Path(file_id_or_path).resolve()).lower()
            except Exception:
                clean_str = raw_norm
            for fid, meta in self._files.items():
                meta_path_low = str(Path(meta["path"]).resolve()).lower()
                meta_name_low = str(meta["filename"]).lower()
                if (
                    fid.lower() == raw_norm
                    or meta_path_low == clean_str
                    or meta_name_low == raw_norm
                    or Path(meta["path"]).name.lower() == raw_norm
                ):
                    target_id = fid
                    target_meta = meta
                    break

        if not target_id or not target_meta:
            return {
                "status": "error",
                "message": f"File '{file_id_or_path}' not found in active vector index.",
            }

        filename = target_meta["filename"]
        chunk_ids_to_remove = set(target_meta.get("chunk_ids", []))

        # Rebuild FAISS index without the removed file's chunks
        del self._files[target_id]
        remaining_chunks = [c for c in self._chunks if c.chunk_id not in chunk_ids_to_remove]

        for cid in chunk_ids_to_remove:
            self._chunk_to_file.pop(cid, None)

        self._chunks = remaining_chunks

        if self._dimension is not None:
            self._index = faiss.IndexFlatIP(self._dimension)
            if self._chunks:
                texts = [c.text for c in self._chunks]
                embeddings = self.embedding_engine.encode(texts)
                self._index.add(embeddings)
        else:
            self._index = None

        if (self.storage_dir / "metadata.json").exists():
            try:
                self.save_db(self.storage_dir)
            except Exception as e:
                logger.warning("Could not persist MediaVectorDB after removal: %s", e)

        logger.info("Removed '%s' from vector DB. Remaining files: %d", filename, len(self._files))
        return {
            "status": "success",
            "removed_file_id": target_id,
            "filename": filename,
            "remaining_files": len(self._files),
            "remaining_chunks": len(self._chunks),
        }

    def list_files(self) -> list[dict[str, Any]]:
        """List all indexed files and media assets."""
        return list(self._files.values())

    def get_vector_db(self) -> dict[str, Any]:
        """Return the raw FAISS vector database object and runtime status."""
        return {
            "faiss_index": self._index,
            "dimension": self._dimension,
            "total_chunks": len(self._chunks),
            "total_files": len(self._files),
            "model_path": self.embedding_engine.model_path,
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Semantic search against all indexed files and media in the FAISS database."""
        if not query or not query.strip() or self._index is None or not self._chunks:
            return []

        q_vec = self.embedding_engine.encode([query.strip()])
        if len(q_vec) == 0:
            return []

        actual_k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(q_vec, actual_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._chunks):
                continue
            f_score = float(score)
            if f_score >= score_threshold:
                chunk = self._chunks[idx]
                fid = self._chunk_to_file.get(chunk.chunk_id, "")
                fmeta = self._files.get(fid, {})
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "file_id": fid,
                    "filename": fmeta.get("filename", chunk.title),
                    "path": fmeta.get("path", chunk.source_url),
                    "media_type": fmeta.get("media_type", "document"),
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "score": round(f_score, 4),
                })

        return results

    def report_on_media(
        self,
        query: str = "",
        file_ids: list[str] | None = None,
        top_k: int = 6,
    ) -> dict[str, Any]:
        """Reads indexed files and media and generates a structured research report."""
        if not self._files:
            return {
                "status": "empty",
                "summary": "No files or media are currently indexed in the vector database.",
                "sources": [],
                "details": [],
            }

        # 1. Gather passages either via semantic search or overview
        hits: list[dict[str, Any]] = []
        if query.strip():
            hits = self.search(query=query, top_k=top_k)
        else:
            # Provide balanced sampling across all indexed files
            for fid, meta in self._files.items():
                if file_ids and fid not in file_ids:
                    continue
                c_ids = meta.get("chunk_ids", [])
                for cid in c_ids[:2]:  # First 2 chunks per file
                    chunk = next((c for c in self._chunks if c.chunk_id == cid), None)
                    if chunk:
                        hits.append({
                            "chunk_id": chunk.chunk_id,
                            "file_id": fid,
                            "filename": meta["filename"],
                            "path": meta["path"],
                            "media_type": meta["media_type"],
                            "text": chunk.text,
                            "page_number": chunk.page_number,
                            "score": 1.0,
                        })

        # 2. Compile structured report
        sources_summary = [
            {
                "file_id": f["file_id"],
                "filename": f["filename"],
                "media_type": f["media_type"],
                "chunks": f["chunk_count"],
            }
            for f in self._files.values()
        ]

        # Synthesis
        findings = []
        for i, hit in enumerate(hits, 1):
            snippet = hit["text"][:240].replace("\n", " ").strip()
            findings.append(
                f"[S{i}] ({hit['filename']}, p.{hit['page_number']}): \"{snippet}...\" (match score: {hit['score']})"
            )

        summary_text = (
            f"Analyzed {len(sources_summary)} indexed file(s) across {len(self._chunks)} chunks "
            f"using EmbeddingGemma-2 FAISS vector database. Found {len(hits)} relevant passage(s)."
        )

        return {
            "status": "success",
            "query": query,
            "summary": summary_text,
            "findings": findings,
            "sources": sources_summary,
            "passages": hits,
        }

    def save_db(self, target_dir: Path | str | None = None) -> Path:
        """Persist FAISS index and file metadata to disk."""
        out_dir = Path(target_dir) if target_dir else self.storage_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        idx_file = out_dir / "faiss.index"
        meta_file = out_dir / "metadata.json"

        if self._index is not None:
            faiss.write_index(self._index, str(idx_file))

        meta_data = {
            "files": self._files,
            "dimension": self._dimension,
            "chunks": [c.model_dump() for c in self._chunks],
            "chunk_to_file": self._chunk_to_file,
        }
        meta_file.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
        logger.info("Saved MediaVectorDB to %s", out_dir)
        return out_dir

    def load_db(self, source_dir: Path | str | None = None) -> bool:
        """Load persisted FAISS index and metadata from disk."""
        s_dir = Path(source_dir) if source_dir else self.storage_dir
        idx_file = s_dir / "faiss.index"
        meta_file = s_dir / "metadata.json"

        if not idx_file.exists() or not meta_file.exists():
            return False

        try:
            self._index = faiss.read_index(str(idx_file))
            meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
            self._files = meta_data.get("files", {})
            self._dimension = meta_data.get("dimension")
            self._chunk_to_file = meta_data.get("chunk_to_file", {})
            self._chunks = [DocumentChunk(**c) for c in meta_data.get("chunks", [])]
            logger.info("Loaded MediaVectorDB from %s with %d files", s_dir, len(self._files))
            return True
        except Exception as e:
            logger.warning("Error loading MediaVectorDB from %s: %s", s_dir, e)
            return False


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)

    return wrapper


# add/remove/search/report share one FAISS index; the UI adds files on a background thread.
_LOCKED_METHODS = ("add_file", "remove_file", "search", "report_on_media", "save_db", "load_db", "list_files", "get_vector_db")
for _name in _LOCKED_METHODS:
    setattr(MediaVectorDB, _name, _locked(getattr(MediaVectorDB, _name)))
