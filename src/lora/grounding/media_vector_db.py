"""Persistent CPU vector retrieval for files and media using local embeddings."""

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

import numpy as np

from lora.grounding.chunk import DocumentChunk, DocumentChunker
from lora.grounding.embed import DEFAULT_MODEL_PATH, EmbeddingEngine
from lora.grounding.vector_index import load_index, make_index, save_index

logger = logging.getLogger(__name__)

DEFAULT_MEDIA_STORAGE = Path(__file__).resolve().parents[3] / "data" / "media_db"
INGEST_VERSION = 2


class MediaVectorDB:
    """Exact CPU vector database for user files and media assets.

    Features:
    - Multi-modal file ingestion: PDF, TXT, MD, CSV, JSON, Code, Images, Audio.
    - Semantic chunking and vector indexing via EmbeddingGemma-2.
    - Exact inner-product search with NumPy or optional FAISS CPU.
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
            model_name_or_path=os.getenv("LORA_EMBEDDING_MODEL_PATH") or DEFAULT_MODEL_PATH
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
        self._index = make_index(dim)

    def _extract_file_content(self, path: Path) -> tuple[str, list[tuple[int | None, str]], str]:
        """Extract text content and page/segment breakdown according to file type."""
        suffix = path.suffix.lower()
        full_text = ""
        pages: list[tuple[int, str]] = []
        media_type = "document"

        if suffix in (".pdf", ".docx"):
            from lora.safety.parsers import MAX_INPUT_BYTES, parse_document

            if path.stat().st_size > MAX_INPUT_BYTES:
                raise ValueError("Document exceeds parser input limit")
            result = parse_document(suffix[1:], path.read_bytes())
            media_type = suffix[1:]
            pages = [(p.get("page") if suffix == ".pdf" else None, p["text"])
                     for p in result["pages"] if p["text"].strip()]
            full_text = "\n\n".join(t for _, t in pages)

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
                import soundfile as sf  # type: ignore

                from lora.voice.stt import FasterWhisperTranscriber

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
            allowed = {".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".tsx", ".jsx",
                       ".html", ".css", ".yaml", ".yml", ".toml", ".xml", ".sql", ".log", ".rst"}
            if suffix not in allowed:
                raise ValueError(f"Unsupported source format: {suffix or 'unknown'}")
            if path.stat().st_size > 8_000_000:
                raise ValueError("Text source exceeds 8 MB limit")
            media_type = "text" if suffix in (".txt", ".md", ".csv") else "code"
            raw = path.read_bytes()
            if b"\x00" in raw or raw.startswith(b"PK\x03\x04"):
                raise ValueError("Source contains binary data rather than UTF-8 text")
            full_text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
            pages.append((None, full_text))

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
                if fmeta.get("ingest_version") != INGEST_VERSION:
                    replacement = MediaVectorDB(self.embedding_engine, self.storage_dir)
                    result = replacement.add_file(p, tags or fmeta.get("tags"))
                    if result["status"] != "success":
                        return result
                    self._append_replacement(replacement, fid)
                    return {**result, "reindexed": True, "total_files": self.file_count, "total_chunks": self.chunk_count}
                return {
                    "status": "exists",
                    "file_id": fid,
                    "message": f"File '{p.name}' is already indexed in the vector database.",
                }

        file_id = f"file-{uuid.uuid4().hex[:8]}"
        started = time.perf_counter()
        try:
            full_text, pages, media_type = self._extract_file_content(p)
        except Exception as exc:
            logger.warning("Source extraction failed for %s: %s", p.name, exc)
            return {"status": "error", "message": f"Could not extract {p.name}: {exc}"}
        extracted_at = time.perf_counter()

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
                page_number=pages[0][0] if pages else None,
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
                    page_number=pages[0][0] if pages else None,
                    token_count=len(full_text.split()),
                    tier=1,
                    content_hash=hashlib.sha256(full_text.encode("utf-8")).hexdigest()[:16],
                )
            )

        # Compute embeddings via EmbeddingGemma-2
        texts = [c.text for c in chunks]
        embeddings = self.embedding_engine.encode(texts)
        embedded_at = time.perf_counter()

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
            "ingest_version": INGEST_VERSION,
            "extracted_chars": len(full_text),
            "estimated_tokens": self.chunker._estimate_tokens(full_text),
            "min_chunk_tokens": min(c.token_count for c in chunks),
            "max_chunk_tokens": max(c.token_count for c in chunks),
            "extraction_seconds": round(extracted_at - started, 3),
            "embedding_seconds": round(embedded_at - extracted_at, 3),
        }

        logger.info(
            "Indexed '%s' (%s) -> %d vector chunks (Total chunks: %d)",
            p.name,
            media_type,
            len(chunks),
            len(self._chunks),
        )
        logger.info("Source timings: extraction=%.3fs embedding/chunking=%.3fs chars=%d chunks=%d estimated_tokens=%d..%d",
                    extracted_at - started, embedded_at - extracted_at, len(full_text), len(chunks),
                    min(c.token_count for c in chunks), max(c.token_count for c in chunks))

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

    def _append_replacement(self, replacement: "MediaVectorDB", old_id: str) -> None:
        """Commit a reindex only after all replacement vectors are ready."""
        retained = [c for c in self._chunks if self._chunk_to_file.get(c.chunk_id) != old_id]
        chunks = retained + replacement._chunks
        vectors = self.embedding_engine.encode([c.text for c in chunks])
        index = make_index(vectors.shape[1])
        index.add(vectors)
        files = {fid: meta for fid, meta in self._files.items() if fid != old_id}
        files.update(replacement._files)
        mapping = {c.chunk_id: self._chunk_to_file[c.chunk_id] for c in retained}
        mapping.update(replacement._chunk_to_file)
        self._index, self._dimension, self._chunks = index, vectors.shape[1], chunks
        self._files, self._chunk_to_file = files, mapping

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
            self._index = make_index(self._dimension)
            if self._chunks and self._index is not None:
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
        """Return the vector index and runtime status (legacy key retained)."""
        from lora.grounding.vector_index import backend_name

        return {
            "backend": backend_name(),
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
                    "section": chunk.section,
                    "score": round(f_score, 4),
                })

        return results

    def report_on_media(
        self,
        query: str = "",
        file_ids: list[str] | None = None,
        top_k: int = 6,
    ) -> dict[str, Any]:
        """Return index diagnostics and sample passages, not a prose summary."""
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
                            "section": chunk.section,
                            "score": None,
                        })

        # 2. Compile structured report
        sources_summary = [
            {
                "file_id": f["file_id"],
                "filename": f["filename"],
                "media_type": f["media_type"],
                "chunks": f["chunk_count"],
                "extracted_chars": f.get("extracted_chars"),
                "estimated_tokens": f.get("estimated_tokens"),
                "min_chunk_tokens": f.get("min_chunk_tokens"),
                "max_chunk_tokens": f.get("max_chunk_tokens"),
                "extraction_seconds": f.get("extraction_seconds"),
                "embedding_seconds": f.get("embedding_seconds"),
            }
            for f in self._files.values()
            if not file_ids or f["file_id"] in file_ids
        ]

        # Synthesis
        findings = []
        for i, hit in enumerate(hits, 1):
            snippet = hit["text"][:240].replace("\n", " ").strip()
            location = f", p.{hit['page_number']}" if hit.get("page_number") else ""
            if hit.get("section"):
                location += f", section: {hit['section']}"
            label = f"similarity: {hit['score']}" if hit["score"] is not None else "sample passage"
            findings.append(
                f"[S{i}] ({hit['filename']}{location}): \"{snippet}...\" ({label})"
            )

        summary_text = (
            f"Analyzed {len(sources_summary)} indexed file(s) across {sum(s['chunks'] for s in sources_summary)} chunks. "
            + (f"Found {len(hits)} retrieved passage(s)." if query.strip() else
               f"Showing {len(hits)} sample passage(s), up to two per file; this is not a relevance search.")
        )
        for source in sources_summary:
            if source.get("extracted_chars") is not None:
                summary_text += (f"\n{source['filename']}: {source['extracted_chars']} extracted characters; "
                                 f"{source['chunks']} chunks, estimated {source['min_chunk_tokens']}–{source['max_chunk_tokens']} tokens per chunk.")

        return {
            "status": "success",
            "query": query,
            "summary": summary_text,
            "findings": findings,
            "sources": sources_summary,
            "passages": hits,
        }

    def summary_batches(self, max_chars: int = 10000, max_batches: int = 24) -> dict[str, Any]:
        """Snapshot document-order passages for bounded writer synthesis.

        Includes the end of each document when it fits. Any coverage limit is
        explicit, and the index lock is released before model inference starts.
        """
        if max_chars < 1500 or max_batches < 1:
            raise ValueError("Summary limits must allow at least one source chunk and one batch")
        batches = []
        included = 0
        for number, (file_id, meta) in enumerate(self._files.items(), 1):
            chunks = [chunk for chunk in self._chunks if self._chunk_to_file.get(chunk.chunk_id) == file_id]
            pending = []
            size = 0
            for chunk in chunks:
                text = chunk.text
                if pending and size + len(text) > max_chars:
                    batches.append({"id": f"S{number}", "title": meta["filename"], "text": "\n\n".join(pending)})
                    pending, size = [], 0
                if len(batches) >= max_batches:
                    break
                pending.append(text)
                size += len(text) + 2
                included += 1
            if pending and len(batches) < max_batches:
                batches.append({"id": f"S{number}", "title": meta["filename"], "text": "\n\n".join(pending)})
            if len(batches) >= max_batches:
                break
        return {"batches": batches, "included_chunks": included, "total_chunks": len(self._chunks),
                "partial": included < len(self._chunks), "files": self.list_files()}

    def save_db(self, target_dir: Path | str | None = None) -> Path:
        """Persist FAISS index and file metadata to disk."""
        out_dir = Path(target_dir) if target_dir else self.storage_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        idx_file = out_dir / "faiss.index"
        meta_file = out_dir / "metadata.json"

        if self._index is not None:
            save_index(self._index, idx_file)

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
            meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
            if any(f.get("ingest_version") != INGEST_VERSION for f in meta_data.get("files", {}).values()):
                rebuilt = MediaVectorDB(self.embedding_engine, self.storage_dir)
                for source in meta_data["files"].values():
                    result = rebuilt.add_file(source["path"], source.get("tags"))
                    if result["status"] != "success":
                        raise ValueError(f"Reindex required for {source['filename']}: {result.get('message', 'extraction failed')}")
                self._index, self._dimension = rebuilt._index, rebuilt._dimension
                self._chunks, self._files, self._chunk_to_file = rebuilt._chunks, rebuilt._files, rebuilt._chunk_to_file
                logger.info("Rebuilt legacy source index from originals; saved index preserved until next save")
                return True
            chunks = [DocumentChunk(**c) for c in meta_data.get("chunks", [])]
            try:
                index = load_index(idx_file)
            except ValueError as exc:
                if not str(exc).startswith("Legacy FAISS index preserved") or not chunks:
                    raise
                # Metadata already contains the original text. Rebuild in RAM
                # without importing the broken DLL or changing saved user files.
                vectors = self.embedding_engine.encode([c.text for c in chunks])
                index = make_index(vectors.shape[1])
                index.add(vectors)
                logger.info("Rebuilt legacy media vectors in RAM; original index preserved")
            if index.ntotal != len(chunks):
                raise ValueError("Persisted vector count does not match source chunks")
            self._index = index
            self._files = meta_data.get("files", {})
            self._dimension = index.d
            self._chunk_to_file = meta_data.get("chunk_to_file", {})
            self._chunks = chunks
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
_LOCKED_METHODS = ("add_file", "remove_file", "search", "report_on_media", "summary_batches", "save_db", "load_db", "list_files", "get_vector_db")
for _name in _LOCKED_METHODS:
    setattr(MediaVectorDB, _name, _locked(getattr(MediaVectorDB, _name)))
