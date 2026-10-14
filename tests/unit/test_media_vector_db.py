"""Unit tests for MediaVectorDB, EmbeddingGemma-2 FAISS index, and media tools."""

import numpy as np
import pytest
from pathlib import Path

from lora.grounding.embed import EmbeddingEngine
from lora.grounding.media_vector_db import MediaVectorDB
from lora.tools.registry import ToolRegistry


class MockGemmaEmbedder(EmbeddingEngine):
    """Deterministic mock for EmbeddingGemma-2 on CPU."""

    def __init__(self, dimension: int = 64):
        super().__init__(model_name_or_path="C:/Models/embeddinggemma-2", device="cpu")
        self._dim = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vecs = []
        for t in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            for i, c in enumerate(t[: self._dim]):
                vec[i] = ord(c) % 7 + 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dim


def test_media_vector_db_ingestion_and_search(tmp_path: Path):
    embedder = MockGemmaEmbedder(dimension=32)
    db = MediaVectorDB(embedding_engine=embedder, storage_dir=tmp_path)

    # 1. Create test markdown and text files
    doc1 = tmp_path / "quantum_computing.md"
    doc1.write_text(
        "# Quantum Computing Overview\n\nQubits leverage quantum superposition and entanglement to execute quantum algorithms faster than classical computers.",
        encoding="utf-8",
    )

    doc2 = tmp_path / "biology_notes.txt"
    doc2.write_text(
        "CRISPR-Cas9 enables precise genome editing by utilizing guide RNA sequences to target specific DNA segments.",
        encoding="utf-8",
    )

    # Ingest doc1
    res1 = db.add_file(doc1, tags=["physics", "quantum"])
    assert res1["status"] == "success"
    assert res1["filename"] == "quantum_computing.md"
    assert res1["chunks_added"] >= 1
    assert db.file_count == 1

    # Ingest doc2
    res2 = db.add_file(doc2, tags=["genetics"])
    assert res2["status"] == "success"
    assert db.file_count == 2
    assert db.chunk_count >= 2

    # Check vector DB status
    vdb_status = db.get_vector_db()
    assert vdb_status["dimension"] == 32
    assert vdb_status["total_files"] == 2

    # Semantic search in FAISS
    hits = db.search("quantum superposition and qubits", top_k=2)
    assert len(hits) >= 1
    assert hits[0]["filename"] == "quantum_computing.md"
    assert hits[0]["score"] > 0.0

    # Agent Report generation
    report = db.report_on_media(query="quantum")
    assert report["status"] == "success"
    assert "Analyzed 2 indexed file(s)" in report["summary"]
    assert len(report["sources"]) == 2
    assert len(report["findings"]) >= 1


def test_media_vector_db_removal_and_rebuild(tmp_path: Path):
    embedder = MockGemmaEmbedder(dimension=32)
    db = MediaVectorDB(embedding_engine=embedder, storage_dir=tmp_path)

    f1 = tmp_path / "file1.txt"
    f1.write_text("Deep neural networks and residual connections.", encoding="utf-8")
    f2 = tmp_path / "file2.txt"
    f2.write_text("Photosynthesis transforms solar light energy into chemical sugars.", encoding="utf-8")

    db.add_file(f1)
    db.add_file(f2)
    assert db.file_count == 2

    # Remove file1
    rem_res = db.remove_file("file1.txt")
    assert rem_res["status"] == "success"
    assert rem_res["remaining_files"] == 1
    assert db.file_count == 1

    # Search should no longer return file1
    hits = db.search("Deep neural networks", top_k=5)
    filenames = [h["filename"] for h in hits]
    assert "file1.txt" not in filenames


def test_media_vector_db_save_and_load(tmp_path: Path):
    embedder = MockGemmaEmbedder(dimension=32)
    db = MediaVectorDB(embedding_engine=embedder, storage_dir=tmp_path)

    f = tmp_path / "persisted_doc.md"
    f.write_text("Saved database document content.", encoding="utf-8")
    db.add_file(f)

    # Save
    db.save_db(tmp_path)
    assert (tmp_path / "faiss.index").exists()
    assert (tmp_path / "metadata.json").exists()

    # Load into fresh instance
    new_db = MediaVectorDB(embedding_engine=embedder, storage_dir=tmp_path)
    loaded = new_db.load_db(tmp_path)
    assert loaded is True
    assert new_db.file_count == 1
    assert new_db.chunk_count == db.chunk_count


def test_media_tools_registered_in_tool_registry(tmp_path: Path):
    embedder = MockGemmaEmbedder(dimension=32)
    media_db = MediaVectorDB(embedding_engine=embedder, storage_dir=tmp_path)
    registry = ToolRegistry(media_db=media_db)

    tools = {t["name"]: t for t in registry.list_tools()}
    assert "add_media_file" in tools
    assert "remove_media_file" in tools
    assert "list_media_files" in tools
    assert "search_media_db" in tools
    assert "report_on_media" in tools

    # Test adding a file via handler
    test_f = tmp_path / "test_research.txt"
    test_f.write_text("Scholarly article regarding distributed consensus.", encoding="utf-8")

    add_handler = registry._tools["add_media_file"].handler
    add_res = add_handler(str(test_f))
    assert add_res["status"] == "success"

    list_handler = registry._tools["list_media_files"].handler
    files = list_handler()
    assert len(files) == 1
    assert files[0]["filename"] == "test_research.txt"
