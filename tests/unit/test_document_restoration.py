"""Real document extraction plus bounded chunking, migration and evidence regressions."""

import json
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lora.grounding.cards import EvidenceCard
from lora.grounding.chunk import DocumentChunker
from lora.grounding.embed import EmbeddingEngine
from lora.grounding.media_vector_db import MediaVectorDB
from lora.verify.corroborate import CorroborationChecker
from lora.verify.evidence import relevant_excerpt


@pytest.fixture
def engine():
    return SimpleNamespace(
        encode=lambda texts: np.ones((len(texts), 4), dtype=np.float32) / 2,
        model_path="test-only",
        _use_fallback=False,
    )


def test_real_docx_extracts_tables_in_document_order(tmp_path, engine):
    path = tmp_path / "notes.docx"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr(
            "word/document.xml",
            """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Experiment</w:t></w:r></w:p>
<w:p><w:r><w:t>Before the results table.</w:t></w:r></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Epochs</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Accuracy</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>Forty</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Ninety percent</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
<w:p><w:r><w:t>After the results table.</w:t></w:r></w:p></w:body></w:document>""",
        )
        package.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        )
        package.writestr(
            "_rels/.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>',
        )
    db = MediaVectorDB(engine, tmp_path / "index")
    result = db.add_file(path)
    assert result["status"] == "success" and result["media_type"] == "docx"
    text = "\n".join(c.text for c in db._chunks)
    assert text.index("Before") < text.index("Forty") < text.index("After")
    assert "word/document.xml" not in text and "PK" not in text
    assert all(c.page_number is None for c in db._chunks)
    assert any(c.section == "Experiment" for c in db._chunks)


def test_corrupt_docx_and_binary_disguised_as_text_are_not_indexed(tmp_path, engine):
    db = MediaVectorDB(engine, tmp_path / "index")
    for name in ["bad.docx", "bad.md", "unknown.bin"]:
        path = tmp_path / name
        path.write_bytes(b"PK\x03\x04\x00corrupt archive")
        result = db.add_file(path)
        assert result["status"] == "error"
    assert db.file_count == 0 and db.chunk_count == 0


@pytest.mark.parametrize(
    "body",
    ["word " * 2500, "long_formula_" * 2000, "模型研究" * 1500],
    ids=["prose", "formula", "unicode"],
)
def test_long_markdown_never_creates_oversized_chunks(body):
    chunker = DocumentChunker(120, 15)
    chunks = chunker.chunk_text(
        "# First\n\n" + body + "\n\n## Second\n\nFinal fact.", "file.md", "Report", page_number=None
    )
    assert len(chunks) > 3
    assert all(c.token_count <= 120 for c in chunks)
    assert chunks[-1].section == "Second" and "Final fact." in chunks[-1].text
    # Hard splitting must preserve all source characters (excluding whitespace).
    no_overlap = DocumentChunker(120, 0).chunk_text(body, "file.md", "Report")
    assert "".join("".join(c.text.split()) for c in no_overlap) == "".join(body.split())


def test_overview_reports_sample_count_without_fake_similarity_or_pages(tmp_path, engine):
    path = tmp_path / "notes.md"
    path.write_text(
        "# One\n\n" + "A fact about stars. " * 500 + "\n\n# Two\n\nAnother fact.", encoding="utf-8"
    )
    db = MediaVectorDB(engine, tmp_path / "index")
    db.add_file(path)
    report = db.report_on_media()
    assert db.chunk_count > 2 and len(report["passages"]) == 2
    assert "sample passage" in report["summary"]
    assert all(hit["score"] is None for hit in report["passages"])
    assert all(
        "p.1" not in finding and "match score" not in finding for finding in report["findings"]
    )
    assert report["sources"][0]["extracted_chars"] == len(path.read_text(encoding="utf-8"))
    assert report["sources"][0]["max_chunk_tokens"] <= 350


def test_legacy_ingestion_rebuilds_from_original_not_binary_cache(tmp_path, engine):
    path = tmp_path / "notes.md"
    path.write_text("# Current source\n\nReal original text.", encoding="utf-8")
    db = MediaVectorDB(engine, tmp_path / "index")
    db.add_file(path)
    db.save_db()
    metadata_path = db.storage_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for source in metadata["files"].values():
        source.pop("ingest_version")
    metadata["chunks"][0]["text"] = "PK compressed garbage"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    original = metadata_path.read_bytes()
    fresh = MediaVectorDB(engine, db.storage_dir)
    assert fresh.load_db()
    assert all("PK compressed" not in c.text for c in fresh._chunks)
    assert metadata_path.read_bytes() == original  # disk replacement remains an explicit save
    assert fresh.list_files()[0]["ingest_version"] == 2


def test_direct_opposition_detected_but_unrelated_negation_not_conflict():
    def card(identifier, text):
        return EvidenceCard(
            card_id=identifier,
            claim_fact=text,
            exact_quote=text,
            source_url=f"https://{identifier}.example",
            title=identifier,
        )

    checker = CorroborationChecker()
    first = card(
        "S1", "Gravity bends the path of light. Redshift stretches light to longer wavelengths."
    )
    second = card(
        "S2",
        "Redshift stretches light to longer wavelengths. This does not imply every galaxy recedes.",
    )
    assert checker.detect_contradictions([first, second]) == []
    opposite = card("S3", "Redshift does not stretch light to longer wavelengths.")
    positive = card("S4", "Redshift does stretch light to longer wavelengths.")
    assert len(checker.detect_contradictions([opposite, positive])) == 1


def test_evidence_selector_keeps_claim_support_at_end_of_long_page():
    fact = "The measured spectral line shifts to longer wavelengths because space expands."
    premise = "An unrelated introduction about baking bread. " * 130 + fact
    excerpt = relevant_excerpt(
        premise, "Spectral lines shift to longer wavelengths because space expands."
    )
    assert len(excerpt) <= 2400 and fact in excerpt


def test_embedding_cache_deduplicates_and_returns_independent_vectors(monkeypatch):
    engine = EmbeddingEngine()
    fake = MagicMock(spec=["encode", "get_embedding_dimension"])
    fake.get_embedding_dimension.return_value = 4
    fake.encode.side_effect = lambda texts, **kw: np.ones((len(texts), 4), dtype=np.float32)
    monkeypatch.setattr(engine, "_ensure_model", lambda: fake)
    first = engine.encode(["source", "source"])
    assert fake.encode.call_args.args[0] == ["source"]
    first[:] = 0
    second = engine.encode(["source"])
    assert np.all(second == 1) and fake.encode.call_count == 1
    assert engine.last_encode_stats["encoded"] == 0
