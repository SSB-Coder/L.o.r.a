"""Offline CPU retrieval smoke/profile against the local weights, using generated sources."""

import json
import os
import time
import zipfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from lora.grounding.embed import EmbeddingEngine  # noqa: E402
from lora.grounding.media_vector_db import MediaVectorDB  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "scratch/retrieval-benchmark"
WORK.mkdir(parents=True, exist_ok=True)


def main():
    md = WORK / "astronomy.md"
    md.write_text(
        "# Redshift\n\n"
        + "Light from distant galaxies shifts toward longer wavelengths as space expands. " * 18
        + "\n\n# Spectra\n\nAstronomers measure redshift using spectral lines.\n",
        encoding="utf-8",
    )
    docx = WORK / "experiment.docx"
    with zipfile.ZipFile(docx, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr(
            "word/document.xml",
            """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>The training experiment used forty epochs.</w:t></w:r></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Accuracy</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Ninety percent</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
</w:body></w:document>""",
        )
    engine = EmbeddingEngine(device="cpu", batch_size=8)
    db = MediaVectorDB(engine, WORK / "index")
    started = time.perf_counter()
    additions = [db.add_file(path) for path in (md, docx)]
    assert all(result["status"] == "success" for result in additions), additions
    assert not engine._use_fallback
    query = "How many epochs were used in the training experiment?"
    first = time.perf_counter()
    hits = db.search(query, top_k=2)
    initial_query = time.perf_counter() - first
    second = time.perf_counter()
    db.search(query, top_k=2)
    repeat_query = time.perf_counter() - second
    assert hits and hits[0]["filename"] == "experiment.docx", hits
    assert engine.last_encode_stats["encoded"] == 0
    result = {
        "device": "cpu",
        "model_path": engine.model_path,
        "fallback": engine._use_fallback,
        "total_seconds": round(time.perf_counter() - started, 3),
        "first_query_seconds": round(initial_query, 4),
        "repeat_query_seconds": round(repeat_query, 4),
        "repeat_query_cache": engine.last_encode_stats,
        "top_source": hits[0]["filename"],
        "sources": db.list_files(),
    }
    target = ROOT / "docs/validation/document-retrieval-cpu.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
