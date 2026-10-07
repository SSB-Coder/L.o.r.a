"""Unit tests for L.O.R.A Filesystem tools (src/lora/tools/files/)."""

from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from lora.safety.policy import PolicyEngine
from lora.tools.files.list import list_dir
from lora.tools.files.paths import open_path
from lora.tools.files.read import read_file
from lora.tools.files.search import search_files


@pytest.fixture
def workspace(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    engine = PolicyEngine(allowed_roots=[str(ws)])
    return ws, engine


def test_list_dir_allowed(workspace):
    ws, engine = workspace
    (ws / "file1.txt").write_text("hello", encoding="utf-8")
    (ws / "subdir").mkdir()

    res = list_dir(str(ws), policy_engine=engine)
    assert res["status"] == "success"
    names = [item["name"] for item in res["items"]]
    assert "file1.txt" in names
    assert "subdir" in names


def test_list_dir_traversal_blocked(workspace, tmp_path: Path):
    _, engine = workspace
    outside = tmp_path / "outside_dir"
    outside.mkdir()

    res = list_dir(str(outside), policy_engine=engine)
    assert res["status"] == "error"
    assert "outside allowed roots" in res["message"].lower() or "traversal" in res["message"].lower()


def test_read_file_text_and_json(workspace):
    ws, engine = workspace
    txt_file = ws / "notes.md"
    txt_file.write_text("# LORA Notes\nContent here.", encoding="utf-8")

    res = read_file(str(txt_file), policy_engine=engine)
    assert res["status"] == "success"
    assert "# LORA Notes" in res["content"]
    assert res["file_type"] == "text"


def test_read_file_pdf(workspace):
    ws, engine = workspace
    pdf_path = ws / "sample.pdf"

    # Create a small valid PDF using PyMuPDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Hello from PyMuPDF page 1")
    doc.save(str(pdf_path))
    doc.close()

    res = read_file(str(pdf_path), policy_engine=engine)
    assert res["status"] == "success"
    assert res["file_type"] == "pdf"
    assert "Hello from PyMuPDF page 1" in res["content"]
    assert len(res["pages"]) == 1


def test_read_file_size_cap_enforced(workspace):
    ws, engine = workspace
    large_file = ws / "huge.txt"
    large_file.write_text("A" * 1000, encoding="utf-8")

    # Pass max_bytes smaller than file
    res = read_file(str(large_file), max_bytes=500, policy_engine=engine)
    assert res["status"] == "error"
    assert "exceeds maximum allowed size" in res["message"].lower()


def test_read_file_blocked_extension(workspace):
    ws, engine = workspace
    exe_file = ws / "hack.exe"
    exe_file.touch()

    res = read_file(str(exe_file), policy_engine=engine)
    assert res["status"] == "error"
    assert "blocklist" in res["message"].lower() or "forbidden" in res["message"].lower()


def test_search_files(workspace):
    ws, engine = workspace
    (ws / "quantum_notes.md").write_text("superposition and entanglement", encoding="utf-8")
    (ws / "neural_nets.txt").write_text("backpropagation", encoding="utf-8")

    # Search by filename
    res_name = search_files(query="quantum", root=str(ws), policy_engine=engine)
    assert res_name["status"] == "success"
    assert len(res_name["matches"]) == 1
    assert "quantum_notes.md" in res_name["matches"][0]["path"]

    # Search by content
    res_content = search_files(query="backpropagation", root=str(ws), policy_engine=engine)
    assert res_content["status"] == "success"
    assert len(res_content["matches"]) == 1
    assert "neural_nets.txt" in res_content["matches"][0]["path"]


@patch("os.startfile", create=True)
def test_open_path_allowed(mock_startfile, workspace):
    ws, engine = workspace
    safe_file = ws / "doc.pdf"
    safe_file.touch()

    res = open_path(str(safe_file), policy_engine=engine)
    assert res["status"] == "success"
    mock_startfile.assert_called_once()


def test_search_files_ignores_pruned_directories(workspace):
    ws, engine = workspace
    git_dir = ws / ".git"
    git_dir.mkdir()
    (git_dir / "secret_quantum.txt").write_text("quantum secrets", encoding="utf-8")

    node_dir = ws / "node_modules"
    node_dir.mkdir()
    (node_dir / "pkg_quantum.txt").write_text("quantum modules", encoding="utf-8")

    res = search_files(query="quantum", root=str(ws), policy_engine=engine)
    assert res["status"] == "success"
    assert len(res["matches"]) == 0


def test_read_file_env_var_expansion(workspace, monkeypatch):
    ws, engine = workspace
    target_file = ws / "env_test.txt"
    target_file.write_text("environment variable expansion content", encoding="utf-8")

    monkeypatch.setenv("LORA_TEST_DIR", str(ws))
    res = read_file("%LORA_TEST_DIR%/env_test.txt", policy_engine=engine)
    assert res["status"] == "success"
    assert "environment variable expansion content" in res["content"]
