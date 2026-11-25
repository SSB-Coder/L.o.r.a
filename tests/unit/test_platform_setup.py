import io
import json
import sys
from pathlib import Path
from runpy import run_path
from unittest.mock import patch

import pytest

from lora.models.cactus_backend import _peak_rss_kib

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
link_asset = run_path(str(SCRIPTS / "download_cactus_models.py"))["link_asset"]
prepare_chat = run_path(str(SCRIPTS / "setup_runtime.py"))["prepare_chat"]


def test_installer_keeps_an_existing_model():
    models = io.BytesIO(b'{"models":[{"name":"lora-model:latest"}]}')
    with patch("urllib.request.urlopen", return_value=models) as request:
        prepare_chat("http://127.0.0.1:11434/", "lora-model")
    request.assert_called_once_with("http://127.0.0.1:11434/api/tags", timeout=10)


def test_installer_pulls_missing_local_model():
    models = io.BytesIO(b'{"models":[]}')
    progress = io.BytesIO(b'{"status":"downloading"}\n{"status":"success"}\n')
    with patch("urllib.request.urlopen", side_effect=[models, progress]) as request:
        prepare_chat("http://127.0.0.1:11434", "lora-model")
    pull = request.call_args.args[0]
    assert pull.full_url == "http://127.0.0.1:11434/api/pull"
    assert json.loads(pull.data) == {"model": "lora-model", "stream": True}


def test_installer_does_not_pull_to_remote_ollama():
    models = io.BytesIO(b'{"models":[]}')
    with patch("urllib.request.urlopen", return_value=models) as request:
        with pytest.raises(RuntimeError, match="Pull it on that server"):
            prepare_chat("https://models.example", "lora-model")
    assert request.call_count == 1


def test_installer_accepts_a_runtime_fallback_on_remote_ollama(monkeypatch):
    monkeypatch.setenv("LORA_MODEL", "lora-test-fallback")
    models = io.BytesIO(b'{"models":[{"name":"lora-test-fallback:latest"}]}')
    with patch("urllib.request.urlopen", return_value=models) as request:
        prepare_chat("https://models.example", "missing-model")
    assert request.call_count == 1


def test_installer_reports_offline_ollama():
    with patch("urllib.request.urlopen", side_effect=OSError):
        with pytest.raises(RuntimeError, match="Ollama is not reachable"):
            prepare_chat("http://127.0.0.1:11434", "lora-model")


def test_cactus_assets_can_be_installed_without_link_permissions(tmp_path):
    source = tmp_path / "cached.cact"
    source.write_bytes(b"asset")
    target = tmp_path / "local.cact"
    target.write_bytes(b"partial")
    with patch.object(type(target), "symlink_to", side_effect=OSError), patch("os.link", side_effect=OSError):
        link_asset(source, target)
    assert target.read_bytes() == source.read_bytes() and not target.is_symlink()


def test_cactus_memory_metrics_are_optional_without_posix_resource():
    with patch.dict(sys.modules, {"resource": None}):
        assert _peak_rss_kib() is None
