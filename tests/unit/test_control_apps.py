"""Unit tests for L.O.R.A App and URL launching tools (src/lora/tools/control/apps.py)."""

from unittest.mock import patch

import pytest

from lora.tools.control.apps import AppLauncher


@pytest.fixture
def launcher():
    return AppLauncher()


def test_resolve_app_exact_and_alias(launcher: AppLauncher):
    app = launcher.resolve_app("vscode")
    assert app is not None
    assert app["name"] == "Visual Studio Code"
    assert app["command"] == "code"

    alias_app = launcher.resolve_app("vs code")
    assert alias_app is not None
    assert alias_app["command"] == "code"


def test_resolve_unallowlisted_app_rejected(launcher: AppLauncher):
    assert launcher.resolve_app("cmd") is None
    assert launcher.resolve_app("powershell") is None
    assert launcher.resolve_app("regedit") is None
    assert launcher.resolve_app("malware") is None


def test_resolve_web_app_shortcuts(launcher: AppLauncher):
    target = launcher.resolve_web_app("chatgpt")
    assert target is not None
    assert target["url"] == "https://chatgpt.com"

    wiki = launcher.resolve_web_app("wikipedia", query="quantum computing")
    assert wiki is not None
    assert "https://en.wikipedia.org" in wiki["url"]
    assert "quantum+computing" in wiki["url"].lower() or "quantum%20computing" in wiki["url"].lower()


@patch("subprocess.Popen")
def test_open_app_allowed(mock_popen, launcher: AppLauncher):
    res = launcher.open_app("notepad")
    assert res["status"] == "success"
    assert "Notepad" in res["message"]
    mock_popen.assert_called_once()


def test_open_app_denied_not_in_allowlist(launcher: AppLauncher):
    res = launcher.open_app("bash")
    assert res["status"] == "error"
    assert "not on the allowed application list" in res["message"].lower()


@patch("webbrowser.open")
def test_open_url_valid(mock_webbrowser, launcher: AppLauncher, public_dns):
    res = launcher.open_url("https://arxiv.org")
    assert res["status"] == "success"
    assert res["url"] == "https://arxiv.org"
    mock_webbrowser.assert_called_once_with("https://arxiv.org")


def test_open_url_blocked_schemes(launcher: AppLauncher):
    for bad_url in ["file:///C:/Windows/System32", "javascript:alert(1)", "data:text/html,hack"]:
        res = launcher.open_url(bad_url)
        assert res["status"] == "error"
        assert "forbidden" in res["message"].lower() or "invalid" in res["message"].lower()


def test_open_url_blocked_router_ips(launcher: AppLauncher):
    for bad_ip in ["http://192.168.1.1", "http://192.168.0.1", "http://10.0.0.1"]:
        res = launcher.open_url(bad_ip)
        assert res["status"] == "error"
        assert "internal" in res["message"].lower() or "blocked" in res["message"].lower()


def test_open_url_blocked_localhost_and_ssrf(launcher: AppLauncher):
    dangerous_urls = [
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:80",
        "http://0x7f000001:5000",
        "http://2130706433:5000",
        "http://0177.0.0.1:5000",
        "http://admin.local:80",
        "http://sub.localhost:8080",
    ]
    for bad_url in dangerous_urls:
        res = launcher.open_url(bad_url)
        assert res["status"] == "error"
        assert any(term in res["message"].lower() for term in ["prohibited", "blocked", "loopback", "internal"])


def test_zed_aliases_resolve_to_configured_app(launcher):
    for query in ("zed", "Zed Editor", "zed ide"):
        assert launcher.resolve_app(query)["name"] == "Zed"


def test_configured_executable_alternative(monkeypatch, launcher):
    monkeypatch.setattr("sys.platform", "linux")
    with patch("subprocess.Popen", side_effect=[FileNotFoundError(), None]) as launch, patch("shutil.which", return_value=None):
        assert launcher.open_app("zed")["status"] == "success"
    assert [call.args[0] for call in launch.call_args_list] == [["zed"], ["zeditor"]]


def test_configured_desktop_launcher_fallback(tmp_path, monkeypatch, launcher):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    entry = tmp_path / "applications/dev.zed.Zed.desktop"
    entry.parent.mkdir()
    entry.write_text("[Desktop Entry]\nType=Application\nName=Zed\nExec=zed %U\n")
    with patch("subprocess.Popen", side_effect=FileNotFoundError()), patch("shutil.which", return_value=None), patch("subprocess.run") as launch:
        assert launcher.open_app("zed")["status"] == "success"
    assert launch.call_args.args[0] == ["gio", "launch", str(entry)]


def test_desktop_id_cannot_escape_applications_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    (tmp_path / "evil.desktop").write_text("[Desktop Entry]\nExec=sh\n")
    assert AppLauncher._linux_desktop_entry({"desktop_ids": ["../evil.desktop"]}) is None
