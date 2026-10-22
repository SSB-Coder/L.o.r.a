"""HUD hooks added for streaming replies, mic status and the blink/flicker fixes."""

import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from lora.ui.hud import LoraDesktopHUD  # noqa: E402

SHADER_SHA256 = "8663256161d64fe6d87ef787b0098f4e5e6d2d9448045edc312bd4e58daf01f2"  # original committed shader
HTML = (Path(__file__).resolve().parents[2] / "src" / "lora" / "ui" / "droplet.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def controller_without_chromium(monkeypatch):
    """Exercise Qt signals/controller code without launching native Chromium.

    Browser rendering is a separate desktop acceptance check; these tests only
    inspect emitted JS, timers and worker-to-GUI delivery.
    """
    import lora.ui.hud as module

    monkeypatch.setattr(module, "WEB_ENGINE_AVAILABLE", False)


@pytest.fixture
def hud(qapp, monkeypatch):
    h = LoraDesktopHUD()
    calls: list[str] = []
    monkeypatch.setattr(h, "_run_js", calls.append)
    h.js_calls = calls  # type: ignore[attr-defined]
    return h


def test_stream_and_present_calls_are_valid_json_escaped(hud):
    nasty = 'He said "hi"\n</script><b>x</b>'
    hud.stream_begin("q?", "Loading…")
    hud.stream_update(nasty)
    hud.present_answer("q?", nasty, sources=[], confidence=0.0, label="m · 1.0s", streamed=True)
    begin, update, present = hud.js_calls
    assert begin.startswith("window.beginStream")
    assert json.dumps(nasty) in update
    assert json.dumps(nasty) in present and "true, {});" in present and json.dumps("m · 1.0s") in present


def test_mic_status_call(hud):
    hud.set_mic_status(False, "no input device")
    assert hud.js_calls == ['window.setMicStatus && window.setMicStatus(false, "no input device");']


def test_sources_summary_button_uses_the_cancellable_runtime(hud):
    submitted = []
    hud.query_submitted.connect(lambda query, depth: submitted.append((query, depth)))
    hud.media_db = MagicMock()
    hud._handle_web_action("on_report_media", {})
    assert submitted == [("Summarize my sources", "")]
    hud.media_db.report_on_media.assert_not_called()


def test_audio_level_updates_are_throttled(hud):
    for _ in range(50):
        hud.set_audio_level(0.5)
    assert len(hud.js_calls) <= 2
    hud.set_audio_level(0.0)  # silence always goes through
    assert "setAudioLevel" in hud.js_calls[-1]


def test_voice_toggle_action_emits_signal(hud):
    seen = []
    hud.voice_toggled.connect(seen.append)
    hud._handle_web_action("on_voice_toggle", {"enabled": False})
    hud._handle_web_action("on_voice_toggle", {"enabled": True})
    assert seen == [False, True]


def test_qt_space_shortcut_is_gone(hud):
    assert not hasattr(hud, "space_shortcut")
    hud.set_voice_state("thinking")
    pressed = []
    hud.ptt_pressed.connect(lambda: pressed.append(1))
    hud._handle_space_ptt()
    assert pressed == []


def test_page_has_no_remote_resources_and_one_key_handler():
    assert not re.search(r'(?:src|href)="https?://', HTML), "HUD must work offline"
    assert "backdrop-filter" not in HTML  # blur over a live WebGL canvas flickers in QtWebEngine
    assert HTML.count("addEventListener('keydown'") == 1  # duplicate Space handlers re-sent PTT on key repeat
    assert "e.repeat" in HTML
    assert "webglcontextlost" in HTML and "webglcontextrestored" in HTML
    assert "details.src a{color:var(--ac)" in HTML  # source links must be readable on the dark panel
    assert "const esc=" in HTML and "esc(code" in HTML  # model output is escaped before it is rendered


def test_droplet_shader_preserved_with_bounded_render_cost():
    """Keep the original shader while applying the agreed laptop frame budget."""
    import hashlib

    fs = re.search(r"const FS=`.*?`;", HTML, re.S).group(0)
    assert hashlib.sha256(fs.encode()).hexdigest() == SHADER_SHA256
    assert "renderScale=.5" in HTML and "now-last<33" in HTML
    assert "document.hidden||sleepSettled||mode==='off'" in HTML


def test_adding_a_source_never_blocks_the_window(qapp, hud, tmp_path, monkeypatch):
    """Embedding a big file used to run on the UI thread and freeze the whole app."""
    import threading
    import time
    from unittest.mock import MagicMock

    seen_threads: list[threading.Thread] = []

    def slow_add(path):
        seen_threads.append(threading.current_thread())
        time.sleep(0.4)
        return {"status": "ok", "filename": "big.pdf", "chunks_added": 900}

    hud.media_db = MagicMock()
    hud.media_db.add_file.side_effect = slow_add
    hud.media_db.list_files.return_value = []
    notices: list[str] = []
    monkeypatch.setattr(hud, "append_message", lambda sender, text: notices.append(text))

    t0 = time.monotonic()
    hud._handle_web_action("on_add_media_file", {"path": "C:/x/big.pdf"})
    assert time.monotonic() - t0 < 0.1  # returned at once
    assert notices and "Adding big.pdf" in notices[0]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not any("Added" in n for n in notices):
        qapp.processEvents()
        time.sleep(0.01)
    assert any("Added big.pdf" in n for n in notices)
    assert seen_threads and seen_threads[0] is not threading.main_thread()
    hud.media_db.list_files.assert_called()  # the list refreshed after the add


def test_add_failure_is_reported_not_swallowed(qapp, hud, monkeypatch):
    import time
    from unittest.mock import MagicMock

    hud.media_db = MagicMock()
    hud.media_db.add_file.side_effect = RuntimeError("embedding model crashed")
    notices: list[str] = []
    monkeypatch.setattr(hud, "append_message", lambda sender, text: notices.append(text))
    hud._handle_web_action("on_add_media_file", {"path": "C:/x/a.pdf"})
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not any("didn't work" in n for n in notices):
        qapp.processEvents()
        time.sleep(0.01)
    assert any("embedding model crashed" in n for n in notices)


def test_file_chooser_is_deferred_not_opened_inside_the_bridge_call(hud, monkeypatch):
    opened = []
    monkeypatch.setattr(hud, "_browse_media", lambda: opened.append(1))
    hud._handle_web_action("on_browse_media", {})
    assert opened == []  # runs on the next event-loop turn, after the page's call returned


def test_injected_scripts_cannot_collide_in_the_pages_global_scope(qapp):
    """Every call used to run at top level, so a second `const el = ...` threw 'Identifier el has already been declared'."""
    from lora.ui.hud import LoraDesktopHUD

    sent: list[str] = []
    h = LoraDesktopHUD()
    h.web_view = MagicMock()
    h.web_view.page.return_value.runJavaScript.side_effect = sent.append
    h.set_input_text("one")
    h.set_input_text("two")
    assert len(sent) == 2
    for js in sent:
        assert js.startswith("(function(){") and js.rstrip().endswith("})();")
