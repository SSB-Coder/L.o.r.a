"""Opt-in smoke checks for the native components of a fresh desktop install."""

import http.client
import json
import os
import re
import signal
import subprocess
import time
import wave
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("LORA_TEST_NATIVE_INSTALL") != "1", reason="requires installed desktop and browser runtimes")


def test_desktop_launcher_starts_and_stops(tmp_path):
    root = Path(__file__).resolve().parents[2]
    windows = os.name == "nt"
    launcher = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", "run_lora.bat"] if windows else ["./run_lora.sh"]
    profile = ["--config-dir", str(tmp_path / "profile")]
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    log_path = tmp_path / "launcher.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [*launcher, "start", *profile, "--port", "0", "--offline", "--no-audio"],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=not windows,
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                output = log_path.read_text(encoding="utf-8", errors="replace")
                port = re.search(r"Gateway daemon running at http://127\.0\.0\.1:(\d+)", output)
                assert process.poll() is None, output
                if port and "L.O.R.A Desktop Platform initialized successfully." in output:
                    break
                assert time.monotonic() < deadline, output
                time.sleep(0.1)

            connection = http.client.HTTPConnection("127.0.0.1", int(port[1]), timeout=5)
            try:
                connection.request("GET", "/healthz")
                response = connection.getresponse()
                assert response.status == 200 and json.load(response) == {"status": "ok"}
            finally:
                connection.close()

            subprocess.run([*launcher, "stop", *profile], cwd=root, env=env, check=True, timeout=15)
            assert process.wait(timeout=30) == 0, log_path.read_text(encoding="utf-8", errors="replace")
        finally:
            if process.poll() is None:
                if windows:
                    subprocess.run(["taskkill", "/pid", str(process.pid), "/t", "/f"], capture_output=True, timeout=10)
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=10)


def test_hud_loads_in_native_qt_webengine():
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication

    from lora.ui.hud import LoraDesktopHUD

    app = QApplication.instance() or QApplication([])
    hud = LoraDesktopHUD()
    assert isinstance(hud.web_view, QWebEngineView), "Qt WebEngine is unavailable"
    loop = QEventLoop()
    result = []

    def record(value):
        result.append(value)
        loop.quit()

    def check_page(loaded):
        if not loaded:
            record(False)
            return
        hud.web_view.page().runJavaScript(
            "document.body.classList.contains('empty') && !!document.querySelector('#orb') "
            "&& typeof window.beginStream === 'function'",
            record,
        )

    hud.web_view.loadFinished.connect(check_page)
    hud.show()
    QTimer.singleShot(20_000, loop.quit)
    loop.exec()
    hud.close()
    app.processEvents()
    assert result == [True], "WebEngine did not load Lora's real HUD"


def test_chromium_starts_without_network_access():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content("<title>Lora install smoke test</title>")
        assert page.title() == "Lora install smoke test"
        browser.close()


def test_native_audio_dependencies_load(tmp_path):
    import av
    import ctranslate2
    import onnxruntime
    import sounddevice
    from faster_whisper.audio import decode_audio

    assert av.library_versions and ctranslate2.get_supported_compute_types("cpu")
    assert "CPUExecutionProvider" in onnxruntime.get_available_providers()
    sounddevice.query_devices()
    audio = tmp_path / "sample.wav"
    with wave.open(str(audio), "wb") as writer:
        writer.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
        writer.writeframes(bytes(3_200))
    assert decode_audio(str(audio)).shape == (1_600,)
