import os
import sys

# Ensure offscreen platform before creating QApplication
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtWidgets import QApplication

from lora.ui.audio_cues import AudioCueManager, SoundCue
from lora.ui.hud import LoraDesktopHUD
from lora.ui.tray import LoraSystemTray
from lora.ui.widget import AmbientDropletWidget, WidgetState


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_audio_cues_generation():
    manager = AudioCueManager(enabled=True)
    # Generate WAV bytes for each cue
    for cue in (SoundCue.WAKE, SoundCue.SLEEP, SoundCue.COMPLETE, SoundCue.ERROR):
        wav_bytes = manager.generate_cue_wav(cue)
        assert len(wav_bytes) > 44  # Valid WAV header + PCM data
        assert wav_bytes.startswith(b"RIFF")
        assert b"WAVE" in wav_bytes


def test_ambient_droplet_widget(qapp):
    widget = AmbientDropletWidget(idle_opacity=0.35)
    assert widget.state == WidgetState.IDLE
    assert widget.windowOpacity() == pytest.approx(0.35, abs=0.01)

    # Transition to THINKING
    widget.set_state(WidgetState.THINKING)
    assert widget.state == WidgetState.THINKING
    assert widget.windowOpacity() == pytest.approx(0.95, abs=0.01)

    # Transition to SLEEPING
    widget.set_state(WidgetState.SLEEPING)
    assert widget.state == WidgetState.SLEEPING
    assert widget.windowOpacity() == pytest.approx(0.30, abs=0.01)

    # Toggle sleep
    widget.toggle_sleep()
    assert widget.state == WidgetState.IDLE


def test_system_tray_creation(qapp):
    tray = LoraSystemTray()
    assert "L.O.R.A" in tray.toolTip()
    # Test sleep toggle notification
    tray.update_state(WidgetState.SLEEPING)
    assert "Sleeping" in tray.toolTip()
    tray.update_state(WidgetState.IDLE)
    assert "Active" in tray.toolTip()


def test_hud_creation_and_voice_states(qapp):
    hud = LoraDesktopHUD()
    assert hud.windowTitle() == "L.O.R.A"

    # Test voice state changes
    hud.set_voice_state("listening")
    assert hud._current_voice_state == "listening"

    hud.set_voice_state("speaking")
    assert hud._current_voice_state == "speaking"

    hud.set_voice_state("idle")
    assert hud._current_voice_state == "idle"


def test_hud_thinking_state_locks_recording(qapp):
    hud = LoraDesktopHUD()

    # When thinking: PTT should be blocked
    hud.set_voice_state("thinking")
    assert hud._current_voice_state == "thinking"

    ptt_events = []
    hud.ptt_pressed.connect(lambda: ptt_events.append(True))
    hud._handle_space_ptt()
    assert len(ptt_events) == 0

    hud._on_ptt_pressed()
    assert len(ptt_events) == 0

    # Returning to idle allows PTT
    hud.set_voice_state("idle")
    hud._on_ptt_pressed()
    assert len(ptt_events) == 1


def _spin(app, predicate, timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        app.processEvents()
        time.sleep(0.01)


def test_hud_media_db_integration(qapp, tmp_path):
    from lora.grounding.media_vector_db import MediaVectorDB
    from lora.grounding.embed import EmbeddingEngine

    embedder = EmbeddingEngine(model_name_or_path="mock-dim-64")
    embedder._use_fallback = True
    embedder._fallback_dim = 64
    media_db = MediaVectorDB(embedding_engine=embedder, storage_dir=tmp_path)
    hud = LoraDesktopHUD(media_db=media_db)

    # Test adding media file through IPC handler
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Sample media document content", encoding="utf-8")

    hud._handle_web_action("on_add_media_file", {"path": str(sample_file)})
    _spin(qapp, lambda: media_db.file_count == 1)  # indexing now runs on a background thread
    assert media_db.file_count == 1

    # Test reporting on media
    hud._handle_web_action("on_report_media", {"query": "Sample"})
    assert media_db.file_count == 1
