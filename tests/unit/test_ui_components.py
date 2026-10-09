import os
import sys

# Ensure offscreen platform before creating QApplication
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtWidgets import QApplication

from lora.ui.audio_cues import AudioCueManager, SoundCue
from lora.ui.hud import HydroOrbWidget, LoraDesktopHUD
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

    # Magnetic snap calculation (Primary screen)
    snapped_x, snapped_y = widget.calculate_snap_position(
        target_x=10, target_y=10, screen_width=1920, screen_height=1080, margin=24
    )
    assert snapped_x == 24
    assert snapped_y == 24

    # Multi-monitor snap calculation (Secondary screen at x=1920)
    sec_x, sec_y = widget.calculate_snap_position(
        target_x=1930, target_y=10, screen_width=1920, screen_height=1080, screen_left=1920, screen_top=0, margin=24
    )
    assert sec_x == 1920 + 24
    assert sec_y == 24


def test_hydro_orb_visualizer_and_hud(qapp):
    orb = HydroOrbWidget()
    assert orb.is_active is False
    orb.set_active(True)
    assert orb.is_active is True

    hud = LoraDesktopHUD()
    assert hud.windowTitle() == "L.O.R.A - Desktop HUD"

    # Add message
    hud.append_message("Assistant", "System initialized and verified.")
    feed_text = hud.chat_feed.toPlainText()
    assert "System initialized" in feed_text

    # Verify HTML escaping prevents injection
    hud.append_message("User", "<script>alert('xss')</script><img src='http://evil.com/leak.png'>")
    feed_html = hud.chat_feed.toHtml()
    assert "&lt;script&gt;" in feed_html or "&lt;img" in feed_html
    assert "<script>" not in feed_html

    # Clipboard inspection
    hud.set_input_text("Inspection test query")
    assert hud.input_field.text() == "Inspection test query"


def test_system_tray_creation(qapp):
    tray = LoraSystemTray()
    assert "L.O.R.A" in tray.toolTip()
    # Test sleep toggle notification
    tray.update_state(WidgetState.SLEEPING)
    assert "Sleeping" in tray.toolTip()
    tray.update_state(WidgetState.IDLE)
    assert "Active" in tray.toolTip()
