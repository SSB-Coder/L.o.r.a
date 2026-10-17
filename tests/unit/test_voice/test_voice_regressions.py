"""Regression tests for the voice bugs found in the Oct 2026 review."""

import subprocess
import sys
import types
from pathlib import Path

import numpy as np

from lora.agent.commands import parse_command
from lora.voice.capture import AudioCapture
from lora.voice.vad import SpeechState, VoiceActivityDetector
from lora.voice.wakeword import WakeWordDetector

SRC = str(Path(__file__).resolve().parents[3] / "src")


def test_voice_capture_imports_without_pulling_the_tool_stack():
    """`import lora.voice.capture` must not import the planner/Laya/Playwright chain."""
    code = (
        "import sys; import lora.voice.capture, lora.voice.vad, lora.voice.wakeword; "
        "bad = [m for m in ('lora.voice.controller','lora.core.controller','laya') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    res = subprocess.run([sys.executable, "-c", code], env={"PYTHONPATH": SRC}, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_vad_isolated_clicks_never_count_as_speech():
    rng = np.random.default_rng(0)
    vad = VoiceActivityDetector(threshold=0.35, min_speech_duration_seconds=0.25)
    for i in range(600):  # ~19 s: one loud click every 40 frames, silence otherwise
        chunk = (rng.standard_normal(512) * (0.05 if i % 40 == 0 else 0.002)).astype(np.float32)
        state, _ = vad.process_chunk(chunk)
        assert state == SpeechState.SILENCE


def test_vad_tolerates_inter_syllable_dip_during_hey_lora():
    """A 1-2 frame consonant/glottal dip between 'Hey' and 'LORA' must not wipe the recorded onset."""
    vad = VoiceActivityDetector(
        sample_rate=16000,
        threshold=0.35,
        silence_duration_seconds=0.15,
        min_speech_duration_seconds=0.16,
    )
    quiet = np.zeros(512, dtype=np.float32)
    loud = np.ones(512, dtype=np.float32) * 0.05

    # Pre-roll silence
    for _ in range(4):
        vad.process_chunk(quiet)
    # "Hey" (3 frames = 96ms < 160ms)
    for _ in range(3):
        st, _ = vad.process_chunk(loud)
    assert st == SpeechState.SILENCE
    # Brief 2-frame inter-syllable dip between "Hey" and "LORA"
    vad.process_chunk(quiet)
    vad.process_chunk(quiet)
    # "LORA" (4 frames = 128ms -> total speech frames = 7 >= 160ms)
    for _ in range(4):
        st, _ = vad.process_chunk(loud)
    assert st == SpeechState.SPEAKING
    for _ in range(6):
        st, _ = vad.process_chunk(quiet)
    assert st == SpeechState.ENDED
    utterance = vad.get_and_reset_utterance()
    # Utterance includes pre-roll + "Hey" + 2-frame dip + "LORA" + trailing silence
    assert len(utterance) >= (4 + 3 + 2 + 4) * 512


def test_vad_utterance_includes_preroll():
    vad = VoiceActivityDetector(threshold=0.5, silence_duration_seconds=0.1, min_speech_duration_seconds=0.05)
    quiet = np.zeros(512, dtype=np.float32)
    loud = np.ones(512, dtype=np.float32) * 0.1
    for _ in range(5):
        vad.process_chunk(quiet)
    for _ in range(4):
        vad.process_chunk(loud)
    for _ in range(6):
        vad.process_chunk(quiet)
    audio = vad.get_and_reset_utterance()
    assert len(audio) > (4 + 4) * 512  # speech + trailing silence + pre-roll before the first loud chunk


def test_capture_without_sounddevice_reports_why(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import raises ImportError
    cap = AudioCapture()
    assert cap.start() is False
    assert cap.is_recording is True  # legacy meaning: start() was called
    assert cap.is_streaming is False
    assert "sounddevice" in (cap.last_error or "")
    assert cap.status_text() == cap.last_error


def _fake_sounddevice(accept_rates, native=48000):
    mod = types.ModuleType("sounddevice")
    state = {"streams": []}

    class InputStream:
        def __init__(self, device, samplerate, channels, blocksize, dtype, callback):
            if samplerate not in accept_rates:
                raise RuntimeError("Invalid sample rate [PaErrorCode -9997]")
            self.samplerate, self.blocksize, self.callback = samplerate, blocksize, callback
            state["streams"].append(self)

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    mod.InputStream = InputStream
    mod.query_devices = lambda device=None, kind=None: {"name": "Test Mic", "default_samplerate": native}
    return mod, state


def test_capture_falls_back_to_native_rate_and_resamples(monkeypatch):
    mod, state = _fake_sounddevice(accept_rates={48000})
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    cap = AudioCapture(sample_rate=16000, chunk_size=512)
    assert cap.start() is True
    assert cap.is_streaming and cap.device_name == "Test Mic" and cap.native_rate == 48000
    stream = state["streams"][0]
    assert stream.blocksize == 1536  # 512 * 3
    got = []
    cap.register_chunk_callback(got.append)
    block = (np.random.default_rng(0).standard_normal((1536, 1)) * 0.1).astype(np.float32)
    stream.callback(block, 1536, None, None)
    assert len(got) == 1 and abs(len(got[0]) - 512) <= 1
    assert cap.blocks_received == 1 and cap.seconds_since_audio() is not None
    cap.stop()
    assert cap.is_streaming is False


def test_capture_callback_errors_are_contained(monkeypatch):
    mod, state = _fake_sounddevice(accept_rates={16000})
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    cap = AudioCapture()
    cap.start()

    def bad(_chunk):
        raise ValueError("boom")

    ok = []
    cap.register_chunk_callback(bad)
    cap.register_chunk_callback(ok.append)
    state["streams"][0].callback(np.zeros((512, 1), dtype=np.float32), 512, None, None)
    assert len(ok) == 1  # one broken callback must not starve the others


def test_wake_word_match_text():
    det = WakeWordDetector(transcriber=None)
    assert det.match_text("Hey LORA what's the weather", False) == (True, "what's the weather")
    assert det.match_text("hey laura", False) == (True, "")
    assert det.match_text("Hey, Flora, open Chrome", False) == (True, "open Chrome")
    assert det.match_text("Hey, Lauren! What time is it?", False) == (True, "What time is it")
    assert det.match_text("we should get lunch", False) == (False, "")
    assert det.match_text("Awaken Lora", True) == (True, "")
    assert det.match_text("hey lora", True) == (False, "")  # sleeping needs the awaken phrase


def test_power_off_requires_explicit_lora_target():
    """Bare 'turn off' from background speech must not trigger app shutdown."""
    assert parse_command("turn off") is None
    assert parse_command("turn off the lights") is None
    cmd = parse_command("turn off lora")
    assert cmd is not None and cmd.kind == "turn_off"
    cmd2 = parse_command("shut down lora")
    assert cmd2 is not None and cmd2.kind == "turn_off"
