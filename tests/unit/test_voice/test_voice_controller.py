"""Unit tests for PushToTalkListener and VoiceController state machine."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lora.voice.capture import AudioCapture
from lora.voice.controller import VoiceController, VoiceState
from lora.voice.hotkey import PushToTalkListener
from lora.voice.stt import FasterWhisperTranscriber, TranscriptionResult
from lora.voice.tts.engine import TtsEngine
from lora.voice.vad import VoiceActivityDetector
from lora.voice.wakeword import WakeWordDetector


def test_push_to_talk_listener_simulation():
    activated = False
    deactivated = False

    def on_act():
        nonlocal activated
        activated = True

    def on_deact():
        nonlocal deactivated
        deactivated = True

    listener = PushToTalkListener(on_activate=on_act, on_deactivate=on_deact)
    assert listener.is_active is False

    listener.simulate_press()
    assert listener.is_active is True
    assert activated is True

    listener.simulate_release()
    assert listener.is_active is False
    assert deactivated is True


@pytest.mark.asyncio
async def test_voice_controller_state_transitions_and_turn():
    state_history = []

    def state_tracker(new_state: VoiceState):
        state_history.append(new_state)

    capture = AudioCapture(sample_rate=16000)
    vad = VoiceActivityDetector()
    wakeword = WakeWordDetector()

    # Mock STT
    mock_stt = MagicMock(spec=FasterWhisperTranscriber)
    mock_stt.transcribe.return_value = TranscriptionResult(
        text="search wikipedia for general relativity",
        language="en",
        duration_seconds=2.0,
    )

    # Mock TTS
    mock_tts = MagicMock(spec=TtsEngine)
    mock_tts.synthesize_and_speak.return_value = []

    # Mock core controller
    mock_core = MagicMock()
    mock_core.aexecute_user_request = AsyncMock(return_value={
        "status": "success",
        "steps": [{
            "tool": "web_search",
            "result": {"query": "general relativity", "count": 3},
        }],
    })

    hotkey = PushToTalkListener()

    controller = VoiceController(
        capture=capture,
        vad=vad,
        wakeword=wakeword,
        stt=mock_stt,
        tts=mock_tts,
        core_controller=mock_core,
        hotkey_listener=hotkey,
        state_callback=state_tracker,
    )

    assert controller.state == VoiceState.IDLE

    # 1. Simulate Push-to-Talk activation
    hotkey.simulate_press()
    assert controller.state == VoiceState.LISTENING

    hotkey.simulate_release()
    assert controller.state == VoiceState.THINKING

    # 2. Execute process_voice_turn with audio
    fake_audio = np.ones(16000, dtype=np.float32) * 0.1
    res = await controller.process_voice_turn(audio_override=fake_audio)

    assert res["status"] == "success"
    assert "Found 3 sources on general relativity" in res["spoken_response"]
    assert controller.state == VoiceState.IDLE

    # Verify states traversed
    assert VoiceState.LISTENING in state_history
    assert VoiceState.THINKING in state_history
    assert VoiceState.ACTING in state_history
    assert VoiceState.SPEAKING in state_history


def test_voice_controller_start_and_stop():
    controller = VoiceController()
    controller.start()
    assert controller._running is True
    assert controller.capture.is_recording is True

    controller.stop()
    assert controller._running is False
    assert controller.capture.is_recording is False
    assert controller.state == VoiceState.IDLE


def test_voice_controller_ptt_blocked_when_thinking():
    controller = VoiceController()
    controller.state = VoiceState.THINKING

    # Attempt to activate PTT while thinking
    controller._on_ptt_activate()
    # Must remain in THINKING, not change to LISTENING
    assert controller.state == VoiceState.THINKING
    assert controller._ptt_active is False

