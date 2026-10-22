"""Contracts for the user-approved CPU voice and conversational behavior."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lora.agent.commands import quick_reply
from lora.dialogue import PERSONA, ConversationHistory, build_messages, temporal_context
from lora.voice.tts import worker
from lora.voice.tts.engine import TtsEngine
from lora.voice.wakeword import SpeakerVerifier, WakeWordDetector


@pytest.mark.parametrize("phrase,expected", [
    ("Hey Laura", ""),
    ("hey lora, what's the time?", "what's the time?"),
    ("Hey, Laura! Tell me a story. Make it funny.", "Tell me a story. Make it funny."),
    ("Hey LORA. First sentence.\nSecond sentence.", "First sentence.\nSecond sentence."),
])
def test_default_transcript_wake_needs_no_weights(phrase, expected):
    stt = Mock()
    stt.transcribe.return_value = SimpleNamespace(text=phrase)
    detector = WakeWordDetector(transcriber=stt, speaker_verifier=SpeakerVerifier(enabled=False))
    assert detector._oww_model is None
    result = detector.process_audio(np.ones(8000, dtype=np.float32))
    assert result.detected and result.extracted_query == expected
    stt.transcribe.assert_called_once()


@pytest.mark.parametrize("phrase", ["Laura", "Hey Dora", "They said hey Laura yesterday", "background TV"])
def test_idle_audio_needs_an_addressed_wake_phrase(phrase):
    detector = WakeWordDetector(speaker_verifier=SpeakerVerifier(enabled=False))
    assert detector.match_text(phrase, False) == (False, "")


def test_sleep_does_not_transcribe_even_with_default_listening():
    stt = Mock()
    detector = WakeWordDetector(transcriber=stt)
    assert not detector.process_audio(np.ones(8000), current_state="sleep").detected
    stt.transcribe.assert_not_called()


def test_kokoro_failure_never_invokes_another_engine(monkeypatch):
    monkeypatch.setattr(worker, "get_kokoro_model", lambda _: None)
    forbidden = Mock(side_effect=AssertionError("Another speech engine was called"))
    for name in ("get_qwen_model", "get_external_server", "_windows_speech_synth", "_fallback_synth"):
        monkeypatch.setattr(worker, name, forbidden)
    engine = TtsEngine()
    assert isinstance(engine.worker, worker.KokoroTtsWorkerProcess)
    result = engine.worker.synthesize("Hello")
    assert result.error and not len(result.samples)
    forbidden.assert_not_called()


def test_kokoro_uses_cpu_session_with_bounded_threads(monkeypatch, tmp_path):
    import kokoro_onnx
    import onnxruntime

    session = Mock()
    model = Mock()
    monkeypatch.setattr(worker, "_CACHED_KOKORO_MODEL", None)
    monkeypatch.setattr(worker, "_KOKORO_STATUS", {"state": "unloaded"})
    monkeypatch.setattr(worker, "find_kokoro_files", lambda _: (tmp_path / "model.onnx", tmp_path / "voices.bin"))
    factory = Mock(return_value=session)
    monkeypatch.setattr(onnxruntime, "InferenceSession", factory)
    monkeypatch.setattr(kokoro_onnx.Kokoro, "from_session", Mock(return_value=model))
    assert worker.get_kokoro_model() is model
    assert factory.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    assert factory.call_args.kwargs["sess_options"].intra_op_num_threads == 2


def test_clock_is_real_local_context_and_companion_persona():
    now = datetime(2026, 10, 22, 1, 23, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    context = temporal_context(now)
    # astimezone reflects the machine's local timezone, preserving the instant.
    local = now.astimezone()
    assert local.strftime("%Y") in context and local.strftime("%z") in context
    assert quick_reply("what time is it", now) == "It's 1:23 AM."
    messages = build_messages("I've had a long day", ConversationHistory())
    assert "Current local time:" in messages[0]["content"]
    assert "companion and capable butler" in PERSONA
    assert "never stiff or formal" in PERSONA


def test_clock_turn_bypasses_all_models_and_remembers_answer():
    from lora.agent.runtime import AgentRuntime

    runtime = object.__new__(AgentRuntime)
    runtime._registry = None
    runtime.history = ConversationHistory()
    runtime.laya_router = Mock()
    runtime.laya_router.predict.side_effect = AssertionError("Clock must not load Laya")
    callback = Mock()
    turn = runtime.run_turn("what time is it", Mock(), callback)
    with pytest.raises(StopIteration) as done:
        turn.send(None)
    result = done.value.value
    assert result.text.startswith("It's ") and result.kind == "smalltalk"
    callback.assert_called_once_with(result.text, True)
    assert runtime.history.messages()[-1]["content"] == result.text


def test_legacy_stock_profile_does_not_force_formal_tone():
    messages = build_messages("Let's chat", ConversationHistory(), profile_context=(
        "- Tone: Professional AI peer.\n- Style: Grounded, precise, highly technical.\n- Likes Python."
    ))
    assert "Professional AI peer" not in messages[0]["content"]
    assert "Likes Python" in messages[0]["content"]
    assert "Casual friend and dependable butler" in messages[0]["content"]
