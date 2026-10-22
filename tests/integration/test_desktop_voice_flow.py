"""Hands-free voice + push-to-talk through the real run_desktop_app (simulated mic, scripted STT)."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("numpy")

from test_desktop_query_flow import run_harness  # noqa: E402


def test_wake_word_flows_ignore_chatter_and_push_to_talk_works():
    r = run_harness("_desktop_voice_harness.py", "voice", timeout=180)
    res = r["results"]
    assert res["a_ignored"] is True  # ordinary conversation without the wake word is ignored
    assert res["b_query"] == ["what's the plan"]  # "Hey LORA <query>" in one utterance
    assert res["c_state_armed"] is True  # "Hey LORA" alone arms the next utterance
    assert r["queries"] == ["what's the plan", "tell me a joke", "open the report"]  # follow-up + PTT
    assert r["mic"][-1] == [True, "Fake USB Mic"]  # status line shows the real device


def test_missing_microphone_is_reported_not_hidden():
    r = run_harness("_desktop_voice_harness.py", "nomic", timeout=120)
    assert r["queries"] == []
    ok, text = r["mic"][-1]
    assert ok is False and "no input device" in text
    assert any("microphone isn't running" in m[1] for m in r["messages"])


def test_capture_stays_open_and_transcription_waits_until_execution_finishes():
    r = run_harness("_desktop_voice_harness.py", "busy", timeout=120)
    assert r["queries"] == ["tell me a joke", "how are you"]
    assert r["results"]["no_stt_during_execution"]
    assert r["results"]["deferred_after_answer"]
    assert r["results"]["capture_starts"] == 1
    assert r["results"]["capture_stops_before_exit"] == 0


def test_sleep_keeps_addressed_wake_listening_without_executing_chatter():
    r = run_harness("_desktop_voice_harness.py", "sleepwake", timeout=120)
    assert r["results"]["sleep_chatter_ignored"]
    assert r["queries"] == ["tell me a joke"]
    assert r["results"]["capture_stops_before_exit"] == 0


def test_exit_during_synthesis_has_no_closed_pipe_traceback_or_queued_retries():
    r = run_harness("_desktop_voice_harness.py", "shutdownspeech", timeout=120)
    result = r["results"]
    assert result["synthesis_was_pending"]
    assert result["speech_stopped_before_worker_closed"]
    assert result["speech_calls"] == 1
    assert result["speech_threads_joined"]
    assert result["terminal_errors"] == []
