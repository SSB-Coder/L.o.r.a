"""Hands-free voice + push-to-talk through the real run_desktop_app (simulated mic, scripted STT)."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("faiss")

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
