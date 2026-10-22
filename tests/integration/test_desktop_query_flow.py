"""Real run_desktop_app wiring against a fake Ollama (offscreen Qt, subprocess).

This is the regression test for "the local models don't reply": the answer used to be
posted back with QTimer.singleShot from a worker thread, which never fires, so the HUD
stayed in 'thinking' forever even though Ollama had answered.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
SRC = str(HERE.parents[1] / "src")


def run_harness(script: str, mode: str, timeout: int = 120) -> dict:
    env = {**os.environ, "LORA_SRC": SRC, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": SRC}
    if hasattr(os, "geteuid") and os.geteuid() == 0:  # Chromium refuses to run as root without this (CI/containers)
        env["QTWEBENGINE_CHROMIUM_FLAGS"] = "--no-sandbox --disable-gpu"
    res = subprocess.run(
        [sys.executable, str(HERE / script), mode], env=env, capture_output=True, text=True, timeout=timeout
    )
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("RESULT:")]
    assert lines, f"harness produced no result\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr[-2000:]}"
    return json.loads(lines[-1][len("RESULT:"):])


def test_reply_streams_into_hud_and_returns_to_idle():
    r = run_harness("_desktop_flow_harness.py", "ok")
    assert not r["timeout"]
    assert r["answer"] == "Right away, friend. Anything else?"
    assert r["n_updates"] >= 2 and r["streamed"] is True
    assert r["ui_calls_on_gui_thread"] is True
    assert r["idle_after"] is True
    assert r["sources"] == []  # no fabricated citations
    assert r["label"] == ""  # the UI never shows which model answered


def test_configured_model_name_resolves_and_options_come_from_config():
    r = run_harness("_desktop_flow_harness.py", "ok")
    assert r["model"] == "SparkLLM/Spark-X2.5-4B:latest"  # default 'spark-x25-4b-q4' matched fuzzily
    assert r["think"] is False and r["keep_alive"] == "24h"
    assert r["options"]["num_ctx"] <= 8192
    assert r["roles"] == ["system", "user"]


def test_inline_think_blocks_are_hidden():
    r = run_harness("_desktop_flow_harness.py", "think")
    assert "hmm" not in r["answer"] and "<think>" not in r["answer"]
    assert "hmm" not in (r["last_update"] or "")


def test_ollama_down_shows_actionable_error_and_recovers():
    r = run_harness("_desktop_flow_harness.py", "down")
    assert not r["timeout"]
    assert "can't reach my local engine" in r["answer"]
    assert "spark" not in r["answer"].lower() and "ollama serve" not in r["answer"]
    assert r["idle_after"] is True


def test_interrupt_has_no_ui_traceback_and_accepts_next_turn():
    r = run_harness("_desktop_flow_harness.py", "interrupt")
    assert not r["timeout"] and not r["ui_errors"]
    assert len(r["answers"]) == 2
    assert "stopped" in r["answers"][0].lower()
    assert r["answers"][1] == "Right away, friend. Anything else?"
    assert r["idle_after"]


def test_sleep_during_generation_is_not_overwritten_by_completion():
    r = run_harness("_desktop_flow_harness.py", "sleepbusy")
    assert not r["ui_errors"]
    assert r["states_after_sleep"] == ["sleeping"]
