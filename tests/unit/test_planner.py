"""Unit tests for L.O.R.A Planner and Controller (src/lora/core/planner.py & controller.py)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from lora.core.controller import Controller
from lora.core.planner import ControlPlanner
from lora.models.laya_router import LayaDecision
from lora.safety.gate import ActionGate
from lora.safety.policy import PolicyEngine
from lora.tools.registry import ToolRegistry


@pytest.fixture
def test_setup(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    engine = PolicyEngine(allowed_roots=[str(ws), str(Path.home() / "Downloads")])
    gate = ActionGate(policy_engine=engine)
    registry = ToolRegistry(action_gate=gate)
    planner = ControlPlanner()
    controller = Controller(planner=planner, tool_registry=registry)
    return controller, planner, registry


def test_fast_path_open_app(test_setup):
    _, planner, _ = test_setup
    decision = LayaDecision(intent="control", intent_confidence=0.95)
    plan = planner.plan("open notepad", decision)

    assert len(plan) == 1
    assert plan[0].tool == "open_app"
    assert plan[0].params["name"] == "notepad"


def test_fast_path_open_web_app(test_setup):
    _, planner, _ = test_setup
    decision = LayaDecision(intent="control", intent_confidence=0.95)
    plan = planner.plan("open claude", decision)

    assert len(plan) == 1
    assert plan[0].tool == "open_url"
    assert "claude.ai" in plan[0].params["url"]


def test_fast_path_open_path(test_setup):
    _, planner, _ = test_setup
    decision = LayaDecision(intent="control", intent_confidence=0.95)
    plan = planner.plan("open my downloads folder", decision)

    assert len(plan) == 1
    assert plan[0].tool == "open_path"
    assert "Downloads" in plan[0].params["path"]


def test_fast_path_focus_window(test_setup):
    _, planner, _ = test_setup
    decision = LayaDecision(intent="control", intent_confidence=0.95)
    plan = planner.plan("focus chrome window", decision)

    assert len(plan) == 1
    assert plan[0].tool == "focus_window"
    assert "chrome" in plan[0].params["title_regex"].lower()


@patch("subprocess.Popen")
def test_controller_execution_flow(mock_popen, test_setup):
    controller, _, _ = test_setup
    decision = LayaDecision(intent="control", intent_confidence=0.95)
    result = controller.execute_user_request("open notepad", decision, origin="desktop")

    assert result["status"] == "success"
    assert len(result["steps"]) == 1
    assert result["steps"][0]["tool"] == "open_app"
