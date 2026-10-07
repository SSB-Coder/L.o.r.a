"""Unit tests for L.O.R.A Tool Registry (src/lora/tools/registry.py)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from lora.safety.gate import ActionGate, DataProvenance
from lora.safety.policy import PolicyEngine
from lora.tools.registry import ToolRegistry


@pytest.fixture
def registry(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    engine = PolicyEngine(allowed_roots=[str(ws)])
    gate = ActionGate(policy_engine=engine)
    return ToolRegistry(action_gate=gate)


def test_registry_has_built_in_tools(registry: ToolRegistry):
    tools = registry.list_tools()
    tool_names = [t["name"] for t in tools]
    assert "open_app" in tool_names
    assert "open_url" in tool_names
    assert "open_path" in tool_names
    assert "list_dir" in tool_names
    assert "read_file" in tool_names
    assert "search_files" in tool_names
    assert "focus_window" in tool_names
    assert "list_windows" in tool_names


@patch("subprocess.Popen")
def test_execute_allowed_tool_via_gate(mock_popen, registry: ToolRegistry):
    res = registry.execute(
        tool_name="open_app",
        params={"name": "notepad"},
        origin="desktop",
        provenance={"name": DataProvenance.USER},
    )
    assert res["status"] == "success"


def test_execute_blocked_tool_via_gate(registry: ToolRegistry):
    # Mobile attempting to open an app is blocked by gate
    res = registry.execute(
        tool_name="open_app",
        params={"name": "notepad"},
        origin="mobile",
    )
    assert res["status"] == "denied"
    assert "forbidden" in res["reason"].lower() or "mobile" in res["reason"].lower()


def test_execute_untrusted_taint_blocked_via_gate(registry: ToolRegistry):
    res = registry.execute(
        tool_name="open_app",
        params={"name": "notepad"},
        origin="desktop",
        provenance={"name": DataProvenance.UNTRUSTED},
    )
    assert res["status"] == "denied"
    assert "untrusted" in res["reason"].lower() or "tainted" in res["reason"].lower()
