"""Unit tests for Gateway origin capability gating (tests/unit/test_gateway_origin_gating.py).

Verifies that requests arriving with origin='mobile' (user_remote):
1. Are restricted strictly to Tier 0 research tools.
2. Cannot access local file tools (read_file, list_dir, search_files, open_path).
3. Cannot access OS control tools (open_app, list_windows, focus_window).
4. Are hard-denied by ActionGate and PolicyEngine.
"""

from pathlib import Path

from lora.gateway.gating import get_allowed_tools_for_origin
from lora.safety.gate import ActionGate
from lora.safety.policy import ActionRiskTier, PolicyEngine, PolicyVerdict


def test_allowed_tools_filtering_for_mobile_origin():
    desktop_tools = get_allowed_tools_for_origin("desktop")
    mobile_tools = get_allowed_tools_for_origin("mobile")

    # Desktop has both research and local control/file tools
    assert "read_file" in desktop_tools
    assert "list_dir" in desktop_tools
    assert "search_files" in desktop_tools
    assert "open_app" in desktop_tools
    assert "search_web" in desktop_tools

    # Mobile MUST NOT have local filesystem or OS tools
    assert "read_file" not in mobile_tools
    assert "list_dir" not in mobile_tools
    assert "search_files" not in mobile_tools
    assert "open_app" not in mobile_tools
    assert "open_path" not in mobile_tools
    assert "focus_window" not in mobile_tools

    # Mobile DOES have read-only research tools
    assert "search_web" in mobile_tools
    assert "fetch_pdf" in mobile_tools
    assert "read_web_page" in mobile_tools


def test_action_gate_hard_denies_remote_mobile_local_file_access():
    policy = PolicyEngine(allowed_roots=[str(Path.home() / "Documents")])
    gate = ActionGate(policy_engine=policy)

    # Mobile user attempting read_file
    result = gate.verify_and_gate(
        tool="read_file",
        origin="user_remote",
        tier=ActionRiskTier.TIER1,
        params={"path": str(Path.home() / "Documents" / "notes.txt")},
    )

    assert result.allowed is False
    assert result.verdict == PolicyVerdict.HARD_DENY
    assert "mobile" in result.reason.lower() or "tier" in result.reason.lower()


def test_action_gate_hard_denies_remote_mobile_app_control():
    policy = PolicyEngine()
    gate = ActionGate(policy_engine=policy)

    # Mobile user attempting open_app
    result = gate.verify_and_gate(
        tool="open_app",
        origin="mobile",
        tier=ActionRiskTier.TIER1,
        params={"app_name": "notepad"},
    )

    assert result.allowed is False
    assert result.verdict == PolicyVerdict.HARD_DENY
    assert "mobile" in result.reason.lower() or "forbidden" in result.reason.lower()
