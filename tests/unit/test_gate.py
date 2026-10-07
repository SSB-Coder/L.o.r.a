"""Unit tests for L.O.R.A Action Gate & Provenance Tracker (src/lora/safety/gate.py)."""

from pathlib import Path

import pytest

from lora.safety.gate import ActionGate, DataProvenance
from lora.safety.killswitch import KillSwitch
from lora.safety.policy import ActionRiskTier, PolicyEngine, PolicyVerdict


@pytest.fixture
def policy_engine(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return PolicyEngine(allowed_roots=[str(ws)])


@pytest.fixture
def kill_switch():
    ks = KillSwitch()
    ks.reset()
    return ks


@pytest.fixture
def action_gate(policy_engine: PolicyEngine, kill_switch: KillSwitch):
    return ActionGate(
        policy_engine=policy_engine,
        kill_switch=kill_switch,
        max_actions_per_task=40,
        max_repeated_actions=3,
        dry_run=False,
    )


def test_action_gate_normal_tier0_allowed(action_gate: ActionGate):
    res = action_gate.verify_and_gate(
        tool="web_search",
        origin="desktop",
        tier=ActionRiskTier.TIER0,
        params={"query": "quantum computing"},
        provenance={"query": DataProvenance.USER},
    )
    assert res.verdict == PolicyVerdict.ALLOW
    assert res.allowed is True
    assert action_gate.action_count == 1


def test_action_gate_budget_exceeded(action_gate: ActionGate):
    action_gate.action_count = 40
    res = action_gate.verify_and_gate(
        tool="web_search",
        origin="desktop",
        tier=ActionRiskTier.TIER0,
        params={"query": "test"},
    )
    assert res.verdict == PolicyVerdict.HARD_DENY
    assert res.allowed is False
    assert "budget exceeded" in res.reason.lower()


def test_action_gate_repeated_action_loop_detection(action_gate: ActionGate):
    # Repeat the same action 3 times
    for _ in range(3):
        action_gate.verify_and_gate(
            tool="focus_window",
            origin="desktop",
            tier=ActionRiskTier.TIER1,
            params={"title": "Notepad"},
            provenance={"title": DataProvenance.USER},
        )
    # The 4th identical action should trigger ASK
    res4 = action_gate.verify_and_gate(
        tool="focus_window",
        origin="desktop",
        tier=ActionRiskTier.TIER1,
        params={"title": "Notepad"},
        provenance={"title": DataProvenance.USER},
    )
    assert res4.verdict == PolicyVerdict.ASK
    assert "repetitive" in res4.reason.lower() or "loop" in res4.reason.lower()


def test_untrusted_taint_cannot_drive_tier1_action(action_gate: ActionGate):
    """Untrusted text scraped from web/PDF can never be an argument to Tier >= 1 actions."""
    res = action_gate.verify_and_gate(
        tool="open_app",
        origin="desktop",
        tier=ActionRiskTier.TIER1,
        params={"app_name": "notepad"},
        provenance={"app_name": DataProvenance.UNTRUSTED},
    )
    assert res.verdict == PolicyVerdict.HARD_DENY
    assert res.allowed is False
    assert "tainted" in res.reason.lower() or "untrusted" in res.reason.lower()


def test_untrusted_taint_cannot_be_typed_into_browser(action_gate: ActionGate):
    """Untrusted text scraped from external sources cannot be typed into browser inputs."""
    res = action_gate.verify_and_gate(
        tool="browser_type",
        origin="desktop",
        tier=ActionRiskTier.TIER1,
        params={"text": "drop table users;"},
        provenance={"text": DataProvenance.UNTRUSTED},
    )
    assert res.verdict == PolicyVerdict.HARD_DENY
    assert res.allowed is False
    assert "tainted" in res.reason.lower() or "untrusted" in res.reason.lower()


def test_kill_switch_blocks_all_actions(action_gate: ActionGate, kill_switch: KillSwitch):
    kill_switch.trigger("User pressed emergency halt")
    res = action_gate.verify_and_gate(
        tool="web_search",
        origin="desktop",
        tier=ActionRiskTier.TIER0,
        params={"query": "test"},
    )
    assert res.verdict == PolicyVerdict.HARD_DENY
    assert res.allowed is False
    assert "kill switch" in res.reason.lower() or "halted" in res.reason.lower()


def test_dry_run_mode(policy_engine: PolicyEngine, kill_switch: KillSwitch):
    gate = ActionGate(policy_engine=policy_engine, kill_switch=kill_switch, dry_run=True)
    res = gate.verify_and_gate(
        tool="open_app",
        origin="desktop",
        tier=ActionRiskTier.TIER1,
        params={"app_name": "vscode"},
        provenance={"app_name": DataProvenance.USER},
    )
    assert gate.dry_run is True
    assert res.allowed is True
