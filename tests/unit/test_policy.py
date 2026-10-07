"""Unit tests for L.O.R.A Hard Policy Engine (src/lora/safety/policy.py)."""

from pathlib import Path

import pytest

from lora.safety.policy import (
    ActionRiskTier,
    PolicyEngine,
    PolicyVerdict,
)


@pytest.fixture
def policy_engine(tmp_path: Path):
    """Create a policy engine configured with temporary workspace as allowed root."""
    allowed_roots = [tmp_path / "allowed_workspace", tmp_path / "downloads"]
    for r in allowed_roots:
        r.mkdir(parents=True, exist_ok=True)

    return PolicyEngine(
        allowed_roots=[str(r) for r in allowed_roots],
        safe_extensions=[".txt", ".md", ".pdf", ".py", ".json", ".csv"],
        blocked_extensions=[".exe", ".bat", ".cmd", ".ps1", ".vbs", ".reg", ".dll", ".scr", ".js"],
    )


def test_hard_deny_deletion_tools(policy_engine: PolicyEngine):
    """Verification that any delete, remove, or unlink tool is hard denied."""
    tools_to_test = [
        "delete_file",
        "remove_directory",
        "unlink_path",
        "rmdir",
        "shutil_rmtree",
        "os_remove",
        "wipe_disk",
    ]
    for tool in tools_to_test:
        res = policy_engine.check_action(tool=tool, origin="desktop", tier=ActionRiskTier.TIER3)
        assert res.verdict == PolicyVerdict.HARD_DENY
        assert res.allowed is False
        assert "forbidden" in res.reason.lower() or "denied" in res.reason.lower()


def test_hard_deny_tier3_actions(policy_engine: PolicyEngine):
    """Verification that any Tier 3 action is hard denied regardless of origin."""
    res = policy_engine.check_action(tool="run_shell", origin="desktop", tier=ActionRiskTier.TIER3)
    assert res.verdict == PolicyVerdict.HARD_DENY
    assert res.allowed is False


def test_path_traversal_blocked(policy_engine: PolicyEngine, tmp_path: Path):
    """Verification that paths escaping allowed roots are strictly blocked."""
    evil_paths = [
        tmp_path / "allowed_workspace" / ".." / "secret.txt",
        tmp_path / "outside.txt",
        "C:\\Windows\\System32\\calc.exe",
        "../../etc/passwd",
        "..\\..\\sensitive.key",
    ]
    for p in evil_paths:
        res = policy_engine.check_path(path=p, origin="desktop", mode="read")
        assert res.verdict == PolicyVerdict.HARD_DENY
        assert res.allowed is False
        assert "traversal" in res.reason.lower() or "outside allowed roots" in res.reason.lower() or "forbidden" in res.reason.lower()


def test_blocked_extensions(policy_engine: PolicyEngine, tmp_path: Path):
    """Verification that executable and dangerous scripts are blocked even within allowed roots."""
    allowed_dir = tmp_path / "allowed_workspace"
    for ext in [".exe", ".bat", ".cmd", ".ps1", ".vbs", ".reg", ".dll", ".js"]:
        target = allowed_dir / f"payload{ext}"
        res = policy_engine.check_path(path=target, origin="desktop", mode="read")
        assert res.verdict == PolicyVerdict.HARD_DENY
        assert res.allowed is False
        assert "extension" in res.reason.lower()


def test_safe_file_read_allowed(policy_engine: PolicyEngine, tmp_path: Path):
    """Verification that safe files inside allowed roots are permitted for desktop."""
    safe_file = tmp_path / "allowed_workspace" / "report.pdf"
    safe_file.touch()

    res = policy_engine.check_path(path=safe_file, origin="desktop", mode="read")
    assert res.verdict == PolicyVerdict.ALLOW
    assert res.allowed is True


def test_mobile_origin_cannot_access_local_files(policy_engine: PolicyEngine, tmp_path: Path):
    """Verification that mobile origin requests are strictly forbidden from local filesystem."""
    safe_file = tmp_path / "allowed_workspace" / "notes.md"
    safe_file.touch()

    res = policy_engine.check_path(path=safe_file, origin="mobile", mode="read")
    assert res.verdict == PolicyVerdict.HARD_DENY
    assert res.allowed is False
    assert "mobile" in res.reason.lower()


def test_mobile_origin_cannot_drive_desktop_ui(policy_engine: PolicyEngine):
    """Verification that mobile requests cannot call Tier >= 1 actions or UI tools."""
    for tool in ["open_app", "open_path", "focus_window", "browser_type"]:
        res = policy_engine.check_action(tool=tool, origin="mobile", tier=ActionRiskTier.TIER1)
        assert res.verdict == PolicyVerdict.HARD_DENY
        assert res.allowed is False
        assert "mobile" in res.reason.lower()


def test_mobile_tier0_research_allowed(policy_engine: PolicyEngine):
    """Verification that mobile can execute Tier 0 research tools."""
    research_tools = ["web_search", "fetch_page", "pdf_fetch", "knowledge_query", "notebook_query"]
    for tool in research_tools:
        res = policy_engine.check_action(tool=tool, origin="mobile", tier=ActionRiskTier.TIER0)
        assert res.verdict == PolicyVerdict.ALLOW
        assert res.allowed is True


def test_typing_domain_allowlist(policy_engine: PolicyEngine):
    """Verification that browser typing only allows designated allowlisted domains."""
    assert policy_engine.check_typing_domain("google.com").allowed is True
    assert policy_engine.check_typing_domain("https://chatgpt.com/c/123").allowed is True
    assert policy_engine.check_typing_domain("malicious-site.org").allowed is False
    assert policy_engine.check_typing_domain("http://untrusted-forum.net").allowed is False
