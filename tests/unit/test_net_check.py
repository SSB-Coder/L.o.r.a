"""Unit tests for L.O.R.A Network Isolation & Checker (src/lora/tools/net_check.py)."""

from unittest.mock import patch

from lora.tools.net_check import (
    CheckResult,
    NetCheckAudit,
    check_firewall_rule,
    check_reachability_probe,
    check_subnet_non_overlap,
    run_net_check,
    validate_bind_address,
)


def test_validate_bind_address_valid():
    ok, msg = validate_bind_address("192.168.50.10:8765")
    assert ok is True
    assert "Valid" in msg


def test_validate_bind_address_forbidden_all_interfaces():
    ok, msg = validate_bind_address("0.0.0.0:8765")
    assert ok is False
    assert "0.0.0.0" in msg


def test_validate_bind_address_forbidden_localhost_for_mobile():
    ok, msg = validate_bind_address("127.0.0.1:8765")
    assert ok is False
    assert "localhost" in msg.lower() or "127.0.0.1" in msg


def test_validate_bind_address_invalid_format():
    ok, msg = validate_bind_address("not_an_ip")
    assert ok is False
    assert "format" in msg.lower() or "invalid" in msg.lower()


def test_check_subnet_non_overlap():
    mobile_ip = "192.168.50.10"
    home_ips = ["192.168.1.15", "10.0.0.5"]
    ok, msg = check_subnet_non_overlap(mobile_ip, home_ips)
    assert ok is True
    assert "No overlap" in msg


def test_check_subnet_non_overlap_detects_collision():
    mobile_ip = "192.168.1.50"
    home_ips = ["192.168.1.15"]
    ok, msg = check_subnet_non_overlap(mobile_ip, home_ips)
    assert ok is False
    assert "collision" in msg.lower() or "overlap" in msg.lower()


def test_check_reachability_probe_unreachable():
    # Attempting connection to an unused port should return false
    reachable = check_reachability_probe("127.0.0.1", 59999, timeout=0.05)
    assert reachable is False


def test_check_firewall_rule_mock():
    with patch("lora.tools.net_check._query_windows_firewall") as mock_fw:
        mock_fw.return_value = (True, "Rule found and enabled")
        ok, msg = check_firewall_rule("LORA Gateway (Isolated)")
        assert ok is True
        assert "found" in msg.lower()


def test_run_net_check_audit():
    with patch("lora.tools.net_check._query_windows_firewall") as mock_fw:
        mock_fw.return_value = (True, "Firewall rule verified")
        with patch("lora.tools.net_check.get_local_ip_addresses") as mock_ips:
            mock_ips.return_value = ["192.168.1.100"]

            audit: NetCheckAudit = run_net_check(
                mobile_bind="192.168.50.10:8765",
                rule_name="LORA Gateway (Isolated)",
                interface_alias="Ethernet 2",
            )
            assert audit.overall_pass is True
            assert len(audit.checks) >= 3
            for c in audit.checks:
                assert isinstance(c, CheckResult)
                assert c.passed is True
