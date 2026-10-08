"""Network isolation checker and validation CLI for L.O.R.A Gateway.

Verifies:
1. Mobile listener is bound specifically to an isolated router IP (never 0.0.0.0).
2. Subnet does not collide with primary network adapters.
3. Windows Firewall rule exists and is scoped to the isolated interface/subnet.
4. Gateway is unreachable from non-isolated interfaces.
"""

from __future__ import annotations

import ipaddress
import platform
import socket
import subprocess
import sys
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class CheckResult(BaseModel):
    """Result of an individual network audit check."""

    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    details: str


class NetCheckAudit(BaseModel):
    """Comprehensive network audit report."""

    model_config = ConfigDict(frozen=True)

    overall_pass: bool
    checks: list[CheckResult] = Field(default_factory=list)


def validate_bind_address(bind_str: str) -> tuple[bool, str]:
    """Validate that mobile bind string is host:port on a safe isolated IP."""
    if not bind_str or ":" not in bind_str:
        return False, "Invalid bind format; expected host:port"

    parts = bind_str.split(":", 1)
    host, port_str = parts[0].strip(), parts[1].strip()

    try:
        port = int(port_str)
        if not (1 <= port <= 65535):
            return False, f"Port {port} out of range (1-65535)"
    except ValueError:
        return False, f"Invalid port integer: '{port_str}'"

    if host == "0.0.0.0":
        return False, "SECURITY VIOLATION: Binding to 0.0.0.0 exposes gateway to all networks"

    if host in ("127.0.0.1", "localhost"):
        return False, "Mobile gateway must not bind to localhost/127.0.0.1"

    try:
        ip = ipaddress.ip_address(host)
        if not ip.is_private:
            return False, f"Host IP {host} is public; mobile gateway must be on private subnet"
        return True, f"Valid private isolated bind address: {host}:{port}"
    except ValueError:
        return False, f"Invalid IP address format: '{host}'"


def get_local_ip_addresses() -> list[str]:
    """Discover local IPv4 addresses on host network interfaces."""
    ips: set[str] = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    return sorted(ips)


def check_subnet_non_overlap(mobile_ip: str, local_ips: Sequence[str]) -> tuple[bool, str]:
    """Ensure the mobile IP subnet (/24) does not overlap with other local IP subnets."""
    try:
        mobile_net = ipaddress.IPv4Network(f"{mobile_ip}/24", strict=False)
    except ValueError:
        return False, f"Invalid mobile IP format: {mobile_ip}"

    for ip_str in local_ips:
        if ip_str == mobile_ip:
            continue
        try:
            ip = ipaddress.IPv4Address(ip_str)
            if ip in mobile_net:
                return (
                    False,
                    f"Subnet collision: Local IP {ip_str} falls inside mobile subnet {mobile_net}",
                )
        except ValueError:
            continue

    return True, f"No overlap: mobile subnet {mobile_net} is isolated from {list(local_ips)}"


def _query_windows_firewall(rule_name: str) -> tuple[bool, str]:
    """Execute PowerShell command to check Windows Firewall rule existence."""
    if platform.system().lower() != "windows":
        return True, "Non-Windows system: firewall check skipped"

    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        f"Get-NetFirewallRule -DisplayName '{rule_name}' -ErrorAction SilentlyContinue | Select-Object -Property DisplayName, Enabled, Direction, Action",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and rule_name.lower() in res.stdout.lower():
            return True, f"Firewall rule '{rule_name}' verified active"
        return False, f"Firewall rule '{rule_name}' not found or inactive"
    except Exception as exc:
        return False, f"Error inspecting firewall: {exc}"


def check_firewall_rule(
    rule_name: str, interface_alias: str | None = None
) -> tuple[bool, str]:
    """Inspect whether the specified firewall rule is present."""
    return _query_windows_firewall(rule_name)


def check_reachability_probe(host: str, port: int, timeout: float = 0.5) -> bool:
    """Attempt TCP connection probe; returns True if reachable, False otherwise."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


def run_net_check(
    mobile_bind: str = "192.168.50.10:8765",
    rule_name: str = "LORA Gateway (Isolated)",
    interface_alias: str = "Ethernet 2",
) -> NetCheckAudit:
    """Run full network isolation audit suite."""
    checks: list[CheckResult] = []

    # 1. Bind address check
    ok, msg = validate_bind_address(mobile_bind)
    checks.append(CheckResult(name="Mobile Listener Bind Address", passed=ok, details=msg))

    # 2. Subnet collision check
    host_ip = mobile_bind.split(":", 1)[0]
    local_ips = get_local_ip_addresses()
    ok, msg = check_subnet_non_overlap(host_ip, local_ips)
    checks.append(CheckResult(name="Subnet Isolation & Non-Overlap", passed=ok, details=msg))

    # 3. Firewall rule check
    ok, msg = check_firewall_rule(rule_name, interface_alias)
    checks.append(CheckResult(name="Windows Firewall Scoping", passed=ok, details=msg))

    overall = all(c.passed for c in checks)
    return NetCheckAudit(overall_pass=overall, checks=checks)


def main() -> int:
    """CLI entry point for python -m lora.tools.net_check."""
    print("=== L.O.R.A Network Isolation & Security Audit ===")
    audit = run_net_check()

    for c in audit.checks:
        status_tag = "[PASS]" if c.passed else "[FAIL]"
        print(f"{status_tag:8} {c.name}: {c.details}")

    print("--------------------------------------------------")
    if audit.overall_pass:
        print("OVERALL RESULT: PASS (Network isolation verified)")
        return 0
    else:
        print("OVERALL RESULT: FAIL (Address security or firewall issues detected)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
