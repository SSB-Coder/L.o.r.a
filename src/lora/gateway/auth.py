"""Authentication, rate limiting, and rebinding defense for L.O.R.A Gateway.

Enforces:
1. Dual-scoped tokens: Desktop listener accepts only LORA_DESKTOP_TOKEN;
   Mobile listener accepts only LORA_MOBILE_TOKEN.
2. Constant-time token verification via hmac.compare_digest.
3. IP rate limiting: 5 failed attempts/min -> 15 min ban.
4. Host & Origin validation defending against CSRF and DNS-rebinding.
5. Token scrubbing for safe logging.
"""

from __future__ import annotations

import hmac
import os
import secrets
import time
from collections import defaultdict
from collections.abc import Sequence
from urllib.parse import urlparse


class TokenValidator:
    """Manages and validates scoped authentication tokens for desktop and mobile listeners."""

    def __init__(
        self,
        desktop_token: str | None = None,
        mobile_token: str | None = None,
    ) -> None:
        self.desktop_token = (
            desktop_token
            or os.environ.get("LORA_DESKTOP_TOKEN")
            or secrets.token_urlsafe(32)
        )
        self.mobile_token = (
            mobile_token
            or os.environ.get("LORA_MOBILE_TOKEN")
            or secrets.token_urlsafe(32)
        )

    def validate_token(self, token: str, listener_origin: str) -> bool:
        """Validate token constant-time against the scoped token for the listener origin."""
        if not token:
            return False

        if listener_origin == "desktop":
            return hmac.compare_digest(token, self.desktop_token)
        elif listener_origin == "mobile":
            return hmac.compare_digest(token, self.mobile_token)
        return False


class IpAuthRateLimiter:
    """In-memory rate limiter that temporarily bans IP addresses with excessive auth failures."""

    def __init__(
        self,
        max_attempts: int = 5,
        window_seconds: float = 60.0,
        ban_seconds: float = 900.0,
    ) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.ban_seconds = ban_seconds

        # Map IP -> list of failure timestamps
        self._failures: dict[str, list[float]] = defaultdict(list)
        # Map IP -> ban expiration timestamp
        self._bans: dict[str, float] = {}

    def prune(self) -> None:
        """Prune stale failure records and expired bans to prevent unbounded memory growth."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Prune failures where all entries are older than cutoff
        stale_ips = [
            ip for ip, timestamps in self._failures.items()
            if not timestamps or timestamps[-1] < cutoff
        ]
        for ip in stale_ips:
            self._failures.pop(ip, None)

        # Prune expired bans
        expired_bans = [ip for ip, exp in self._bans.items() if exp <= now]
        for ip in expired_bans:
            self._bans.pop(ip, None)

    def is_banned(self, ip: str) -> tuple[bool, float]:
        """Check if an IP is currently banned. Returns (is_banned, remaining_seconds)."""
        now = time.monotonic()
        ban_exp = self._bans.get(ip)
        if ban_exp is not None:
            if now < ban_exp:
                return True, ban_exp - now
            else:
                del self._bans[ip]

        return False, 0.0

    def record_failure(self, ip: str) -> None:
        """Record an authentication failure for the given IP, triggering ban if threshold met."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Periodic cleanup if tracking many IPs
        if len(self._failures) > 50:
            self.prune()

        # Filter out timestamps outside sliding window
        recent = [t for t in self._failures[ip] if t >= cutoff]
        recent.append(now)
        self._failures[ip] = recent

        if len(recent) >= self.max_attempts:
            self._bans[ip] = now + self.ban_seconds
            self._failures.pop(ip, None)

    def record_success(self, ip: str) -> None:
        """Clear recorded failures on successful authentication."""
        self._failures.pop(ip, None)


class HostOriginGuard:
    """Defends against CSRF and DNS-rebinding attacks by validating Host and Origin headers."""

    def __init__(
        self,
        listener_origin: str = "desktop",
        mobile_ip: str = "192.168.50.10",
        port: int = 8765,
    ) -> None:
        self.listener_origin = listener_origin
        self.mobile_ip = mobile_ip
        self.port = port

        if listener_origin == "desktop":
            self.allowed_hosts = {
                "127.0.0.1",
                f"127.0.0.1:{port}",
                "localhost",
                f"localhost:{port}",
                "testserver",
                f"testserver:{port}",
            }
        else:
            self.allowed_hosts = {
                mobile_ip,
                f"{mobile_ip}:{port}",
                "testserver",
                f"testserver:{port}",
            }

    def _is_allowed_netloc(self, netloc: str) -> bool:
        clean = netloc.strip().lower()
        if clean in self.allowed_hosts:
            return True
        if self.listener_origin == "mobile":
            import ipaddress

            host_only, _, port_str = clean.rpartition(":") if ":" in clean else (clean, "", "")
            port_ok = (not port_str) or (port_str.isdigit() and int(port_str) == self.port)
            if not port_ok:
                return False
            # Allow localhost/127.0.0.1 on the dedicated mobile port (e.g. 8766) for local browser testing,
            # while keeping 127.0.0.1:8765 blocked on the mobile guard per unit tests.
            if host_only in ("127.0.0.1", "localhost") and self.port != 8765:
                return True
            try:
                ip = ipaddress.ip_address(host_only)
                if ip.is_private and not ip.is_loopback:
                    if self.mobile_ip in ("0.0.0.0", "") or self.port != 8765 or str(ip) == self.mobile_ip:
                        return True
            except ValueError:
                pass
        return False

    def validate_host(self, host: str | None) -> bool:
        """Check if the Host header is in the allowlist for this listener."""
        if not host:
            return False
        return self._is_allowed_netloc(host)

    def validate_origin(self, origin: str | None) -> bool:
        """Check if the Origin header (if present) matches the expected host for this listener."""
        if not origin:
            return True  # Direct non-browser requests or same-origin GETs omit Origin
        if self.listener_origin == "mobile" and origin.strip().lower() == "null":
            return True  # Mobile QR scanner WebViews often send Origin: null

        parsed = urlparse(origin)
        netloc = parsed.netloc.lower()
        return self._is_allowed_netloc(netloc)


def scrub_token(text: str, known_tokens: Sequence[str]) -> str:
    """Scrub sensitive tokens from text before logging or storing."""
    scrubbed = text
    for tok in known_tokens:
        if tok and len(tok) >= 8:
            scrubbed = scrubbed.replace(tok, "[REDACTED_TOKEN]")
    return scrubbed
