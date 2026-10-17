"""Egress Control and URL Scrubber for L.O.R.A.

Enforces outbound network boundaries:
1. Blocks forbidden schemes (file://, javascript:, data:, gopher:, etc.).
2. Blocks internal loopbacks, private networks, and gateway admin addresses.
3. Enforces domain allowlists for sensitive actions (e.g., typing, chatbot queries).
4. Scrubs tracking and potential exfiltration query parameters from URLs extracted from untrusted pages.
"""

import ipaddress
import re
import socket
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import BaseModel, Field


class EgressCheckResult(BaseModel):
    allowed: bool
    reason: str
    cleaned_url: str = ""
    domain: str = ""
    is_private_or_loopback: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class EgressFilter:
    """Deterministic egress filter and URL scrubber."""

    BLOCKED_SCHEMES = {"file", "javascript", "data", "gopher", "ftp", "ldap", "dict", "tftp"}

    BLOCKED_ROUTER_HOSTS = {
        "192.168.1.1",
        "192.168.0.1",
        "192.168.50.1",
        "10.0.0.1",
        "routerlogin.net",
        "tplinkwifi.net",
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
    }

    # Tracking and potential exfiltration query parameter names to remove
    TRACKING_PARAMS = {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
        "msclkid",
        "mc_eid",
        "yclid",
        "_hsenc",
        "_hsmi",
    }

    DEFAULT_ALLOWED_TYPING_DOMAINS = [
        "google.com",
        "gemini.google.com",
        "duckduckgo.com",
        "chatgpt.com",
        "claude.ai",
        "perplexity.ai",
        "chat.deepseek.com",
        "wikipedia.org",
        "arxiv.org",
    ]

    def __init__(self, allowed_typing_domains: list[str] | None = None) -> None:
        self.allowed_typing_domains = [
            d.lower() for d in (allowed_typing_domains or self.DEFAULT_ALLOWED_TYPING_DOMAINS)
        ]

    def _is_ip_or_loopback_blocked(self, host: str) -> tuple[bool, str]:
        """Verify host is not loopback, private IP, or gateway."""
        if not host:
            return True, "Host is empty or missing."

        clean = host.strip("[]").lower()

        if (
            clean == "localhost"
            or clean.endswith(".localhost")
            or clean.endswith(".local")
            or clean.endswith(".internal")
            or clean.endswith(".lan")
        ):
            return True, f"Loopback/local domain '{clean}' is blocked."

        if clean in self.BLOCKED_ROUTER_HOSTS:
            return True, f"Known router gateway host '{clean}' is blocked."

        # Numeric / Hex representation
        try:
            val = None
            if clean.isdigit():
                val = int(clean)
            elif clean.startswith("0x") or clean.startswith("0X"):
                val = int(clean, 16)
            if val is not None:
                ip_obj = ipaddress.ip_address(val)
                if (
                    ip_obj.is_loopback
                    or ip_obj.is_private
                    or ip_obj.is_reserved
                    or ip_obj.is_unspecified
                ):
                    return True, f"Private/loopback IP address '{ip_obj}' is blocked."
        except Exception:
            pass

        # Standard IP (IPv4 & IPv6)
        try:
            ip = ipaddress.ip_address(clean)
            if ip.is_loopback or ip.is_unspecified:
                return True, f"Loopback IP '{clean}' is blocked."
            if ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                return True, f"Internal/private/link-local IP '{clean}' is blocked."
        except ValueError:
            pass

        # Socket inet_aton for octal or dotted variants
        try:
            packed = socket.inet_aton(clean)
            ip_str = socket.inet_ntoa(packed)
            ip_sock = ipaddress.ip_address(ip_str)
            if ip_sock.is_loopback or ip_sock.is_unspecified:
                return True, f"Loopback IP '{ip_str}' is blocked."
            if ip_sock.is_private or ip_sock.is_link_local or ip_sock.is_multicast or ip_sock.is_reserved:
                return True, f"Internal network address '{ip_str}' is blocked."
        except (OSError, ValueError):
            pass

        return False, ""

    def scrub_url(self, url: str) -> str:
        """Strip tracking parameters and clean URL for safe navigation."""
        try:
            parsed = urlparse(url.strip())
            if not parsed.scheme or not parsed.netloc:
                return url.strip()

            query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
            filtered_pairs = []

            for k, v in query_pairs:
                k_lower = k.lower()
                # 1. Skip tracking parameters
                if k_lower in self.TRACKING_PARAMS:
                    continue
                # 2. Block potential exfiltration payloads: suspiciously long hex or base64 (>200 chars)
                if len(v) > 200 and (re.fullmatch(r"[A-Za-z0-9+/=_-]+", v)):
                    continue
                filtered_pairs.append((k, v))

            new_query = urlencode(filtered_pairs)
            return urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment,
            ))
        except Exception:
            return url.strip()

    def check_egress(
        self,
        url: str,
        require_typing_allowlist: bool = False,
    ) -> EgressCheckResult:
        """Validate whether outbound network navigation or request is permitted."""
        url_clean = url.strip()
        try:
            parsed = urlparse(url_clean)
        except Exception as e:
            return EgressCheckResult(
                allowed=False,
                reason=f"Malformed URL structure: {e}",
            )

        scheme = (parsed.scheme or "").lower()
        if scheme not in ["http", "https"]:
            return EgressCheckResult(
                allowed=False,
                reason=f"Disallowed URL scheme '{scheme}'. Only HTTP and HTTPS are permitted.",
            )

        host = (parsed.hostname or "").lower()
        blocked_ip, reason = self._is_ip_or_loopback_blocked(host)
        if blocked_ip:
            return EgressCheckResult(
                allowed=False,
                reason=reason,
                domain=host,
                is_private_or_loopback=True,
            )

        # Check domain allowlist if typing or sensitive interaction is requested
        if require_typing_allowlist:
            domain_matched = False
            for allowed in self.allowed_typing_domains:
                if host == allowed or host.endswith("." + allowed):
                    domain_matched = True
                    break
            if not domain_matched:
                return EgressCheckResult(
                    allowed=False,
                    reason=f"Domain '{host}' is not on the external typing allowlist.",
                    domain=host,
                )

        cleaned = self.scrub_url(url_clean)

        return EgressCheckResult(
            allowed=True,
            reason="Egress allowed under system policy.",
            cleaned_url=cleaned,
            domain=host,
            is_private_or_loopback=False,
        )
