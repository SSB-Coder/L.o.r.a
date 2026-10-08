"""Unit tests for Gateway Authentication, Rate Limiting, and Rebinding Guards (tests/unit/test_gateway_auth.py)."""

import os
from unittest.mock import patch

from lora.gateway.auth import (
    HostOriginGuard,
    IpAuthRateLimiter,
    TokenValidator,
    scrub_token,
)


def test_token_validator_desktop_and_mobile_isolation():
    with patch.dict(os.environ, {"LORA_DESKTOP_TOKEN": "desk_secret_123", "LORA_MOBILE_TOKEN": "mob_secret_456"}):
        validator = TokenValidator()

        # Desktop listener accepts desktop token, rejects mobile token
        assert validator.validate_token("desk_secret_123", listener_origin="desktop") is True
        assert validator.validate_token("mob_secret_456", listener_origin="desktop") is False
        assert validator.validate_token("wrong_token", listener_origin="desktop") is False

        # Mobile listener accepts mobile token, rejects desktop token
        assert validator.validate_token("mob_secret_456", listener_origin="mobile") is True
        assert validator.validate_token("desk_secret_123", listener_origin="mobile") is False
        assert validator.validate_token("wrong_token", listener_origin="mobile") is False


def test_token_validator_auto_generates_tokens_if_not_set():
    with patch.dict(os.environ, {}, clear=True):
        validator = TokenValidator()
        assert len(validator.desktop_token) >= 16
        assert len(validator.mobile_token) >= 16
        assert validator.desktop_token != validator.mobile_token


def test_ip_rate_limiter_bans_after_5_failures():
    limiter = IpAuthRateLimiter(max_attempts=5, window_seconds=60.0, ban_seconds=900.0)
    test_ip = "192.168.50.55"

    assert limiter.is_banned(test_ip)[0] is False

    for _ in range(4):
        limiter.record_failure(test_ip)
        assert limiter.is_banned(test_ip)[0] is False

    # 5th failure triggers 15 min ban
    limiter.record_failure(test_ip)
    banned, remaining = limiter.is_banned(test_ip)
    assert banned is True
    assert remaining > 800.0


def test_ip_rate_limiter_resets_on_success():
    limiter = IpAuthRateLimiter(max_attempts=5, window_seconds=60.0, ban_seconds=900.0)
    test_ip = "192.168.50.55"

    for _ in range(3):
        limiter.record_failure(test_ip)
    limiter.record_success(test_ip)

    # After reset, failures start from 0
    for _ in range(4):
        limiter.record_failure(test_ip)
    assert limiter.is_banned(test_ip)[0] is False


def test_host_origin_guard_desktop():
    guard = HostOriginGuard(listener_origin="desktop")

    # Allowed desktop hosts
    assert guard.validate_host("127.0.0.1:8765") is True
    assert guard.validate_host("localhost:8765") is True
    assert guard.validate_host("localhost") is True

    # Malicious hosts
    assert guard.validate_host("evil.com") is False
    assert guard.validate_host("192.168.1.100") is False

    # Origins
    assert guard.validate_origin(None) is True
    assert guard.validate_origin("http://127.0.0.1:8765") is True
    assert guard.validate_origin("http://evil.com") is False


def test_host_origin_guard_mobile():
    guard = HostOriginGuard(listener_origin="mobile", mobile_ip="192.168.50.10", port=8765)

    # Allowed mobile host
    assert guard.validate_host("192.168.50.10:8765") is True
    assert guard.validate_host("192.168.50.10") is True

    # Malicious or wrong listener host
    assert guard.validate_host("127.0.0.1:8765") is False
    assert guard.validate_host("attacker.com") is False

    # Origins
    assert guard.validate_origin(None) is True
    assert guard.validate_origin("http://192.168.50.10:8765") is True
    assert guard.validate_origin("http://malicious-site.com") is False


def test_scrub_token():
    text = "Connecting with token desk_secret_998877 and query 'quantum physics'"
    scrubbed = scrub_token(text, known_tokens=["desk_secret_998877"])
    assert "desk_secret_998877" not in scrubbed
    assert "[REDACTED_TOKEN]" in scrubbed
