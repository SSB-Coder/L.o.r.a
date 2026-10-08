"""Unit tests for Gateway QR pairing utility (tests/unit/test_gateway_qr.py)."""

from lora.gateway.qr import generate_pairing_qr_terminal, get_pairing_url


def test_get_pairing_url():
    url = get_pairing_url(ip="192.168.50.10", port=8765, token="test_token_abc")
    assert url == "http://192.168.50.10:8765/app?token=test_token_abc"


def test_generate_pairing_qr_terminal():
    qr_text = generate_pairing_qr_terminal(ip="192.168.50.10", port=8765, token="test_token_abc")
    assert isinstance(qr_text, str)
    assert len(qr_text) > 0
    # Must contain typical QR unicode block characters
    assert "█" in qr_text or "▀" in qr_text or "▄" in qr_text or "#" in qr_text
