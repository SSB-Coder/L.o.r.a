"""QR pairing utility for L.O.R.A Mobile Gateway."""

from __future__ import annotations

import io

import qrcode


def get_pairing_url(ip: str, port: int, token: str) -> str:
    """Generate the full mobile app pairing URL containing the mobile token."""
    return f"http://{ip}:{port}/app?token={token}"


def generate_pairing_qr_terminal(ip: str, port: int, token: str) -> str:
    """Generate an ANSI/Unicode text QR code suitable for printing in terminal or HUD console."""
    url = get_pairing_url(ip, port, token)
    qr = qrcode.QRCode()
    qr.add_data(url)
    qr.make(fit=True)

    buf = io.StringIO()
    qr.print_ascii(out=buf)
    return buf.getvalue()
