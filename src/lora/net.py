"""HTTP client defaults for web research.

On many Windows PCs an antivirus or company proxy re-signs HTTPS traffic with a root certificate that lives in
the *Windows* certificate store but not in Python's bundled one, so pages fail with
"CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate". ``truststore`` makes Python use the
Windows store. If it is not installed we fall back to the default (still verifying, never disabled).
"""

from __future__ import annotations

import logging
import ssl
from typing import Any

import httpx

logger = logging.getLogger(__name__)
_CTX: ssl.SSLContext | None = None


def ssl_context() -> ssl.SSLContext:
    global _CTX
    if _CTX is None:
        try:
            import truststore

            _CTX = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except Exception:
            logger.info("truststore not available; using Python's bundled certificates.")
            _CTX = ssl.create_default_context()
    return _CTX


def async_client(headers: dict[str, str] | None = None, timeout: float | httpx.Timeout = 12.0, **kwargs: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=timeout, verify=ssl_context(), **kwargs)
