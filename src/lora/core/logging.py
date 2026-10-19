"""Structured logging and sensitive data scrubbing for LORA."""

import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Patterns to match sensitive tokens in strings
SENSITIVE_PATTERNS = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE),
    re.compile(r"(LORA_TOKEN\s*=\s*)[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE),
    re.compile(r"([?&]token=)[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key\s*[:=]\s*)[A-Za-z0-9_\-\.]{8,}", re.IGNORECASE),
    re.compile(r"(sk-[A-Za-z0-9_\-]{16,})", re.IGNORECASE),
]

SENSITIVE_KEYS = {
    "token",
    "api_key",
    "apikey",
    "secret",
    "password",
    "credential",
    "auth_token",
    "bearer_token",
}


def setup_logging(log_dir: Path | str | None = None, level: int = logging.INFO) -> None:
    """Configure standard console (and optional file) logging for L.O.R.A."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        try:
            p = Path(log_dir)
            p.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(p / "lora.log", encoding="utf-8"))
        except Exception:
            pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def scrub_sensitive_data(data: Any) -> Any:
    """Recursively scrub sensitive keys and token patterns from strings, dicts, and lists."""
    if isinstance(data, str):
        result = data
        for pattern in SENSITIVE_PATTERNS:
            result = pattern.sub(r"\1[REDACTED]" if r"\1" in pattern.pattern else "[REDACTED]", result)
        return result

    if isinstance(data, Mapping):
        scrubbed_dict = {}
        for k, v in data.items():
            if str(k).lower() in SENSITIVE_KEYS:
                scrubbed_dict[k] = "[REDACTED]"
            else:
                scrubbed_dict[k] = scrub_sensitive_data(v)
        return scrubbed_dict

    if isinstance(data, list):
        return [scrub_sensitive_data(item) for item in data]

    if isinstance(data, tuple):
        return tuple(scrub_sensitive_data(item) for item_ in data for item in (item_,))

    return data


class AuditLogger:
    """Generates structured, scrubbed audit log records."""

    def create_event(
        self,
        task_id: str,
        action: str,
        tool: str,
        args: dict[str, Any],
        tier: int,
        decision: str,
        reason: str | None = None,
        result: str | None = None,
    ) -> dict[str, Any]:
        """Create a timestamped audit event with all arguments scrubbed."""
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "task_id": task_id,
            "action": action,
            "tool": tool,
            "tier": tier,
            "decision": decision,
            "reason": reason,
            "result": result,
            "args": scrub_sensitive_data(args),
        }
