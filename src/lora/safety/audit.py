"""Security Audit Logger for L.O.R.A.

Records all policy evaluations, gate decisions, and action attempts
with automatic scrubbing of sensitive tokens, secrets, and personal information.
"""

import logging
from typing import Any

from lora.core.logging import scrub_sensitive_data
from lora.storage.db import DatabaseManager

logger = logging.getLogger("lora.safety.audit")


class SecurityAuditLogger:
    """Records audit logs to SQLite and runtime telemetry with strict redaction."""

    def __init__(self, db: DatabaseManager | None = None) -> None:
        self.db = db

    async def log_security_event(
        self,
        task_id: str,
        action: str,
        tool: str,
        tier: int,
        gate_decision: str,
        args: dict[str, Any] | None = None,
        result: str | None = None,
    ) -> str:
        """Record an action audit event with scrubbed parameters."""
        raw_args = args or {}
        scrubbed_args = scrub_sensitive_data(raw_args)
        scrubbed_action = scrub_sensitive_data(action) if action else ""
        scrubbed_result = scrub_sensitive_data(result) if result else None

        log_msg = (
            f"AUDIT [task={task_id}] tool={tool} tier={tier} "
            f"decision={gate_decision} action={scrubbed_action}"
        )
        if gate_decision == "ALLOW":
            logger.info(log_msg)
        else:
            logger.warning(log_msg)

        if self.db is not None:
            return await self.db.record_audit_event(
                task_id=task_id,
                action=scrubbed_action,
                tool=tool,
                args=scrubbed_args,
                tier=tier,
                gate_decision=gate_decision,
                result=scrubbed_result,
            )
        return ""
