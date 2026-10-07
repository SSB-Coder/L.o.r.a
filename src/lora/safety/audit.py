"""Action audit logging and safety inspection utilities."""

from typing import Any

from lora.core.logging import AuditLogger
from lora.storage.db import DatabaseManager


class ActionAuditor:
    """Coordinates in-memory and database audit logging."""

    def __init__(self, db: DatabaseManager | None = None) -> None:
        self.db = db
        self.logger = AuditLogger()

    async def log_action(
        self,
        task_id: str,
        action: str,
        tool: str,
        args: dict[str, Any],
        tier: int,
        gate_decision: str,
        reason: str | None = None,
        result: str | None = None,
    ) -> dict[str, Any]:
        """Create structured audit event and persist to SQLite if DB is attached."""
        event = self.logger.create_event(
            task_id=task_id,
            action=action,
            tool=tool,
            args=args,
            tier=tier,
            decision=gate_decision,
            reason=reason,
            result=result,
        )

        if self.db is not None:
            await self.db.record_audit_event(
                task_id=task_id,
                action=action,
                tool=tool,
                args=event["args"],
                tier=tier,
                gate_decision=gate_decision,
                result=result,
            )

        return event
