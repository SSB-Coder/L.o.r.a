"""Unit tests for L.O.R.A Security Audit Logger (src/lora/safety/audit.py)."""

from pathlib import Path

import pytest

from lora.safety.audit import SecurityAuditLogger
from lora.storage.db import DatabaseManager


@pytest.fixture
async def test_db(tmp_path: Path):
    db_path = tmp_path / "test_lora.db"
    db = DatabaseManager(db_path=db_path)
    await db.initialize()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_security_audit_logger_record(test_db: DatabaseManager):
    logger = SecurityAuditLogger(db=test_db)
    log_id = await logger.log_security_event(
        task_id="task-42",
        action="policy_check",
        tool="open_app",
        tier=1,
        gate_decision="ALLOW",
        args={"app_name": "vscode", "api_key": "sk-secret12345"},
        result="permitted",
    )
    assert log_id is not None

    events = await test_db.get_audit_events(task_id="task-42")
    assert len(events) == 1
    row = events[0]
    assert row["tool"] == "open_app"
    assert row["tier"] == 1
    assert row["gate_decision"] == "ALLOW"
    assert "sk-secret12345" not in row["args"]
    assert "[REDACTED]" in row["args"]
