"""SQLite checkpoint persistence layer for L.O.R.A Supervisor."""

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lora.supervisor.state import SupervisorState


class SqliteCheckpointer:
    """Stores and restores immutable Supervisor states in a local SQLite database."""

    def __init__(self, db_path: str | Path = "data/supervisor_checkpoints.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS supervisor_checkpoints (
                    session_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    node_name TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (session_id, step_index)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checkpoint_session
                ON supervisor_checkpoints(session_id, step_index)
                """
            )
            conn.commit()

    def save_checkpoint(self, state: SupervisorState) -> int:
        """Serialize and persist the given state at its current step count."""
        state_json = state.model_dump_json()
        now = time.time()
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO supervisor_checkpoints
                (session_id, step_index, node_name, state_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    state.session_id,
                    state.step_count,
                    state.current_node.value,
                    state_json,
                    now,
                ),
            )
            conn.commit()
        return state.step_count

    def get_latest_checkpoint(self, session_id: str) -> SupervisorState | None:
        """Fetch the most recent checkpoint for a given session."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT state_json FROM supervisor_checkpoints
                WHERE session_id = ?
                ORDER BY step_index DESC
                LIMIT 1
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return SupervisorState.model_validate_json(row["state_json"])

    def get_checkpoint_at_step(self, session_id: str, step_index: int) -> SupervisorState | None:
        """Fetch an exact step checkpoint for a given session."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT state_json FROM supervisor_checkpoints
                WHERE session_id = ? AND step_index = ?
                """,
                (session_id, step_index),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return SupervisorState.model_validate_json(row["state_json"])

    def list_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        """List summary metadata of all checkpoints for a session in order."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT step_index, node_name, created_at
                FROM supervisor_checkpoints
                WHERE session_id = ?
                ORDER BY step_index ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "step_index": row["step_index"],
                    "node_name": row["node_name"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    def rollback_to_step(self, session_id: str, step_index: int) -> SupervisorState:
        """Restore state at the specified step and delete subsequent checkpoints."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT state_json FROM supervisor_checkpoints
                WHERE session_id = ? AND step_index = ?
                """,
                (session_id, step_index),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"No checkpoint found for session '{session_id}' at step {step_index}")

            target_state = SupervisorState.model_validate_json(row["state_json"])

            # Delete subsequent steps to clean branch
            conn.execute(
                """
                DELETE FROM supervisor_checkpoints
                WHERE session_id = ? AND step_index > ?
                """,
                (session_id, step_index),
            )
            conn.commit()

        return target_state
