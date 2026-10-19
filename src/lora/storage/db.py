"""Asynchronous SQLite storage layer with WAL mode for LORA."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import os
import sqlite3
try:
    import aiosqlite
    _USE_FALLBACK = False
except ImportError:  # pragma: no cover - stdlib sqlite3 async wrapper fallback
    _USE_FALLBACK = True

if _USE_FALLBACK:
    import types as _types

    class _FallbackCursor:
        def __init__(self, cur: sqlite3.Cursor) -> None:
            self._cur = cur

        @property
        def lastrowid(self) -> int | None:
            return self._cur.lastrowid

        @property
        def rowcount(self) -> int:
            return self._cur.rowcount

        async def fetchone(self) -> Any:
            return self._cur.fetchone()

        async def fetchall(self) -> list[Any]:
            return self._cur.fetchall()

        async def close(self) -> None:
            self._cur.close()

        def __aiter__(self):
            rows = list(self._cur.fetchall())
            async def _gen():
                for r in rows:
                    yield r
            return _gen()

        async def __aenter__(self) -> "_FallbackCursor":
            return self

        async def __aexit__(self, *args: Any) -> None:
            self._cur.close()

    class _FallbackExec:
        def __init__(self, conn: sqlite3.Connection, sql: str, params: Any) -> None:
            self._conn, self._sql, self._params = conn, sql, params
            self._cur: _FallbackCursor | None = None

        def _run(self) -> _FallbackCursor:
            if self._cur is None:
                self._cur = _FallbackCursor(self._conn.execute(self._sql, self._params))
            return self._cur

        def __await__(self):
            async def _c():
                return self._run()
            return _c().__await__()

        async def __aenter__(self) -> _FallbackCursor:
            return self._run()

        async def __aexit__(self, *args: Any) -> None:
            if self._cur is not None:
                await self._cur.close()

    class _FallbackConn:
        def __init__(self, path: str | Path) -> None:
            self._path = str(path)
            self._conn: sqlite3.Connection | None = None
            self.row_factory: Any = None

        def _ensure(self) -> sqlite3.Connection:
            if self._conn is None:
                self._conn = sqlite3.connect(self._path, check_same_thread=False)
                if self.row_factory is not None:
                    self._conn.row_factory = self.row_factory
            return self._conn

        def __setattr__(self, name: str, value: Any) -> None:
            super().__setattr__(name, value)
            if name == "row_factory" and getattr(self, "_conn", None) is not None:
                self._conn.row_factory = value

        def execute(self, sql: str, params: Any = ()) -> _FallbackExec:
            return _FallbackExec(self._ensure(), sql, params)

        async def executescript(self, script: str) -> _FallbackCursor:
            return _FallbackCursor(self._ensure().executescript(script))

        async def commit(self) -> None:
            self._ensure().commit()

        async def close(self) -> None:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

        def __await__(self):
            async def _c():
                self._ensure()
                return self
            return _c().__await__()

        async def __aenter__(self) -> "_FallbackConn":
            self._ensure()
            return self

        async def __aexit__(self, *args: Any) -> None:
            await self.close()

    aiosqlite = _types.ModuleType("aiosqlite")  # type: ignore
    aiosqlite.Row = sqlite3.Row  # type: ignore
    aiosqlite.Connection = _FallbackConn  # type: ignore
    aiosqlite.connect = lambda db, **kw: _FallbackConn(db)  # type: ignore

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS notebooks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    origin TEXT NOT NULL DEFAULT 'desktop',
    visibility TEXT NOT NULL DEFAULT 'all',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    tier INTEGER NOT NULL,
    is_preprint INTEGER NOT NULL DEFAULT 0,
    fetched_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    FOREIGN KEY(notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    page INTEGER,
    section TEXT,
    text TEXT NOT NULL,
    embedding BLOB,
    FOREIGN KEY(source_id) REFERENCES sources(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence_cards (
    id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    quote TEXT NOT NULL,
    paraphrase TEXT NOT NULL,
    tier INTEGER NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS answers (
    id TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    genome TEXT,
    text TEXT NOT NULL,
    score REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    answer_id TEXT NOT NULL,
    text TEXT NOT NULL,
    card_ids TEXT,
    nli_score REAL,
    verifier_verdict TEXT,
    final_verdict TEXT,
    FOREIGN KEY(answer_id) REFERENCES answers(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    task_id TEXT NOT NULL,
    action TEXT NOT NULL,
    tool TEXT NOT NULL,
    tier INTEGER NOT NULL,
    args TEXT,
    gate_decision TEXT NOT NULL,
    result TEXT
);
"""


class DatabaseManager:
    """Manages asynchronous SQLite connections and queries with WAL mode."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def get_connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(self.db_path)
            self._conn.row_factory = aiosqlite.Row
        return self._conn

    async def initialize(self) -> None:
        """Run initial schema migration and ensure WAL mode is active."""
        conn = await self.get_connection()
        if os.environ.get("LORA_SQLITE_NO_WAL", "0") != "1":
            try:
                await conn.execute("PRAGMA journal_mode = WAL")
            except Exception:
                pass
        await conn.executescript(SCHEMA_SQL)
        await conn.commit()

    async def get_journal_mode(self) -> str:
        conn = await self.get_connection()
        async with conn.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
            return row[0] if row else "unknown"

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # Notebook CRUD
    async def create_notebook(
        self,
        title: str,
        origin: str = "desktop",
        visibility: str = "all",
    ) -> str:
        nb_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        conn = await self.get_connection()
        await conn.execute(
            "INSERT INTO notebooks (id, title, origin, visibility, created_at) VALUES (?, ?, ?, ?, ?)",
            (nb_id, title, origin, visibility, now),
        )
        await conn.commit()
        return nb_id

    async def list_notebooks(self, visibility: str = "all") -> list[dict[str, Any]]:
        conn = await self.get_connection()
        if visibility == "mobile_visible":
            query = "SELECT * FROM notebooks WHERE visibility != 'desktop_only' ORDER BY created_at DESC"
            params = ()
        else:
            query = "SELECT * FROM notebooks ORDER BY created_at DESC"
            params = ()

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # Sources & Chunks CRUD
    async def add_source(
        self,
        notebook_id: str,
        url: str,
        title: str,
        tier: int,
        is_preprint: bool = False,
        content_hash: str = "",
    ) -> str:
        src_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        conn = await self.get_connection()
        await conn.execute(
            """INSERT INTO sources
               (id, notebook_id, url, title, tier, is_preprint, fetched_at, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (src_id, notebook_id, url, title, tier, 1 if is_preprint else 0, now, content_hash),
        )
        await conn.commit()
        return src_id

    async def get_sources(self, notebook_id: str) -> list[dict[str, Any]]:
        conn = await self.get_connection()
        async with conn.execute(
            "SELECT * FROM sources WHERE notebook_id = ? ORDER BY fetched_at ASC",
            (notebook_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["is_preprint"] = bool(d["is_preprint"])
                result.append(d)
            return result

    async def add_chunk(
        self,
        source_id: str,
        page: int | None,
        section: str | None,
        text: str,
        embedding: bytes | None = None,
    ) -> str:
        chunk_id = str(uuid.uuid4())
        conn = await self.get_connection()
        await conn.execute(
            """INSERT INTO chunks (id, source_id, page, section, text, embedding)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (chunk_id, source_id, page, section, text, embedding),
        )
        await conn.commit()
        return chunk_id

    async def add_evidence_card(
        self,
        chunk_id: str,
        quote: str,
        paraphrase: str,
        tier: int,
    ) -> str:
        card_id = str(uuid.uuid4())
        conn = await self.get_connection()
        await conn.execute(
            """INSERT INTO evidence_cards (id, chunk_id, quote, paraphrase, tier)
               VALUES (?, ?, ?, ?, ?)""",
            (card_id, chunk_id, quote, paraphrase, tier),
        )
        await conn.commit()
        return card_id

    async def get_evidence_cards(self, notebook_id: str) -> list[dict[str, Any]]:
        conn = await self.get_connection()
        query = """
            SELECT ec.*, c.source_id, c.page, c.section
            FROM evidence_cards ec
            JOIN chunks c ON ec.chunk_id = c.id
            JOIN sources s ON c.source_id = s.id
            WHERE s.notebook_id = ?
        """
        async with conn.execute(query, (notebook_id,)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # Audit logging
    async def record_audit_event(
        self,
        task_id: str,
        action: str,
        tool: str,
        args: dict[str, Any],
        tier: int,
        gate_decision: str,
        result: str | None = None,
    ) -> str:
        log_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        conn = await self.get_connection()
        await conn.execute(
            """INSERT INTO audit_log
               (id, timestamp, task_id, action, tool, tier, args, gate_decision, result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (log_id, now, task_id, action, tool, tier, json.dumps(args), gate_decision, result),
        )
        await conn.commit()
        return log_id

    async def get_audit_events(self, task_id: str) -> list[dict[str, Any]]:
        conn = await self.get_connection()
        async with conn.execute(
            "SELECT * FROM audit_log WHERE task_id = ? ORDER BY timestamp ASC",
            (task_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
