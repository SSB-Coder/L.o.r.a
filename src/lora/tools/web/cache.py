"""Persistent SQLite-backed HTTP Cache for L.O.R.A Web Connectors."""

import asyncio
import json
import time
from pathlib import Path

import aiosqlite
from pydantic import BaseModel, Field


class CachedResponse(BaseModel):
    url: str
    status_code: int
    headers: dict[str, str] = Field(default_factory=dict)
    body: str
    etag: str = ""
    last_modified: str = ""
    fetched_at: float
    ttl_seconds: float
    is_expired: bool = False


class HttpCache:
    """Stores HTTP responses with TTL and conditional ETag/Last-Modified support."""

    DEFAULT_DB_PATH = Path("cache/http_cache.db")

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None
        self._db_loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()

    async def _get_db(self) -> aiosqlite.Connection:
        loop = asyncio.get_running_loop()
        if self._db is None or self._db_loop != loop:
            async with self._lock:
                if self._db is None or self._db_loop != loop:
                    if self._db:
                        try:
                            await self._db.close()
                        except Exception:
                            pass
                    db = await aiosqlite.connect(self.db_path)
                    await db.execute("PRAGMA journal_mode = WAL")
                    await db.execute("PRAGMA synchronous = NORMAL")
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS http_cache (
                            url TEXT PRIMARY KEY,
                            status_code INTEGER NOT NULL,
                            headers TEXT NOT NULL,
                            body TEXT NOT NULL,
                            etag TEXT DEFAULT '',
                            last_modified TEXT DEFAULT '',
                            fetched_at REAL NOT NULL,
                            ttl_seconds REAL NOT NULL
                        )
                    """)
                    await db.commit()
                    self._db = db
                    self._db_loop = loop
        return self._db

    async def close(self) -> None:
        """Close persistent database connection."""
        async with self._lock:
            if self._db:
                try:
                    await self._db.close()
                except Exception:
                    pass
                self._db = None
                self._db_loop = None

    async def get(self, url: str) -> CachedResponse | None:
        """Fetch cached response for URL, returning None if absent or expired."""
        db = await self._get_db()
        url_clean = url.strip()
        now = time.time()

        async with db.execute(
            "SELECT status_code, headers, body, etag, last_modified, fetched_at, ttl_seconds FROM http_cache WHERE url = ?",
            (url_clean,),
        ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None

                status_code, headers_json, body, etag, last_modified, fetched_at, ttl_seconds = row
                is_expired = (now - fetched_at) > ttl_seconds

                if is_expired:
                    return None

                try:
                    headers = json.loads(headers_json)
                except Exception:
                    headers = {}

                return CachedResponse(
                    url=url_clean,
                    status_code=status_code,
                    headers=headers,
                    body=body,
                    etag=etag or "",
                    last_modified=last_modified or "",
                    fetched_at=fetched_at,
                    ttl_seconds=ttl_seconds,
                    is_expired=False,
                )

    async def put(
        self,
        url: str,
        status_code: int,
        headers: dict[str, str],
        body: str,
        etag: str = "",
        last_modified: str = "",
        ttl_seconds: float = 3600.0,
    ) -> None:
        """Store or update response in SQLite cache."""
        db = await self._get_db()
        url_clean = url.strip()
        now = time.time()
        headers_json = json.dumps(headers)

        await db.execute(
            """
            INSERT INTO http_cache (url, status_code, headers, body, etag, last_modified, fetched_at, ttl_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                status_code = excluded.status_code,
                headers = excluded.headers,
                body = excluded.body,
                etag = excluded.etag,
                last_modified = excluded.last_modified,
                fetched_at = excluded.fetched_at,
                ttl_seconds = excluded.ttl_seconds
            """,
            (url_clean, status_code, headers_json, body, etag, last_modified, now, ttl_seconds),
        )
        await db.commit()

    async def clear_expired(self) -> int:
        """Purge all expired entries from cache and return count deleted."""
        db = await self._get_db()
        now = time.time()
        cursor = await db.execute(
            "DELETE FROM http_cache WHERE (? - fetched_at) > ttl_seconds",
            (now,),
        )
        count = cursor.rowcount
        await db.commit()
        return count
