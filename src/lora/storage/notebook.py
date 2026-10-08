"""Persistent research notebook storage using SQLite."""

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

from lora.grounding.cards import EvidenceCard
from lora.grounding.chunk import DocumentChunk

logger = logging.getLogger(__name__)

DEFAULT_NOTEBOOK_DB = Path("data/notebooks/notebooks.db")


class SavedNotebook(BaseModel):
    """A persisted research notebook with sources, chunks, and verified answer."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    answer: str
    created_at: float
    chunks: list[DocumentChunk] = Field(default_factory=list)
    cards: list[EvidenceCard] = Field(default_factory=list)


class NotebookStore:
    """Manages persistent SQLite storage of promoted research notebooks."""

    def __init__(self, db_path: Path = DEFAULT_NOTEBOOK_DB) -> None:
        self.db_path = Path(db_path)
        self._initialized = False

    async def init_db(self) -> None:
        """Create notebook tables if they do not exist."""
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS notebooks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS notebook_chunks (
                    id TEXT PRIMARY KEY,
                    notebook_id TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    paragraph_index INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    tier INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    FOREIGN KEY (notebook_id) REFERENCES notebooks(id)
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS notebook_cards (
                    id TEXT PRIMARY KEY,
                    notebook_id TEXT NOT NULL,
                    claim_fact TEXT NOT NULL,
                    exact_quote TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    page_or_section TEXT NOT NULL,
                    paragraph_index INTEGER NOT NULL,
                    tier INTEGER NOT NULL,
                    FOREIGN KEY (notebook_id) REFERENCES notebooks(id)
                )
                """
            )
            await db.commit()
        self._initialized = True

    async def save_notebook(
        self,
        title: str,
        chunks: list[DocumentChunk],
        cards: list[EvidenceCard],
        answer: str,
        notebook_id: str | None = None,
    ) -> str:
        """Persist a research session to SQLite."""
        await self.init_db()
        nb_id = notebook_id or f"nb-{uuid.uuid4().hex[:8]}"
        now = time.time()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO notebooks (id, title, answer, created_at) VALUES (?, ?, ?, ?)",
                (nb_id, title, answer, now),
            )
            for c in chunks:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO notebook_chunks
                    (id, notebook_id, source_url, title, text, paragraph_index, page_number, token_count, tier, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        c.chunk_id,
                        nb_id,
                        c.source_url,
                        c.title,
                        c.text,
                        c.paragraph_index,
                        c.page_number,
                        c.token_count,
                        c.tier,
                        c.content_hash,
                    ),
                )
            for card in cards:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO notebook_cards
                    (id, notebook_id, claim_fact, exact_quote, source_url, title, page_or_section, paragraph_index, tier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card.card_id,
                        nb_id,
                        card.claim_fact,
                        card.exact_quote,
                        card.source_url,
                        card.title,
                        card.page_or_section,
                        card.paragraph_index,
                        card.tier,
                    ),
                )
            await db.commit()

        logger.info("Saved notebook %s ('%s') with %d chunks and %d cards", nb_id, title, len(chunks), len(cards))
        return nb_id

    async def get_notebook(self, notebook_id: str) -> SavedNotebook | None:
        """Retrieve a saved notebook by ID."""
        await self.init_db()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT id, title, answer, created_at FROM notebooks WHERE id = ?",
                (notebook_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                nb_id, title, answer, created_at = row

            chunks: list[DocumentChunk] = []
            async with db.execute(
                """
                SELECT id, source_url, title, text, paragraph_index, page_number, token_count, tier, content_hash
                FROM notebook_chunks WHERE notebook_id = ?
                """,
                (notebook_id,),
            ) as cursor:
                async for c_row in cursor:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=c_row[0],
                            source_url=c_row[1],
                            title=c_row[2],
                            text=c_row[3],
                            paragraph_index=c_row[4],
                            page_number=c_row[5],
                            token_count=c_row[6],
                            tier=c_row[7],
                            content_hash=c_row[8],
                        )
                    )

            cards: list[EvidenceCard] = []
            async with db.execute(
                """
                SELECT id, claim_fact, exact_quote, source_url, title, page_or_section, paragraph_index, tier
                FROM notebook_cards WHERE notebook_id = ?
                """,
                (notebook_id,),
            ) as cursor:
                async for card_row in cursor:
                    cards.append(
                        EvidenceCard(
                            card_id=card_row[0],
                            claim_fact=card_row[1],
                            exact_quote=card_row[2],
                            source_url=card_row[3],
                            title=card_row[4],
                            page_or_section=card_row[5],
                            paragraph_index=card_row[6],
                            tier=card_row[7],
                        )
                    )

            return SavedNotebook(
                id=nb_id,
                title=title,
                answer=answer,
                created_at=created_at,
                chunks=chunks,
                cards=cards,
            )

    async def list_notebooks(self) -> list[dict[str, Any]]:
        """List summary of all saved research notebooks."""
        await self.init_db()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT id, title, created_at FROM notebooks ORDER BY created_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]
