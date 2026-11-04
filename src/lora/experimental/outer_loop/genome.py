"""Prompt Genome, Template Compiler, Edit Validator, and Registry for L.O.R.A Outer Loop."""

import hashlib
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PromptGenome(BaseModel):
    """Immutable representation of an evolvable post-retrieval prompt genome."""

    model_config = ConfigDict(frozen=True)

    system_prompt_base: str = "You are L.O.R.A, a grounded research assistant."
    refinement_rules: tuple[str, ...] = Field(default_factory=tuple)
    few_shot_examples: tuple[dict[str, str], ...] = Field(default_factory=tuple)
    writer_style: str = "concise_academic"
    version: int = 0

    @property
    def content_hash(self) -> str:
        """Deterministic content hash representing this prompt genome snapshot."""
        examples_str = str(sorted([str(sorted(ex.items())) for ex in self.few_shot_examples]))
        payload = (
            f"{self.system_prompt_base}|"
            f"{'#'.join(sorted(self.refinement_rules))}|"
            f"{examples_str}|"
            f"{self.writer_style}|"
            f"{self.version}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class TemplateCompiler:
    """Compiles complete writer prompts by slotting frozen templates and evolvable rules."""

    def compile(
        self,
        genome: PromptGenome,
        user_request: str,
        evidence_cards: list[dict[str, Any]],
    ) -> str:
        """Slot base persona, evidence, rules, and user query into final prompt."""
        # Evidence slot
        cards_text = []
        for i, card in enumerate(evidence_cards, 1):
            title = card.get("source_title", f"Source {i}")
            snippet = card.get("snippet", "")
            cards_text.append(f"[{i}] {title}: {snippet}")
        evidence_block = "\n".join(cards_text) if cards_text else "No evidence cards provided."

        # Rules slot
        rules_block = "\n".join(f"- {rule}" for rule in genome.refinement_rules)

        return (
            f"{genome.system_prompt_base}\n\n"
            f"Style: {genome.writer_style}\n\n"
            f"Refinement Directives:\n{rules_block}\n\n"
            f"Evidence Cards:\n{evidence_block}\n\n"
            f"User Question: {user_request}\n\n"
            f"Task: Synthesize a concise, strictly grounded response based only on the evidence above."
        )


class EditValidator:
    """Enforces strict physical and semantic limits to prevent prompt drift and bloat."""

    def __init__(
        self,
        max_rules_added: int = 2,
        max_chars_added: int = 250,
    ) -> None:
        self.max_rules_added = max_rules_added
        self.max_chars_added = max_chars_added

    def validate_edit(
        self,
        base: PromptGenome,
        candidate: PromptGenome,
    ) -> tuple[bool, str]:
        """Validate proposed prompt mutation against strict constraints."""
        # 1. Base system prompt cannot be mutated
        if candidate.system_prompt_base != base.system_prompt_base:
            return False, "Base system prompt cannot be mutated."

        # 2. Rule count limit
        rule_diff = len(candidate.refinement_rules) - len(base.refinement_rules)
        if rule_diff > self.max_rules_added:
            return False, f"Rule limit exceeded: added {rule_diff} rules (max {self.max_rules_added})."

        # 3. Character length delta limit
        base_chars = sum(len(r) for r in base.refinement_rules)
        candidate_chars = sum(len(r) for r in candidate.refinement_rules)
        char_diff = candidate_chars - base_chars
        if char_diff > self.max_chars_added:
            return False, f"Character limit exceeded: added {char_diff} characters (max {self.max_chars_added})."

        return True, "Valid edit"


class GenomeRegistry:
    """SQLite-backed, version-controlled registry managing genome lineage and rollbacks."""

    def __init__(self, db_path: str | Path = "data/genomes.db") -> None:
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
                CREATE TABLE IF NOT EXISTS genome_registry (
                    content_hash TEXT PRIMARY KEY,
                    parent_hash TEXT,
                    version INTEGER NOT NULL,
                    genome_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.commit()

    def save_genome(self, genome: PromptGenome, parent_hash: str | None = None) -> str:
        """Persist a new genome snapshot into registry."""
        g_hash = genome.content_hash
        json_data = genome.model_dump_json()
        now = time.time()

        with self._lock, self._get_connection() as conn:
            # Mark previous entries as inactive for pointer tracking
            conn.execute("UPDATE genome_registry SET is_active = 0")
            conn.execute(
                """
                INSERT OR REPLACE INTO genome_registry
                (content_hash, parent_hash, version, genome_json, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (g_hash, parent_hash, genome.version, json_data, now),
            )
            conn.commit()

        return g_hash

    def get_genome(self, content_hash: str) -> PromptGenome | None:
        """Retrieve a specific genome snapshot by its content hash."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT genome_json FROM genome_registry WHERE content_hash = ?",
                (content_hash,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return PromptGenome.model_validate_json(row["genome_json"])

    def get_latest_genome(self) -> PromptGenome | None:
        """Fetch the currently active genome snapshot."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT genome_json FROM genome_registry
                WHERE is_active = 1
                ORDER BY version DESC, created_at DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            if not row:
                return None
            return PromptGenome.model_validate_json(row["genome_json"])

    def get_lineage(self) -> list[dict[str, Any]]:
        """List chronological lineage of genome snapshots."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT content_hash, parent_hash, version, created_at, is_active
                FROM genome_registry
                ORDER BY version ASC, created_at ASC
                """
            )
            rows = cursor.fetchall()
            return [
                {
                    "content_hash": row["content_hash"],
                    "parent_hash": row["parent_hash"],
                    "version": row["version"],
                    "created_at": row["created_at"],
                    "is_active": bool(row["is_active"]),
                }
                for row in rows
            ]

    def rollback_to(self, content_hash: str) -> PromptGenome:
        """Roll back active pointer to the specified historical genome."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT genome_json FROM genome_registry WHERE content_hash = ?",
                (content_hash,),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Genome hash '{content_hash}' not found in registry.")

            target_genome = PromptGenome.model_validate_json(row["genome_json"])

            # Set active pointer to this hash
            conn.execute("UPDATE genome_registry SET is_active = 0")
            conn.execute(
                "UPDATE genome_registry SET is_active = 1 WHERE content_hash = ?",
                (content_hash,),
            )
            conn.commit()

        return target_genome
