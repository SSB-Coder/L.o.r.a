"""Claim-by-claim fact checking of a research answer against the pages it was written from.

Wraps the repo's V1-V8 pipeline (citation check, claim decomposition, number check, groundedness,
LLM escalation). The groundedness step uses a fast word-overlap check by default; set
LORA_VERIFY_NLI=1 to use the cross-encoder NLI model instead (it needs a one-time model download).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from lora.agent.research import Source
from lora.dialogue import strip_think
from lora.grounding.cards import EvidenceCard

if os.environ.get("LORA_VERIFY_NLI") != "1":
    os.environ.setdefault("LORA_VERIFY_OFFLINE", "1")


@dataclass
class Verification:
    verdict: str  # verified | repaired | abstain | unchecked
    score: float
    supported: int
    total: int
    unverified: list[str] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.total > 0 and not self.unverified

    def for_ui(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "supported": self.supported,
            "total": self.total,
            "score": round(self.score, 2),
            "unverified": self.unverified[:6],
        }


def cards_from_sources(sources: list[Source]) -> list[EvidenceCard]:
    cards = []
    for s in sources:
        quote = (s.text or s.snippet).strip()
        if quote:
            cards.append(EvidenceCard(card_id=s.id, claim_fact=s.title, exact_quote=quote, source_url=s.url, title=s.title))
    return cards


class VerifierLLM:
    """The ``generate(prompt)`` interface the pipeline's LLM fact-checker expects, on top of Ollama.

    Only the few claims the fast check could not settle reach the model; ``max_calls`` bounds that.
    Past the budget it raises, and the pipeline falls back to its word-overlap check.
    """

    def __init__(self, client: Any, model: str, options: dict[str, Any] | None = None, max_calls: int = 4) -> None:
        self.client, self.model, self.options = client, model, options or {}
        self.max_calls, self.calls = max_calls, 0

    async def generate(self, prompt: str, **_: Any) -> SimpleNamespace:
        if self.calls >= self.max_calls:
            raise RuntimeError("verifier call budget used")
        self.calls += 1
        text = ""
        async for delta in self.client.stream_chat(
            self.model,
            [{"role": "user", "content": prompt}],
            options={**self.options, "temperature": 0},
            think=False,
            keep_alive="24h",
        ):
            text += delta
        return SimpleNamespace(response=strip_think(text))


async def verify_answer(answer: str, sources: list[Source], llm: VerifierLLM | None = None) -> Verification:
    from lora.verify.pipeline import VerificationPipeline

    cards = cards_from_sources(sources)
    if not cards or not answer.strip():
        return Verification("unchecked", 0.0, 0, 0)
    report = await VerificationPipeline(ollama_client=llm).verify(answer, cards)
    details = list(report.claim_details)
    claims = [{"text": d.claim_text, "status": "supported" if d.is_supported else "unsupported", "reason": d.unsupported_reason} for d in details]
    return Verification(
        verdict=str(report.verdict.value),
        score=float(report.composite_faithfulness_score),
        supported=sum(1 for d in details if d.is_supported),
        total=len(details),
        unverified=list(report.unverified_claims),
        claims=claims,
    )
