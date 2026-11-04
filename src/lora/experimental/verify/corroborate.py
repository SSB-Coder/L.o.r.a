"""V5: Cross-source corroboration and contradiction detector."""

import re
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from lora.grounding.cards import EvidenceCard

_NEG_WORDS = {"not", "never", "no", "false", "failed", "disproved", "rejected", "impossible", "unlikely"}
_STOP_WORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "has", "have", "had",
    "been", "will", "would", "could", "should", "can", "into", "about", "their", "there", "which",
}


class CorroborationResult(BaseModel):
    """Result of cross-source corroboration and contradiction verification."""

    model_config = ConfigDict(frozen=True)

    is_corroborated: bool
    independent_source_count: int
    is_single_source: bool
    contradictions: list[str] = Field(default_factory=list)


class CorroborationChecker:
    """Checks whether critical claims are corroborated or contradicted across independent sources."""

    def __init__(self, min_independent_sources: int = 2) -> None:
        self.min_independent_sources = min_independent_sources

    def _extract_domain(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower() or url.lower()
        except Exception:
            return url.lower()

    def detect_contradictions(self, cards: list[EvidenceCard]) -> list[str]:
        """Flag direct polarity reversals of the same sentence-level assertion.

        Shared topic words plus unrelated negation in a long page are not a
        contradiction. Paraphrased conflicts need semantic review instead.
        """
        if len(cards) < 2:
            return []
        conflicts: list[str] = []
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                c1, c2 = cards[i], cards[j]
                d1, d2 = self._extract_domain(c1.source_url), self._extract_domain(c2.source_url)
                if d1 and d2 and d1 == d2:
                    continue
                def assertions(text):
                    out = []
                    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.lower()):
                        words = re.findall(r"\b[\w'-]+\b", sentence.replace("n't", " not"))
                        negative = any(w in {"not", "never"} for w in words)
                        core = tuple(w for w in words if w not in {"not", "never"})
                        if len(core) >= 4:
                            out.append((core, negative))
                    return out
                conflicts_here = [(a, neg_a) for a, neg_a in assertions(c1.exact_quote)
                                  for b, neg_b in assertions(c2.exact_quote)
                                  if a == b and neg_a != neg_b]
                if conflicts_here:
                    conflicts.append(
                        f"Source [{c1.card_id}] vs [{c2.card_id}] directly oppose: {' '.join(conflicts_here[0][0])}."
                    )
        return conflicts

    def check_corroboration(self, cards: list[EvidenceCard]) -> CorroborationResult:
        """Evaluate if evidence cards represent multiple independent sources and check for conflicts."""
        if not cards:
            return CorroborationResult(
                is_corroborated=False,
                independent_source_count=0,
                is_single_source=False,
                contradictions=[],
            )

        unique_domains = {self._extract_domain(c.source_url) for c in cards if c.source_url}
        count = len(unique_domains)
        is_single = count <= 1
        is_corroborated = count >= self.min_independent_sources
        contradictions = self.detect_contradictions(cards)

        return CorroborationResult(
            is_corroborated=is_corroborated,
            independent_source_count=count,
            is_single_source=is_single,
            contradictions=contradictions,
        )
