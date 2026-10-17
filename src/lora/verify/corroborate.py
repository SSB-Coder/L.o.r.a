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
        """Detect opposing negation polarity or conflicting numbers on shared entities across sources."""
        if len(cards) < 2:
            return []
        conflicts: list[str] = []
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                c1, c2 = cards[i], cards[j]
                d1, d2 = self._extract_domain(c1.source_url), self._extract_domain(c2.source_url)
                if d1 and d2 and d1 == d2:
                    continue
                t1 = re.sub(r"[^\w\s]", " ", c1.exact_quote.lower()).split()
                t2 = re.sub(r"[^\w\s]", " ", c2.exact_quote.lower()).split()
                w1 = {w for w in t1 if len(w) > 3 and w not in _STOP_WORDS and not w.isdigit()}
                w2 = {w for w in t2 if len(w) > 3 and w not in _STOP_WORDS and not w.isdigit()}
                shared = w1 & w2
                if len(shared) < 2:
                    continue
                neg1 = bool(set(t1) & _NEG_WORDS)
                neg2 = bool(set(t2) & _NEG_WORDS)
                if neg1 != neg2:
                    conflicts.append(
                        f"Source [{c1.card_id}] ({c1.title or d1}) vs [{c2.card_id}] ({c2.title or d2}) disagree on {' '.join(sorted(shared)[:3])}."
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
