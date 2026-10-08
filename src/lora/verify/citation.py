"""V1: Citation existence and reference integrity check."""

import re

from pydantic import BaseModel, ConfigDict, Field

from lora.grounding.cards import EvidenceCard


class CitationCheckResult(BaseModel):
    """Result of V1 citation and quote existence check."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    forged_citations: list[str] = Field(default_factory=list)
    invalid_quotes: list[str] = Field(default_factory=list)


class CitationExistenceChecker:
    """Verifies that every cited source exists and every quoted string is authentic."""

    def __init__(self) -> None:
        pass

    def check_citations(
        self,
        text: str,
        cards: list[EvidenceCard],
    ) -> CitationCheckResult:
        """Check all bracketed citations and quotes against available cards."""
        card_map = {c.card_id: c for c in cards}

        # 1. Check citations [S1 ...], [S2 ...]
        cited_ids = re.findall(r"\[(S\d+)[^\]]*\]", text)
        forged: list[str] = []
        for cid in cited_ids:
            if cid not in card_map:
                forged.append(cid)

        # 2. Check quoted phrases "..."
        quotes = re.findall(r'"([^"]{6,})"', text)
        invalid_quotes: list[str] = []

        combined_source_quotes = " ".join(c.exact_quote for c in cards).lower()
        for q in quotes:
            cleaned_q = q.strip().lower()
            if cleaned_q not in combined_source_quotes:
                invalid_quotes.append(q)

        is_valid = len(forged) == 0 and len(invalid_quotes) == 0

        return CitationCheckResult(
            is_valid=is_valid,
            forged_citations=forged,
            invalid_quotes=invalid_quotes,
        )
