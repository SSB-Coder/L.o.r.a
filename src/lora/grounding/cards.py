"""Evidence cards representation and extraction for grounded synthesis."""

import re
import time

from pydantic import BaseModel, ConfigDict, Field

from lora.grounding.chunk import DocumentChunk


class EvidenceCard(BaseModel):
    """An atomic piece of grounded evidence extracted from an authoritative source."""

    model_config = ConfigDict(frozen=True)

    card_id: str
    claim_fact: str
    exact_quote: str
    source_url: str
    title: str
    page_or_section: str = ""
    paragraph_index: int = 0
    tier: int = 2
    timestamp: float = Field(default_factory=time.time)


class EvidenceCardExtractor:
    """Extracts compact, quote-grounded evidence cards from document chunks."""

    def __init__(self) -> None:
        pass

    def extract_cards(
        self,
        chunks: list[DocumentChunk],
        max_cards: int = 12,
    ) -> list[EvidenceCard]:
        """Extracts up to max_cards structured evidence cards from top chunks."""
        cards: list[EvidenceCard] = []
        card_counter = 1

        for chunk in chunks:
            if len(cards) >= max_cards:
                break

            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", chunk.text.strip())
                if len(s.strip()) > 15
            ]

            if not sentences:
                continue

            # Pick the most informative sentence in the chunk as quote & claim
            chosen_quote = sentences[0]
            section_ref = (
                f"p. {chunk.page_number}"
                if chunk.page_number > 0
                else f"§ {chunk.paragraph_index + 1}"
            )

            cards.append(
                EvidenceCard(
                    card_id=f"S{card_counter}",
                    claim_fact=chosen_quote,
                    exact_quote=chosen_quote,
                    source_url=chunk.source_url,
                    title=chunk.title,
                    page_or_section=section_ref,
                    paragraph_index=chunk.paragraph_index,
                    tier=chunk.tier,
                )
            )
            card_counter += 1

        return cards
