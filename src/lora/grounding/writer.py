"""Grounded research synthesis writer using local models."""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lora.grounding.cards import EvidenceCard
from lora.grounding.copycheck import CopyChecker
from lora.grounding.modes import DepthMode, DepthPromptBuilder


class CitationItem(BaseModel):
    """An exact source citation linking an inline reference to an authoritative source."""

    model_config = ConfigDict(frozen=True)

    card_id: str
    title: str
    source_url: str
    page_or_section: str = ""
    tier: int = 2


class GroundedAnswer(BaseModel):
    """The synthesized research answer with citations and verification metadata."""

    model_config = ConfigDict(frozen=True)

    text: str
    depth: DepthMode
    citations: list[CitationItem]
    overlap_ratio: float = 0.0
    unverified_claims: list[str] = Field(default_factory=list)


class ResearchWriter:
    """Synthesizes truthful answers from extracted evidence cards at requested depth."""

    def __init__(
        self,
        ollama_client: Any,
        prompt_builder: DepthPromptBuilder | None = None,
        copy_checker: CopyChecker | None = None,
        model_name: str = "granite4.2:8b",
    ) -> None:
        self.ollama_client = ollama_client
        self.prompt_builder = prompt_builder or DepthPromptBuilder()
        self.copy_checker = copy_checker or CopyChecker()
        self.model_name = model_name

    def _extract_citations(
        self, text: str, cards: list[EvidenceCard]
    ) -> list[CitationItem]:
        card_map = {c.card_id: c for c in cards}
        found_ids = set(re.findall(r"\[(S\d+)[^\]]*\]", text))

        citations: list[CitationItem] = []
        for cid in sorted(found_ids):
            if cid in card_map:
                c = card_map[cid]
                citations.append(
                    CitationItem(
                        card_id=c.card_id,
                        title=c.title,
                        source_url=c.source_url,
                        page_or_section=c.page_or_section,
                        tier=c.tier,
                    )
                )
        return citations

    async def synthesize(
        self,
        query: str,
        cards: list[EvidenceCard],
        depth: DepthMode = DepthMode.STANDARD,
    ) -> GroundedAnswer:
        """Synthesize a source-grounded explanation from evidence cards."""
        if not cards:
            return GroundedAnswer(
                text="I couldn't find reliable sources to answer this question.",
                depth=depth,
                citations=[],
                overlap_ratio=0.0,
            )

        prompt = self.prompt_builder.build_prompt(query=query, cards=cards, depth=depth)

        # Generate answer from local LLM
        generated = await self.ollama_client.generate(prompt=prompt)
        if hasattr(generated, "response"):
            raw_text = generated.response
        elif isinstance(generated, str):
            raw_text = generated
        else:
            raw_text = str(generated)

        # Extract citations
        citations = self._extract_citations(raw_text, cards)
        if not citations and cards:
            # If model forgot bracket tags, append primary source
            first = cards[0]
            citations.append(
                CitationItem(
                    card_id=first.card_id,
                    title=first.title,
                    source_url=first.source_url,
                    page_or_section=first.page_or_section,
                    tier=first.tier,
                )
            )

        # Check for verbatim overlap
        source_quotes = [c.exact_quote for c in cards]
        copy_res = self.copy_checker.check_overlap(
            generated_text=raw_text, source_texts=source_quotes
        )

        return GroundedAnswer(
            text=raw_text.strip(),
            depth=depth,
            citations=citations,
            overlap_ratio=copy_res.overlap_ratio,
        )
