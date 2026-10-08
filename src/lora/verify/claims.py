"""V2: Atomic claim decomposition."""

import re
import uuid

from pydantic import BaseModel, ConfigDict, Field


class AtomicClaim(BaseModel):
    """An individual atomic factual claim with associated citations."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    claim_text: str
    cited_sources: list[str] = Field(default_factory=list)
    has_numbers: bool = False


class ClaimDecomposer:
    """Decomposes an answer into atomic, verifiable factual claims."""

    def __init__(self) -> None:
        pass

    def _contains_numbers(self, text: str) -> bool:
        # Detect digits, percentages, years, currency
        return bool(re.search(r"\b\d+(\.\d+)?%?|\$\d+", text))

    def _split_sentences(self, text: str) -> list[str]:
        # Protect periods inside bracket citations e.g. [S1 p. 2]
        def protect_brackets(m: re.Match) -> str:
            return m.group(0).replace(".", "§DOT§")

        protected = re.sub(r"\[[^\]]*\]", protect_brackets, text)
        sentences = [
            s.replace("§DOT§", ".").strip()
            for s in re.split(
                r"(?<=[.!?])\s+|;\s+|(?<=\])\s*,\s*(?=(?:while|whereas|and|but|however|[A-Z]))",
                protected.strip(),
            )
            if len(s.strip()) > 5
        ]
        return sentences

    def decompose(self, text: str) -> list[AtomicClaim]:
        """Split answer into atomic factual sentences with mapped citations."""
        if not text or not text.strip():
            return []

        raw_sentences = self._split_sentences(text)

        claims: list[AtomicClaim] = []
        for i, s in enumerate(raw_sentences):
            cited = re.findall(r"\[(S\d+)[^\]]*\]", s)
            clean_text = re.sub(r"\[S\d+[^\]]*\]", "", s).strip()
            clean_text = re.sub(r"\s+([.,;:!?])", r"\1", clean_text)
            clean_text = re.sub(r"^(?:while|whereas|and|but|however)\s+", "", clean_text, flags=re.IGNORECASE)
            if clean_text:
                clean_text = clean_text[0].upper() + clean_text[1:]
            has_nums = self._contains_numbers(clean_text)

            claims.append(
                AtomicClaim(
                    claim_id=f"clm-{i+1}-{uuid.uuid4().hex[:4]}",
                    claim_text=clean_text,
                    cited_sources=cited,
                    has_numbers=has_nums,
                )
            )

        return claims
