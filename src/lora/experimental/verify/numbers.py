"""V6: Number and entity verification."""

import re

from pydantic import BaseModel, ConfigDict, Field


class NumberCheckResult(BaseModel):
    """Result of number and numerical entity verification."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    mismatched_numbers: list[str] = Field(default_factory=list)


_BENIGN_TECH_TOKENS = re.compile(
    r"\b(?:O\([N12dlog\s^²³*+]+\)|[123]D|S\d+|V[1-8]|L[123]|GPT-\d+(?:\.\d+)?|Llama-?\d+(?:\.\d+)?|Qwen\d*(?:\.\d+)?|Mamba-?[123]|X\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


class NumberEntityChecker:
    """Verifies that all numerical values, percentages, and dates in claims match the source."""

    def __init__(self) -> None:
        pass

    def _extract_numbers(self, text: str) -> list[str]:
        stripped = _BENIGN_TECH_TOKENS.sub(" ", text or "")
        stripped = re.sub(r"\[[^\]]*\]", " ", stripped)
        stripped = re.sub(r"^\s*\d+[.)]\s+", " ", stripped)
        raw = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?[kKmMbB%]?|\b\d+(?:\.\d+)?[kKmMbB%]?|\$\d+(?:\.\d+)?", stripped)
        cleaned: list[str] = []
        for r in raw:
            val = r.replace(",", "").strip()
            if val and not val.upper().startswith("S"):
                cleaned.append(val)
        return cleaned

    def verify_numbers(self, premise: str, claim: str) -> NumberCheckResult:
        """Verify that any number present in the claim appears in the premise."""
        claim_nums = self._extract_numbers(claim)
        if not claim_nums:
            return NumberCheckResult(is_valid=True, mismatched_numbers=[])

        premise_low = (premise or "").lower()
        premise_nums = {n.lower() for n in self._extract_numbers(premise)}
        mismatched: list[str] = []
        for c_num in claim_nums:
            c_low = c_num.lower()
            clean_c = c_low.rstrip("%kmb+").lstrip("$")
            if (
                c_low in premise_nums
                or clean_c in premise_nums
                or clean_c in premise_low
                or c_low in premise_low
            ):
                continue
            # Check shorthand 100k <-> 100,000 / 100000
            if c_low.endswith("k") and clean_c.isdigit():
                expanded = str(int(clean_c) * 1000)
                if expanded in premise_low or f"{clean_c},000" in premise_low:
                    continue
            mismatched.append(c_num)

        is_valid = len(mismatched) == 0
        return NumberCheckResult(
            is_valid=is_valid,
            mismatched_numbers=mismatched,
        )
