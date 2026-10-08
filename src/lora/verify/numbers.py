"""V6: Number and entity verification."""

import re

from pydantic import BaseModel, ConfigDict, Field


class NumberCheckResult(BaseModel):
    """Result of number and numerical entity verification."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    mismatched_numbers: list[str] = Field(default_factory=list)


class NumberEntityChecker:
    """Verifies that all numerical values, percentages, and dates in claims match the source."""

    def __init__(self) -> None:
        pass

    def _extract_numbers(self, text: str) -> list[str]:
        # Extract patterns like 1,500, 42.5%, $100, 2024
        raw = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\b\d+(?:\.\d+)?%?|\$\d+(?:\.\d+)?", text)
        cleaned: list[str] = []
        for r in raw:
            # Strip citation artifacts like S1, S2
            val = r.replace(",", "").strip()
            if val and not val.startswith("S"):
                cleaned.append(val)
        return cleaned

    def verify_numbers(self, premise: str, claim: str) -> NumberCheckResult:
        """Verify that any number present in the claim appears in the premise."""
        claim_nums = self._extract_numbers(claim)
        if not claim_nums:
            return NumberCheckResult(is_valid=True, mismatched_numbers=[])

        premise_nums = set(self._extract_numbers(premise))
        # Also check plain text presence without formatting
        mismatched: list[str] = []
        for c_num in claim_nums:
            clean_c = c_num.rstrip("%").lstrip("$")
            if c_num in premise_nums or clean_c in premise or c_num in premise:
                continue
            mismatched.append(c_num)

        is_valid = len(mismatched) == 0
        return NumberCheckResult(
            is_valid=is_valid,
            mismatched_numbers=mismatched,
        )
