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


_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

_CONVERSATIONAL_SIGNOFF = re.compile(
    r"\b(?:"
    r"suggest(?:ing)?\s+taking\s+a\s+break|"
    r"rest\s+(?:your\s+)?(?:eyes|mind|head)|"
    r"get\s+(?:some\s+)?(?:sleep|rest)|"
    r"catch\s+(?:some\s+)?(?:sleep|rest|zs)|"
    r"let\s+me\s+know\s+if|"
    r"hope\s+this\s+helps|"
    r"feel\s+free\s+to|"
    r"you(?:'ve|\s+have)\s+got\s+this|"
    r"i(?:'ll|\s+will)\s+be\s+here\s+when|"
    r"ready\s+when\s+you\s+are|"
    r"take\s+care|"
    r"good\s+night"
    r")\b",
    re.IGNORECASE,
)


def extract_citation_ids(text: str) -> list[str]:
    """Extract all citation IDs (`S1`, `S2`, etc.) from `[S1]`, `[S1, S4]`, `[1]`, or superscripts."""
    found: list[str] = []
    for bracket in re.findall(r"\[([^\]]+)\]", text or ""):
        s_ids = re.findall(r"\bS(\d+)\b", bracket, flags=re.IGNORECASE)
        if s_ids:
            for num in s_ids:
                cid = f"S{num}"
                if cid not in found:
                    found.append(cid)
        elif re.fullmatch(r"[\d\s,;]+", bracket.strip()):
            for num in re.findall(r"\b(\d{1,2})\b", bracket):
                if 1 <= int(num) <= 20:
                    cid = f"S{num}"
                    if cid not in found:
                        found.append(cid)
    for sup in re.findall(r"[¹²³⁴⁵⁶⁷⁸⁹]+", text or ""):
        digits = sup.translate(_SUPERSCRIPT_MAP)
        for d in digits:
            if d != "0":
                cid = f"S{d}"
                if cid not in found:
                    found.append(cid)
    return found


class ClaimDecomposer:
    """Decomposes an answer into atomic, verifiable factual claims."""

    def __init__(self) -> None:
        pass

    def _contains_numbers(self, text: str) -> bool:
        return bool(re.search(r"\b\d+(\.\d+)?%?|\$\d+", text))

    def _split_sentences(self, text: str) -> list[str]:
        # Remove fenced code blocks and trailing footnote separator blocks
        cleaned_body = re.sub(r"```[\s\S]*?```", " ", text or "")
        cleaned_body = cleaned_body.split("\n---\n")[0]

        def protect_brackets(m: re.Match) -> str:
            return m.group(0).replace(".", "§DOT§")

        protected = re.sub(r"\[[^\]]*\]", protect_brackets, cleaned_body)
        sentences: list[str] = []
        for line in protected.splitlines():
            line = line.strip()
            if not line:
                continue
            # Skip markdown headings or short bullet titles without assertions
            if re.match(r"^#{1,6}\s+", line):
                continue
            parts = re.split(
                r"(?<=[.!?])\s+|;\s+|(?<=\])\s*,\s*(?=(?:while|whereas|and|but|however|[A-Z]))",
                line,
            )
            for p in parts:
                restored = p.replace("§DOT§", ".").strip()
                if len(restored) > 5:
                    sentences.append(restored)
        return sentences

    def decompose(self, text: str) -> list[AtomicClaim]:
        """Split answer into atomic factual sentences with mapped citations."""
        if not text or not text.strip():
            return []

        raw_sentences = self._split_sentences(text)

        claims: list[AtomicClaim] = []
        for i, s in enumerate(raw_sentences):
            cited = extract_citation_ids(s)
            # Strip markdown bullet markers, bold lead-in labels, and citation brackets
            clean_text = re.sub(r"^[\s>*•-]+|\^\d+[.)]\s+", "", s).strip()
            clean_text = re.sub(r"^\d+[.)]\s+", "", clean_text).strip()
            clean_text = re.sub(r"^\*\*[^*\n]{1,45}?:\*\*\s*", "", clean_text).strip()
            clean_text = re.sub(r"^\*\*[^*\n]{1,45}?\*\*\s*[:—-]\s*", "", clean_text).strip()
            clean_text = re.sub(r"\[(?:S?\d+[^]]*)\]", "", clean_text, flags=re.IGNORECASE).strip()
            clean_text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+", "", clean_text).strip()
            clean_text = re.sub(r"\s+([.,;:!?])", r"\1", clean_text)
            clean_text = re.sub(r"^(?:while|whereas|and|but|however)\s+", "", clean_text, flags=re.IGNORECASE)
            clean_text = clean_text.strip(" *•-")
            # Skip short section titles / headings (e.g., "Hybrid Architectures") that have no factual predicate
            words = clean_text.split()
            if not clean_text or (len(words) <= 3 and not cited and not clean_text.endswith(".")):
                continue
            if clean_text.endswith(":"):
                continue

            # Skip uncited conversational sign-offs, friendly nudges, or persona rest advice (not factual claims)
            if not cited and _CONVERSATIONAL_SIGNOFF.search(clean_text):
                continue

            clean_text = clean_text[0].upper() + clean_text[1:]
            has_nums = self._contains_numbers(clean_text)

            claims.append(
                AtomicClaim(
                    claim_id=f"clm-{len(claims)+1}-{uuid.uuid4().hex[:4]}",
                    claim_text=clean_text,
                    cited_sources=cited,
                    has_numbers=has_nums,
                )
            )

        return claims
