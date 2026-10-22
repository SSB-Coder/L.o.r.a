"""Bounded sliding-window instruction scanner for untrusted ingestion."""

import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)
_PATTERNS = (
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|rules)",
    r"(?:system|developer)\s*(?:message|override|prompt)\s*[:=]",
    r"(?:reveal|print|send|exfiltrate)\b.{0,60}\b(?:token|password|secret|system prompt)",
    r"(?:you are now|act as)\s+(?:an?\s+)?(?:system|developer|unrestricted)",
    r"(?:<\|(?:im_start|system)|\[INST\]|</?system>)",
    r"disregard\s+(?:your|the|all)\s+(?:rules|instructions|policy)",
)
_RULES = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _PATTERNS]


@dataclass(frozen=True)
class InjectionResult:
    quarantined: bool
    reasons: tuple[str, ...] = ()
    backend: str = "sliding-window-scanner"


def classify_chunk(text: str, window_words: int = 256, overlap: int = 64) -> InjectionResult:
    """Inspect the entire chunk, including obfuscated instructions near its end.

    This is a conservative scanner, not a trained neural classifier or a claim
    that arbitrary prompt injection is solved. Quarantine never grants authority.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Cf")
    words = normalized.split()
    reasons = set()
    for start in range(0, len(words), max(1, window_words - overlap)):
        window = " ".join(words[start:start + window_words])
        for rule in _RULES:
            if rule.search(window):
                reasons.add(rule.pattern)
    return InjectionResult(bool(reasons), tuple(sorted(reasons)))


def safe_chunks(chunks):
    accepted, quarantined = [], []
    for chunk in chunks:
        if classify_chunk(chunk.text).quarantined:
            quarantined.append(chunk)
        else:
            accepted.append(chunk)
    if quarantined:
        logger.warning("Quarantined %d passages containing embedded instructions", len(quarantined))
    return accepted, quarantined
