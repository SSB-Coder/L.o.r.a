"""Paraphrase and n-gram verbatim copy verification."""

import re

from pydantic import BaseModel, ConfigDict


class CopyCheckResult(BaseModel):
    """Result of an n-gram overlap check against source material."""

    model_config = ConfigDict(frozen=True)

    is_clean: bool
    overlap_ratio: float
    matched_ngrams: list[str]


class CopyChecker:
    """Verifies that generated answers paraphrase sources in their own words."""

    def __init__(self, n_gram: int = 6, max_overlap_ratio: float = 0.25) -> None:
        self.n_gram = n_gram
        self.max_overlap_ratio = max_overlap_ratio

    def _tokenize(self, text: str) -> list[str]:
        # Lowercase and strip punctuation
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return [w for w in cleaned.split() if w]

    def _get_ngrams(self, tokens: list[str]) -> set[tuple[str, ...]]:
        if len(tokens) < self.n_gram:
            return set()
        return {
            tuple(tokens[i : i + self.n_gram])
            for i in range(len(tokens) - self.n_gram + 1)
        }

    def _remove_quoted_spans(self, text: str) -> str:
        # Strip explicitly double-quoted text so legitimate quoted citations are not penalized
        return re.sub(r'"[^"]*"', " ", text)

    def check_overlap(
        self,
        generated_text: str,
        source_texts: list[str],
    ) -> CopyCheckResult:
        """Check whether unquoted generated text overlaps too heavily with source texts."""
        if not generated_text or not source_texts:
            return CopyCheckResult(is_clean=True, overlap_ratio=0.0, matched_ngrams=[])

        unquoted_gen = self._remove_quoted_spans(generated_text)
        gen_tokens = self._tokenize(unquoted_gen)
        if len(gen_tokens) < self.n_gram:
            return CopyCheckResult(is_clean=True, overlap_ratio=0.0, matched_ngrams=[])

        gen_ngrams = self._get_ngrams(gen_tokens)
        if not gen_ngrams:
            return CopyCheckResult(is_clean=True, overlap_ratio=0.0, matched_ngrams=[])

        source_ngrams: set[tuple[str, ...]] = set()
        for src in source_texts:
            src_tokens = self._tokenize(src)
            source_ngrams.update(self._get_ngrams(src_tokens))

        matched = gen_ngrams.intersection(source_ngrams)
        overlap_ratio = len(matched) / len(gen_ngrams)
        is_clean = overlap_ratio <= self.max_overlap_ratio

        matched_str_list = [" ".join(gram) for gram in list(matched)[:5]]

        return CopyCheckResult(
            is_clean=is_clean,
            overlap_ratio=float(overlap_ratio),
            matched_ngrams=matched_str_list,
        )
