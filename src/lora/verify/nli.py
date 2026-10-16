"""V3: Non-LLM Natural Language Inference (NLI) groundedness checker on CPU."""

import logging
import os
import re
from typing import Any

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

DEFAULT_NLI_MODEL = os.getenv("LORA_NLI_MODEL", "vectara/hallucination_evaluation_model")


class NliScoreResult(BaseModel):
    """Result of natural language inference entailment scoring."""

    model_config = ConfigDict(frozen=True)

    score: float
    is_entailed: bool


class NliGroundednessChecker:
    """Evaluates whether premise text semantically entails a given claim on CPU (0 MB VRAM)."""

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        threshold: float = 0.80,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self._model: Any = None
        self._model_failed: bool = False

    def _load_model(self) -> Any:
        if os.getenv("LORA_VERIFY_OFFLINE", "0") == "1" or os.getenv("LORA_VERIFY_NLI") != "1":
            return None
        if self._model is None and not self._model_failed:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading NLI groundedness model '%s' on %s", self.model_name, self.device)
                self._model = CrossEncoder(self.model_name, device=self.device, trust_remote_code=True)
            except Exception as e:
                logger.debug("NLI model load deferred/fallback: %s", e)
                self._model_failed = True
        return self._model

    _STOPWORDS = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "has", "have", "had",
        "been", "will", "would", "could", "should", "can", "may", "might", "into", "onto", "upon", "about",
        "over", "under", "between", "among", "through", "during", "before", "after", "above", "below",
        "their", "there", "these", "those", "which", "while", "where", "when", "what", "who", "whom",
        "whose", "such", "than", "then", "them", "they", "its", "our", "your", "his", "her", "also",
        "more", "most", "some", "many", "much", "very", "just", "only", "both", "each", "other", "another",
    }

    _NEGATIONS = {"not", "never", "no", "neither", "nor", "cannot", "isnt", "arent", "wasnt", "werent"}

    def _score_window(self, window_text: str, h_words: list[str], h_has_neg: bool, check_neg: bool) -> float:
        w_clean = re.sub(r"[^\w\s]", " ", window_text.lower())
        w_tokens = w_clean.split()
        if not w_tokens:
            return 0.0
        w_set = set(w_tokens)
        w_stems = {w[:5] for w in w_set if len(w) >= 5}

        # Deduplicate hypothesis words so repeating one shared noun cannot mask missing predicate words
        unique_h = list(dict.fromkeys(h_words))
        matched = sum(
            1 for w in unique_h
            if w in w_set or (len(w) >= 5 and w[:5] in w_stems)
        )
        ratio = matched / len(unique_h)

        # Only penalize negation polarity mismatch when comparing against a short single-sentence premise
        # (prevents long multi-paragraph web articles containing "not" in an unrelated sentence from failing all claims)
        if check_neg and len(w_tokens) <= 28:
            w_has_neg = bool(w_set & self._NEGATIONS)
            if w_has_neg != h_has_neg:
                return 0.20

        if ratio >= 0.55:
            score = 0.80 + 0.20 * ((ratio - 0.55) / 0.45)
        else:
            score = (ratio / 0.55) * 0.70
        return min(1.0, max(0.0, float(score)))

    def _heuristic_entailment_score(self, premise: str, hypothesis: str) -> float:
        """Sentence-window & full-premise semantic containment score on CPU (handles long web articles & paraphrasing)."""
        h_clean = re.sub(r"[^\w\s]", " ", hypothesis.lower())
        h_tokens = h_clean.split()
        h_has_neg = bool(set(h_tokens) & self._NEGATIONS)
        h_words = [w for w in h_tokens if len(w) > 2 and w not in self._STOPWORDS]

        if not h_words:
            return 1.0 if premise.strip() else 0.5

        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", premise) if len(s.strip()) > 8]
        best = self._score_window(premise, h_words, h_has_neg, check_neg=(len(sents) <= 1))
        for i, s in enumerate(sents):
            sc = self._score_window(s, h_words, h_has_neg, check_neg=True)
            if sc > best:
                best = sc
            if i + 1 < len(sents):
                pair = f"{s} {sents[i + 1]}"
                sc2 = self._score_window(pair, h_words, h_has_neg, check_neg=False)
                if sc2 > best:
                    best = sc2
        return best

    def check_entailment(
        self,
        premise: str,
        hypothesis: str,
    ) -> NliScoreResult:
        """Score semantic entailment between premise (source) and hypothesis (claim)."""
        if not premise or not hypothesis:
            return NliScoreResult(score=0.0, is_entailed=False)

        model = None
        try:
            model = self._load_model()
        except Exception:
            model = None

        if model is not None:
            try:
                scores = model.predict([[premise[:1800], hypothesis]])
                score = float(scores[0])
                if score < 0.0 or score > 1.0:
                    import math

                    score = 1.0 / (1.0 + math.exp(-score))
                return NliScoreResult(
                    score=score,
                    is_entailed=score >= self.threshold,
                )
            except Exception as e:
                logger.warning("Error in NLI model predict: %s. Using heuristic.", e)

        heuristic_score = self._heuristic_entailment_score(premise, hypothesis)
        return NliScoreResult(
            score=heuristic_score,
            is_entailed=heuristic_score >= self.threshold,
        )
