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

    def _heuristic_entailment_score(self, premise: str, hypothesis: str) -> float:
        """Fast content-word and stem containment score on CPU (handles natural paraphrasing)."""
        p_clean = re.sub(r"[^\w\s]", "", premise.lower())
        h_clean = re.sub(r"[^\w\s]", "", hypothesis.lower())

        # Detect negation mismatch to avoid false positives on contradictory claims
        negations = {"not", "never", "no", "neither", "nor", "cannot", "isnt", "arent", "wasnt", "werent"}
        p_has_neg = bool(set(p_clean.split()) & negations)
        h_has_neg = bool(set(h_clean.split()) & negations)
        if p_has_neg != h_has_neg:
            return 0.20

        p_words = set(p_clean.split())
        p_stems = {w[:5] for w in p_words if len(w) >= 5}
        h_words = [w for w in h_clean.split() if len(w) > 2 and w not in self._STOPWORDS]

        if not h_words:
            return 1.0 if not p_words else 0.5

        matched = sum(
            1 for w in h_words
            if w in p_words or (len(w) >= 5 and w[:5] in p_stems)
        )
        ratio = matched / len(h_words)

        # Scale semantic alignment: content-word/stem overlap >= 55% maps to >= 0.80 entailment
        if ratio >= 0.55:
            score = 0.80 + 0.20 * ((ratio - 0.55) / 0.45)
        else:
            score = (ratio / 0.55) * 0.70
        return min(1.0, max(0.0, float(score)))

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
                scores = model.predict([[premise, hypothesis]])
                score = float(scores[0])
                # Sigmoid normalization if logits
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
