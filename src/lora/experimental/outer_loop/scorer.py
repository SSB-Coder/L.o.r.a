"""Multi-Dimensional Scoring Rubric and Deadlock-Free Gate for L.O.R.A Outer Loop."""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ScoreBreakdown(BaseModel):
    """Immutable scorecard evaluating a synthesized answer across 7 dimensions."""

    model_config = ConfigDict(frozen=True)

    composite_score: float
    faithfulness: float
    coverage: float
    citation_precision: float
    relevance: float
    depth_fit: float
    clarity: float
    efficiency: float
    feedback_text: str = ""
    component_scores: dict[str, float] = Field(default_factory=dict)


class OuterLoopScorer:
    """Reference-free objective scorer combining deterministic checks and rubric dimensions."""

    WEIGHT_FAITHFULNESS = 0.35
    WEIGHT_COVERAGE = 0.20
    WEIGHT_CITATIONS = 0.15
    WEIGHT_RELEVANCE = 0.10
    WEIGHT_DEPTH = 0.10
    WEIGHT_CLARITY = 0.05
    WEIGHT_EFFICIENCY = 0.05

    def check_faithfulness_gate(
        self,
        candidate_faithfulness: float,
        baseline_faithfulness: float,
    ) -> bool:
        """Deadlock-free gate: candidate must be >= 0.85 OR improve over baseline."""
        return candidate_faithfulness >= 0.85 or candidate_faithfulness > baseline_faithfulness

    def calculate_score(
        self,
        answer: str,
        user_request: str,
        evidence_cards: list[dict[str, Any]],
        verification_report: dict[str, Any],
        citations_found: list[str] | None = None,
        depth_mode: str = "standard",
    ) -> ScoreBreakdown:
        """Compute the weighted composite score and actionable feedback text."""
        citations = citations_found or []
        ans_clean = answer.strip().lower()
        ans_words = set(re.findall(r"\w+", ans_clean))

        # 1. Faithfulness (35%)
        faithfulness = float(
            verification_report.get(
                "composite_faithfulness_score",
                verification_report.get("supported_claims_ratio", 0.0),
            )
        )

        # 2. Coverage (20%): overlap of key terms from evidence cards in answer
        card_terms: set[str] = set()
        for card in evidence_cards:
            text = f"{card.get('source_title', '')} {card.get('snippet', '')}".lower()
            card_terms.update(w for w in re.findall(r"\w+", text) if len(w) > 4)

        if card_terms:
            covered = len(ans_words.intersection(card_terms))
            coverage = min(1.0, (covered / len(card_terms)) * 2.0)  # normalized multiplier
        else:
            coverage = 0.50

        # 3. Citation Precision (15%)
        if citations:
            citation_precision = 1.0
        elif evidence_cards:
            citation_precision = 0.0  # Penalized if evidence exists but no citations given
        else:
            citation_precision = 1.0

        # 4. Relevance (10%): overlap with user query
        req_words = {w for w in re.findall(r"\w+", user_request.lower()) if len(w) > 2}
        if req_words:
            rel_overlap = len(ans_words.intersection(req_words))
            relevance = min(1.0, rel_overlap / len(req_words))
        else:
            relevance = 1.0

        # 5. Depth Fit (10%)
        word_count = len(ans_words)
        if depth_mode in ["eli5", "simple"]:
            depth_fit = 1.0 if 30 <= word_count <= 250 else 0.70
        elif depth_mode in ["detailed", "expert"]:
            depth_fit = 1.0 if word_count >= 80 else 0.65
        else:  # standard
            depth_fit = 1.0 if 40 <= word_count <= 500 else 0.80

        # 6. Clarity (5%)
        clarity = 1.0 if len(answer.split(".")) >= 1 and word_count >= 10 else 0.50

        # 7. Efficiency (5%)
        efficiency = 1.0 if word_count <= 800 else max(0.20, 1.0 - (word_count - 800) / 1000.0)

        # Weighted composite score
        composite = (
            self.WEIGHT_FAITHFULNESS * faithfulness
            + self.WEIGHT_COVERAGE * coverage
            + self.WEIGHT_CITATIONS * citation_precision
            + self.WEIGHT_RELEVANCE * relevance
            + self.WEIGHT_DEPTH * depth_fit
            + self.WEIGHT_CLARITY * clarity
            + self.WEIGHT_EFFICIENCY * efficiency
        )
        composite = max(0.0, min(1.0, composite))

        # Feedback generation
        feedback_lines = []
        if faithfulness < 0.85:
            feedback_lines.append(f"Faithfulness ({faithfulness:.2f}) below 0.85 gate.")
        if coverage < 0.60:
            feedback_lines.append(f"Low source coverage ({coverage:.2f}): include more key facts from cards.")
        if citation_precision < 0.50:
            feedback_lines.append("Missing inline citation brackets [1].")
        if depth_fit < 0.80:
            feedback_lines.append(f"Adjust length and complexity for depth mode '{depth_mode}'.")

        feedback_text = "; ".join(feedback_lines) if feedback_lines else "Strong adherence across all rubric dimensions."

        scores_dict = {
            "faithfulness": faithfulness,
            "coverage": coverage,
            "citation_precision": citation_precision,
            "relevance": relevance,
            "depth_fit": depth_fit,
            "clarity": clarity,
            "efficiency": efficiency,
        }

        return ScoreBreakdown(
            composite_score=round(composite, 4),
            faithfulness=round(faithfulness, 4),
            coverage=round(coverage, 4),
            citation_precision=round(citation_precision, 4),
            relevance=round(relevance, 4),
            depth_fit=round(depth_fit, 4),
            clarity=round(clarity, 4),
            efficiency=round(efficiency, 4),
            feedback_text=feedback_text,
            component_scores=scores_dict,
        )
