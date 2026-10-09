"""Unit tests for Multi-Dimensional Scoring Rubric and Deadlock-Free Gate."""

import pytest

from lora.outer_loop.scorer import OuterLoopScorer, ScoreBreakdown


@pytest.fixture
def scorer():
    return OuterLoopScorer()


def test_score_breakdown_composite_calculation(scorer):
    breakdown = scorer.calculate_score(
        answer="Quantum computers use qubits to process superposition states [1].",
        user_request="How do quantum computers work?",
        evidence_cards=[
            {"source_title": "Quantum Intro", "snippet": "Quantum computers use qubits and superposition."},
        ],
        verification_report={
            "supported_claims_ratio": 1.0,
            "composite_faithfulness_score": 0.95,
        },
        citations_found=["[1]"],
        depth_mode="standard",
    )

    assert isinstance(breakdown, ScoreBreakdown)
    assert 0.0 <= breakdown.composite_score <= 1.0
    assert breakdown.faithfulness == 0.95
    assert breakdown.citation_precision == 1.0
    assert breakdown.composite_score >= 0.80


def test_deadlock_free_faithfulness_gate_standard(scorer):
    # Case 1: Faithfulness >= 0.85 passes gate regardless of baseline
    assert scorer.check_faithfulness_gate(candidate_faithfulness=0.88, baseline_faithfulness=0.90) is True

    # Case 2: Candidate < 0.85, but strictly improves over poor baseline (e.g. 0.82 > 0.70) -> PASSES!
    assert scorer.check_faithfulness_gate(candidate_faithfulness=0.82, baseline_faithfulness=0.70) is True

    # Case 3: Candidate < 0.85 AND does not improve over baseline (e.g. 0.80 <= 0.82) -> FAILS!
    assert scorer.check_faithfulness_gate(candidate_faithfulness=0.80, baseline_faithfulness=0.82) is False


def test_low_coverage_and_empty_citations_penalized(scorer):
    breakdown = scorer.calculate_score(
        answer="Unrelated text about baking cakes.",
        user_request="Quantum physics",
        evidence_cards=[
            {"source_title": "Physics 1", "snippet": "Schrodinger equation and wave function."},
        ],
        verification_report={
            "supported_claims_ratio": 0.20,
            "composite_faithfulness_score": 0.20,
        },
        citations_found=[],  # zero citations
        depth_mode="standard",
    )

    assert breakdown.composite_score < 0.40
    assert breakdown.coverage < 0.30
    assert breakdown.citation_precision == 0.0
