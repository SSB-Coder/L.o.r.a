"""Unit tests for Trace-Driven Prompt Optimizer and TraceAnalyzer."""

import pytest

from lora.outer_loop.genome import EditValidator, PromptGenome
from lora.outer_loop.optimizer import (
    PromptMutationProposal,
    PromptOptimizerAgent,
    TraceAnalyzer,
)
from lora.outer_loop.scorer import ScoreBreakdown


@pytest.fixture
def base_genome():
    return PromptGenome(
        system_prompt_base="You are L.O.R.A, a grounded research assistant.",
        refinement_rules=["Assert only claims directly grounded in evidence."],
        writer_style="concise_academic",
        version=0,
    )


def test_trace_analyzer_extracts_unverified_claims():
    analyzer = TraceAnalyzer()
    v_report = {
        "unverified_critical_claims": ["Mars has liquid oceans."],
    }
    rt_report = {
        "veto_reasons": ["Skeptic objection: claim 'Mars has liquid oceans' is unsupported."],
    }
    score_breakdown = ScoreBreakdown(
        composite_score=0.60,
        faithfulness=0.50,
        coverage=0.70,
        citation_precision=0.80,
        relevance=0.90,
        depth_fit=0.80,
        clarity=0.80,
        efficiency=1.0,
        feedback_text="Faithfulness below 0.85 gate.",
    )

    diag = analyzer.analyze(v_report, rt_report, score_breakdown)
    assert len(diag.target_flaws) >= 1
    assert any("Mars has liquid oceans" in f for f in diag.target_flaws)
    assert diag.priority_dimension == "faithfulness"


def test_optimizer_proposes_grounded_mutation(base_genome):
    optimizer = PromptOptimizerAgent(edit_validator=EditValidator())
    proposal = PromptMutationProposal(
        target_flaw="Unverified claims found in draft",
        action="append_rule",
        rule_to_add="Do not extrapolate beyond the provided evidence cards.",
        rationale="Eliminates speculative assertions",
    )

    candidate = optimizer.apply_mutation(base_genome=base_genome, proposal=proposal)
    assert candidate.version == 1
    assert len(candidate.refinement_rules) == 2
    assert "Do not extrapolate beyond the provided evidence cards." in candidate.refinement_rules


def test_optimizer_rejects_empty_target_flaw(base_genome):
    optimizer = PromptOptimizerAgent(edit_validator=EditValidator())
    invalid_proposal = PromptMutationProposal(
        target_flaw="",  # empty target flaw must be rejected!
        action="append_rule",
        rule_to_add="New rule",
        rationale="None",
    )

    with pytest.raises(ValueError, match="Target flaw cannot be empty"):
        optimizer.apply_mutation(base_genome=base_genome, proposal=invalid_proposal)


def test_optimizer_generates_mutation_from_diagnostic(base_genome):
    analyzer = TraceAnalyzer()
    optimizer = PromptOptimizerAgent()

    score_breakdown = ScoreBreakdown(
        composite_score=0.55,
        faithfulness=0.50,
        coverage=0.60,
        citation_precision=0.20,
        relevance=0.80,
        depth_fit=0.80,
        clarity=0.80,
        efficiency=1.0,
        feedback_text="Missing inline citation brackets [1].",
    )

    diag = analyzer.analyze(
        verification_report={"unverified_critical_claims": ["Unsupported assertion"]},
        roundtable_report={"veto_reasons": []},
        score_breakdown=score_breakdown,
    )

    proposal = optimizer.propose_mutation(diag)
    assert proposal.target_flaw != ""
    assert proposal.rule_to_add != ""

    candidate = optimizer.apply_mutation(base_genome, proposal)
    assert candidate.version == 1
