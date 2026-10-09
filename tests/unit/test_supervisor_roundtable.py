"""Unit tests for Sequential Round-Table personas, family invariants, and arbitration."""

import pytest

from lora.supervisor.roundtable import (
    ChairPersona,
    EditorPersona,
    FactCheckerPersona,
    PersonaEvaluation,
    RoundTableEvaluator,
    SafetyOfficerPersona,
    SkepticPersona,
    get_model_family,
)


def test_model_family_mapping():
    assert get_model_family("granite4.2:8b") == "granite"
    assert get_model_family("spark-x25-4b-q4") == "spark"
    assert get_model_family("ornith-1.5:9b") == "qwen"
    assert get_model_family("Qwythos-9B-v2") == "qwen"
    assert get_model_family("gemma4:e4b") == "gemma"


def test_strict_model_family_invariant_enforcement():
    evaluator = RoundTableEvaluator()

    # Author = spark, Judge = granite -> OK
    evaluator.validate_family_invariant(author_model="spark-x25-4b-q4", judge_model="granite4.2:8b")

    # Author = granite, Judge = granite -> VIOLATION
    with pytest.raises(ValueError, match="Strict Model Family Invariant violated"):
        evaluator.validate_family_invariant(author_model="granite4.2:8b", judge_model="granite4.2:8b")

    # Author = ornith-1.5:9b, Judge = Qwythos-9B-v2 (both Qwen family) -> VIOLATION
    with pytest.raises(ValueError, match="Strict Model Family Invariant violated"):
        evaluator.validate_family_invariant(author_model="ornith-1.5:9b", judge_model="Qwythos-9B-v2")


def test_personas_evaluate_clean_draft():
    skeptic = SkepticPersona()
    fact_checker = FactCheckerPersona()
    editor = EditorPersona()
    safety = SafetyOfficerPersona()

    draft = "Quantum computers use superposition and entanglement to perform calculations [1]."
    claims = [{"text": "Quantum computers use superposition and entanglement", "status": "supported"}]
    cards = [{"source_title": "Quantum", "domain": "wikipedia.org", "snippet": "superposition and entanglement"}]

    res_sk = skeptic.evaluate(draft=draft, claims=claims, depth_mode="standard")
    res_fc = fact_checker.evaluate(draft=draft, claims=claims, evidence_cards=cards)
    res_ed = editor.evaluate(draft=draft, depth_mode="standard")
    res_sf = safety.evaluate(draft=draft)

    assert res_sk.verdict == "approve"
    assert res_fc.verdict == "approve"
    assert res_ed.verdict == "approve"
    assert res_sf.verdict == "approve"


def test_disagreement_rule_hard_fail_beats_majority():
    """Any hard fail (unsupported claim, safety breach) beats majority vote."""
    chair = ChairPersona()

    evals = [
        PersonaEvaluation(persona="skeptic", verdict="approve", score=0.95),
        PersonaEvaluation(persona="editor", verdict="approve", score=0.90),
        PersonaEvaluation(persona="fact_checker", verdict="veto", score=0.20, notes=["Fabricated claim found"]),
        PersonaEvaluation(persona="safety_officer", verdict="approve", score=1.0),
    ]

    report = chair.synthesize(evaluations=evals)
    # Even though 3 of 4 approved with high scores, hard veto overrides!
    assert report.decision in ["revise", "abstain"]
    assert report.hard_veto is True
    assert "Fabricated claim found" in report.veto_reasons


def test_roundtable_evaluator_sequential_execution_deep_mode():
    evaluator = RoundTableEvaluator()

    draft = "Light travels at approximately 300,000 km/s in vacuum [1]."
    claims = [{"text": "Light travels at 300,000 km/s in vacuum", "status": "supported"}]
    cards = [{"source_title": "Speed of light", "domain": "wikipedia.org", "snippet": "300,000 km/s in vacuum"}]

    report = evaluator.run_roundtable(
        draft=draft,
        claims=claims,
        evidence_cards=cards,
        depth_mode="standard",
        run_mode="deep",
        author_model="spark-x25-4b-q4",
        judge_model="granite4.2:8b",
    )

    assert report.decision == "approve"
    assert report.hard_veto is False
    assert report.composite_score >= 0.85
    assert len(report.persona_scores) >= 4


def test_roundtable_evaluator_quick_mode_fast_path():
    evaluator = RoundTableEvaluator()

    draft = "Quick mode answer."
    report = evaluator.run_roundtable(
        draft=draft,
        claims=[],
        evidence_cards=[],
        depth_mode="standard",
        run_mode="quick",
        author_model="spark-x25-4b-q4",
        judge_model="granite4.2:8b",
    )

    # In quick mode, full round table is bypassed for latency conservation
    assert report.decision == "approve"
    assert report.is_fast_path is True
