"""Unit tests for V3 NLI groundedness and V6 Number/Entity verification."""

from lora.verify.nli import NliGroundednessChecker
from lora.verify.numbers import NumberEntityChecker


def test_v3_nli_groundedness_entailed():
    checker = NliGroundednessChecker()
    premise = (
        "Superconducting quantum circuits operate at cryogenic temperatures "
        "below 20 millikelvin inside dilution refrigerators."
    )
    hypothesis = "Superconducting circuits require temperatures below 20 millikelvin."
    res = checker.check_entailment(premise=premise, hypothesis=hypothesis)
    assert res.score >= 0.80
    assert res.is_entailed is True


def test_v3_nli_groundedness_unsupported():
    checker = NliGroundednessChecker()
    premise = "Sourdough bread is baked using flour, water, and wild yeast culture."
    hypothesis = "Quantum processors achieve fault tolerance using topological braiding."
    res = checker.check_entailment(premise=premise, hypothesis=hypothesis)
    assert res.score < 0.80
    assert res.is_entailed is False


def test_v3_nli_negation_contradiction():
    checker = NliGroundednessChecker()
    premise = "The drug was proven safe and effective in Phase III clinical trials."
    hypothesis = "The drug was not proven safe in Phase III clinical trials."
    res = checker.check_entailment(premise=premise, hypothesis=hypothesis)
    assert res.score <= 0.20
    assert res.is_entailed is False


def test_v6_number_entity_checker_matching():
    checker = NumberEntityChecker()
    premise = "The study evaluated 1,500 participants and measured a 42.5% increase in retention in 2024."
    claim_valid = "Participants showed a 42.5% increase in retention [S1]."
    res = checker.verify_numbers(premise=premise, claim=claim_valid)
    assert res.is_valid is True
    assert res.mismatched_numbers == []


def test_v6_number_entity_checker_mismatched():
    checker = NumberEntityChecker()
    premise = "The study evaluated 1,500 participants with a 42.5% increase in 2024."
    # Fabricated number: 99.9% not present in premise
    claim_invalid = "The system achieved a 99.9% increase across participants [S1]."
    res = checker.verify_numbers(premise=premise, claim=claim_invalid)
    assert res.is_valid is False
    assert any("99.9" in m for m in res.mismatched_numbers)
