"""Truthfulness evaluation benchmark suite for LORA Verification Pipeline."""

import pytest

from lora.grounding.cards import EvidenceCard
from lora.verify.pipeline import VerificationPipeline
from lora.verify.report import VerificationVerdict


@pytest.mark.asyncio
async def test_eval_answerable_grounded_question():
    cards = [
        EvidenceCard(
            card_id="S1",
            claim_fact="Gravitational waves were first directly detected by LIGO in September 2015.",
            exact_quote="Gravitational waves were first directly detected by LIGO in September 2015.",
            source_url="https://doi.org/10.1103/PhysRevLett.116.061102",
            title="Observation of Gravitational Waves",
            page_or_section="p. 1",
            tier=1,
        )
    ]
    pipeline = VerificationPipeline()
    answer = "LIGO directly detected gravitational waves in September 2015 [S1 p. 1]."
    report = await pipeline.verify(answer_text=answer, cards=cards)
    assert report.composite_faithfulness_score >= 0.85
    assert report.verdict == VerificationVerdict.VERIFIED


@pytest.mark.asyncio
async def test_eval_unanswerable_question_abstention():
    pipeline = VerificationPipeline()
    # No source cards available for future event or nonexistent topic
    report = await pipeline.verify(
        answer_text="The 2035 quantum teleportation satellite was launched successfully [S1].",
        cards=[],
    )
    assert report.verdict == VerificationVerdict.ABSTAIN
    assert report.requires_abstention is True


@pytest.mark.asyncio
async def test_eval_false_premise_detection():
    cards = [
        EvidenceCard(
            card_id="S1",
            claim_fact="Mars atmospheric pressure averages roughly 610 Pascals.",
            exact_quote="Mars atmospheric pressure averages roughly 610 Pascals.",
            source_url="https://example.com/mars",
            title="Mars Atmosphere",
            tier=1,
        )
    ]
    pipeline = VerificationPipeline()
    # Answer falsely asserts Mars atmosphere is thicker than Earth's
    answer = (
        "Mars atmospheric pressure is 610 Pascals [S1]. "
        "This makes Mars atmosphere significantly denser than Earth's atmosphere [S1]."
    )
    report = await pipeline.verify(answer_text=answer, cards=cards)
    # The unsupported claim must be caught
    assert len(report.unverified_claims) >= 1
    assert any("denser" in c.lower() for c in report.unverified_claims)


@pytest.mark.asyncio
async def test_eval_conflicting_sources_handling():
    cards = [
        EvidenceCard(
            card_id="S1",
            claim_fact="Study A measured speed of sound at 343 m/s at 20 degrees Celsius.",
            exact_quote="Study A measured speed of sound at 343 m/s at 20 degrees Celsius.",
            source_url="https://source-a.org",
            title="Acoustic Study A",
            tier=1,
        ),
        EvidenceCard(
            card_id="S2",
            claim_fact="Study B measured speed of sound at 331 m/s at 0 degrees Celsius.",
            exact_quote="Study B measured speed of sound at 331 m/s at 0 degrees Celsius.",
            source_url="https://source-b.org",
            title="Acoustic Study B",
            tier=1,
        ),
    ]
    pipeline = VerificationPipeline()
    answer = (
        "Study A measured 343 m/s at 20 degrees Celsius [S1], "
        "while Study B measured 331 m/s at 0 degrees Celsius [S2]."
    )
    report = await pipeline.verify(answer_text=answer, cards=cards)
    assert report.verdict == VerificationVerdict.VERIFIED
    assert len(report.claim_details) >= 2


@pytest.mark.asyncio
async def test_eval_gold_100_claims_benchmark():
    import json
    from pathlib import Path
    from lora.verify.nli import NliGroundednessChecker
    from lora.verify.numbers import NumberEntityChecker

    gold_path = Path(__file__).parent / "gold_claims_100.json"
    items = json.loads(gold_path.read_text(encoding="utf-8"))
    assert len(items) == 100

    nli = NliGroundednessChecker(threshold=0.80)
    num_checker = NumberEntityChecker()
    correct = 0
    for item in items:
        premise = item["premise"]
        claim = item["claim"]
        expected = bool(item["expected_supported"])
        num_res = num_checker.verify_numbers(premise=premise, claim=claim)
        nli_res = nli.check_entailment(premise=premise, hypothesis=claim)
        predicted = bool(num_res.is_valid and nli_res.is_entailed)
        if predicted == expected:
            correct += 1

    accuracy = correct / len(items)
    assert accuracy >= 0.92, f"Gold 100-claim accuracy {accuracy:.2%} below 92% Gate G3 target"
