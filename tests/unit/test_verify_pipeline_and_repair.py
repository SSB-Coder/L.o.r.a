"""Unit tests for V8 Verification Pipeline, Composite Scoring, and Repair Loop."""

import pytest

from lora.grounding.cards import EvidenceCard
from lora.verify.pipeline import VerificationPipeline
from lora.verify.report import ConfidenceLevel, VerificationVerdict


@pytest.mark.asyncio
async def test_verification_pipeline_fully_supported():
    cards = [
        EvidenceCard(
            card_id="S1",
            claim_fact="Transformers replaced recurrent networks in 2017 with self-attention.",
            exact_quote="Transformers replaced recurrent networks in 2017 with self-attention.",
            source_url="https://arxiv.org/abs/1706.03762",
            title="Attention Is All You Need",
            page_or_section="p. 1",
            tier=1,
        )
    ]
    pipeline = VerificationPipeline()
    answer_text = "Transformers replaced recurrent networks in 2017 with self-attention [S1 p. 1]."

    report = await pipeline.verify(answer_text=answer_text, cards=cards)

    assert report.composite_faithfulness_score >= 0.85
    assert report.verdict == VerificationVerdict.VERIFIED
    assert report.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)
    assert report.unverified_claims == []


@pytest.mark.asyncio
async def test_verification_pipeline_repair_loop_unsupported_claim():
    cards = [
        EvidenceCard(
            card_id="S1",
            claim_fact="Transformers process sequences.",
            exact_quote="Transformers process sequences.",
            source_url="https://arxiv.org/abs/1706.03762",
            title="Attention Paper",
            tier=1,
        )
    ]
    pipeline = VerificationPipeline()
    # Contains one supported claim and one completely unsupported fabricated claim
    answer_text = (
        "Transformers process sequences effectively [S1]. "
        "They run at 900 gigahertz without any electricity [S1]."
    )

    report = await pipeline.verify(answer_text=answer_text, cards=cards)

    # The fabricated claim should be detected and flagged
    assert len(report.unverified_claims) >= 1
    assert any("electricity" in c.lower() or "900" in c for c in report.unverified_claims)
    # The repaired answer excludes the unverified claim or marks it
    assert "900 gigahertz" not in report.repaired_answer or "Could not verify" in report.repaired_answer


@pytest.mark.asyncio
async def test_verification_pipeline_abstains_on_empty_cards():
    pipeline = VerificationPipeline()
    report = await pipeline.verify(answer_text="Any assertion [S1].", cards=[])
    assert report.verdict == VerificationVerdict.ABSTAIN
    assert report.requires_abstention is True
