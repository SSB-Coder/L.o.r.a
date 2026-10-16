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
async def test_verification_pipeline_long_article_with_negations_and_multicitations():
    """Regression test: long web articles containing 'not' must not poison affirmative claims or multi-citations."""
    long_article_1 = (
        "Mamba is a selective state space model (SSM) that achieves Transformer-quality results on language "
        "and vision benchmarks, making it practical for 100K+ token windows without KV-cache tricks. "
        "Unlike earlier RNNs that could not parallelize training, Mamba uses a hardware-aware parallel scan. "
        "Pure SSMs underperform attention on tasks that require copying or pointing to a specific earlier token."
    )
    long_article_4 = (
        "Transformers dominate in tasks requiring in-context recall, induction heads, retrieval, and exact lookups, "
        "as well as multi-modal fusion via cross-attention. Recent hybrid architectures such as AI21 Jamba do not "
        "pick a single side but interleave Mamba blocks with sparse attention layers and Mixture-of-Experts."
    )
    cards = [
        EvidenceCard(
            card_id="S1",
            claim_fact="Mamba vs Transformer",
            exact_quote=long_article_1,
            source_url="https://arxiv.org/abs/2312.00752",
            title="Mamba Paper",
            tier=1,
        ),
        EvidenceCard(
            card_id="S4",
            claim_fact="Hybrid Architectures",
            exact_quote=long_article_4,
            source_url="https://dev.to/mamba-vs-transformers",
            title="Mamba vs Transformers Review",
            tier=2,
        ),
    ]
    answer_text = (
        "- **Where Mamba Wins:** Mamba achieves Transformer-quality results on language and vision benchmarks, "
        "making it practical for 100K+ token windows without KV-cache tricks [S1, S4].\n"
        "- **Where Transformers Win:** Transformers dominate in tasks requiring in-context recall, induction heads, "
        "retrieval, and exact lookups, as well as multi-modal fusion [S1, S4].\n"
        "- **Hybrid Architectures**\n"
        "- Hybrid designs such as AI21 Jamba interleave Mamba blocks with sparse attention layers and Mixture-of-Experts [S4]."
    )
    pipeline = VerificationPipeline()
    report = await pipeline.verify(answer_text=answer_text, cards=cards)
    assert report.composite_faithfulness_score >= 0.85
    assert report.verdict == VerificationVerdict.VERIFIED
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
    answer_text = (
        "Transformers process sequences effectively [S1]. "
        "They run at 900 gigahertz without any electricity [S1]."
    )

    report = await pipeline.verify(answer_text=answer_text, cards=cards)

    assert len(report.unverified_claims) >= 1
    assert any("electricity" in c.lower() or "900" in c for c in report.unverified_claims)
    assert "900 gigahertz" not in report.repaired_answer or "Could not verify" in report.repaired_answer


@pytest.mark.asyncio
async def test_verification_pipeline_abstains_on_empty_cards():
    pipeline = VerificationPipeline()
    report = await pipeline.verify(answer_text="Any assertion [S1].", cards=[])
    assert report.verdict == VerificationVerdict.ABSTAIN
    assert report.requires_abstention is True
