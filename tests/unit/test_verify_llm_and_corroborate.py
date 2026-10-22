"""Unit tests for V4 LLM Fact-Checker and V5 Cross-Source Corroborator."""

import json
from unittest.mock import AsyncMock

import pytest

from lora.grounding.cards import EvidenceCard
from lora.verify.corroborate import CorroborationChecker
from lora.verify.llm_check import LlmFactChecker


@pytest.mark.asyncio
async def test_v4_llm_fact_checker_supported():
    mock_ollama = AsyncMock()
    mock_ollama.generate = AsyncMock(
        return_value=json.dumps({
            "supported": True,
            "supporting_quote": "Transformers process tokens in parallel using self-attention.",
            "reason": "Direct textual support found.",
        })
    )

    checker = LlmFactChecker(ollama_client=mock_ollama)
    premise = "Transformers process tokens in parallel using self-attention."
    claim = "Tokens are computed in parallel by transformers."

    res = await checker.verify_claim(premise=premise, claim=claim)
    assert res.is_supported is True
    assert res.supporting_quote == "Transformers process tokens in parallel using self-attention."


@pytest.mark.asyncio
async def test_v4_llm_fact_checker_unsupported():
    mock_ollama = AsyncMock()
    mock_ollama.generate = AsyncMock(
        return_value=json.dumps({
            "supported": False,
            "supporting_quote": None,
            "reason": "Source does not mention this assertion.",
        })
    )

    checker = LlmFactChecker(ollama_client=mock_ollama)
    res = await checker.verify_claim(
        premise="Apples grow on trees.", claim="Bananas are rich in potassium."
    )
    assert res.is_supported is False
    assert res.supporting_quote is None


@pytest.mark.asyncio
@pytest.mark.parametrize("supported, quote", [
    (True, "Transformers process tokens in parallel."),
    ("false", "Transformers process tokens in parallel using self-attention."),
    ("true", "Transformers process tokens in parallel using self-attention."),
    (1, "Transformers process tokens in parallel using self-attention."),
    (True, None),
])
async def test_verifier_rejects_altered_quote_or_non_boolean_approval(supported, quote):
    client = AsyncMock()
    client.generate.return_value = json.dumps({"supported": supported, "supporting_quote": quote})
    result = await LlmFactChecker(client).verify_claim(
        premise="Transformers process tokens in parallel using self-attention.",
        claim="Tokens are computed in parallel by transformers.",
    )
    assert result.is_supported is False


def test_v5_corroboration_checker():
    checker = CorroborationChecker()

    card1 = EvidenceCard(
        card_id="S1",
        claim_fact="Revenue reached 10M.",
        exact_quote="Revenue reached 10M.",
        source_url="https://source-a.com/report",
        title="Report A",
        tier=2,
    )
    card2 = EvidenceCard(
        card_id="S2",
        claim_fact="Revenue reached 10M.",
        exact_quote="Revenue reached 10M.",
        source_url="https://source-b.org/filing",
        title="Filing B",
        tier=1,
    )

    # Two distinct domains -> corroborated
    res_multi = checker.check_corroboration(cards=[card1, card2])
    assert res_multi.is_corroborated is True
    assert res_multi.independent_source_count == 2
    assert res_multi.is_single_source is False

    # Single domain -> single-source flagged
    res_single = checker.check_corroboration(cards=[card1])
    assert res_single.is_corroborated is False
    assert res_single.is_single_source is True
