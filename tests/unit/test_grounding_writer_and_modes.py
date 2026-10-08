"""Unit tests for DepthMode, DepthPromptBuilder, and ResearchWriter."""

from unittest.mock import AsyncMock

import pytest

from lora.grounding.cards import EvidenceCard
from lora.grounding.modes import DepthMode, DepthPromptBuilder
from lora.grounding.writer import ResearchWriter


def test_depth_prompt_builder():
    builder = DepthPromptBuilder()
    cards = [
        EvidenceCard(
            card_id="S1",
            claim_fact="Qubits lose state due to environmental decoherence.",
            exact_quote="Qubits lose state due to environmental decoherence.",
            source_url="https://arxiv.org/abs/2401.0001",
            title="Qubit Decoherence",
            page_or_section="p. 2",
            paragraph_index=1,
            tier=1,
        )
    ]
    prompt_eli5 = builder.build_prompt("Explain qubits", cards, DepthMode.ELI5)
    assert "everyday analogies" in prompt_eli5.lower() or "analogy" in prompt_eli5.lower()
    assert "[S1" in prompt_eli5

    prompt_detailed = builder.build_prompt("Explain qubits", cards, DepthMode.DETAILED)
    assert "mechanisms" in prompt_detailed.lower() or "detailed" in prompt_detailed.lower()


@pytest.mark.asyncio
async def test_research_writer_synthesis():
    mock_ollama = AsyncMock()
    mock_ollama.generate = AsyncMock(
        return_value="Quantum computers encounter decoherence when interacting with surroundings [S1 p. 2]."
    )

    writer = ResearchWriter(ollama_client=mock_ollama)
    cards = [
        EvidenceCard(
            card_id="S1",
            claim_fact="Qubits suffer from noise.",
            exact_quote="Qubits suffer from ambient noise.",
            source_url="https://example.com/paper",
            title="Paper A",
            page_or_section="p. 2",
            paragraph_index=0,
            tier=1,
        )
    ]

    answer = await writer.synthesize(
        query="Why are qubits noisy?",
        cards=cards,
        depth=DepthMode.STANDARD,
    )

    assert "decoherence" in answer.text.lower()
    assert "[S1 p. 2]" in answer.text
    assert len(answer.citations) >= 1
    assert answer.citations[0].source_url == "https://example.com/paper"
