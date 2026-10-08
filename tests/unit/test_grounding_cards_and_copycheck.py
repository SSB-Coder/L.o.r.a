"""Unit tests for EvidenceCard, EvidenceCardExtractor, and CopyChecker."""

import pytest

from lora.grounding.cards import EvidenceCard, EvidenceCardExtractor
from lora.grounding.chunk import DocumentChunk
from lora.grounding.copycheck import CopyChecker


def test_evidence_card_immutable():
    card = EvidenceCard(
        card_id="card-1",
        claim_fact="Transformers replaced recurrent networks for sequence tasks.",
        exact_quote="Transformers replaced recurrent networks for sequence tasks in 2017.",
        source_url="https://arxiv.org/abs/1706.03762",
        title="Attention Is All You Need",
        page_or_section="p. 1",
        paragraph_index=0,
        tier=1,
    )
    assert card.card_id == "card-1"
    assert card.tier == 1
    with pytest.raises(Exception):
        card.tier = 2


def test_evidence_card_extractor():
    extractor = EvidenceCardExtractor()
    chunk = DocumentChunk(
        chunk_id="c1",
        source_url="https://example.com/quantum",
        title="Quantum Computing",
        text="Superconducting qubits operate at millikelvin temperatures near absolute zero. "
        "Cryogenic dilution refrigerators maintain this low temperature environment. "
        "Decoherence remains the primary engineering obstacle for large-scale fault tolerance.",
        paragraph_index=0,
        page_number=3,
        token_count=35,
        tier=1,
        content_hash="qhash",
    )
    cards = extractor.extract_cards(chunks=[chunk], max_cards=5)
    assert len(cards) >= 1
    for card in cards:
        assert card.source_url == "https://example.com/quantum"
        assert card.title == "Quantum Computing"
        assert card.tier == 1
        assert len(card.exact_quote) > 0
        assert card.exact_quote in chunk.text


def test_copy_checker_verbatim_overlap():
    checker = CopyChecker(n_gram=6, max_overlap_ratio=0.30)
    source_text = (
        "Superconducting qubits operate at millikelvin temperatures near absolute zero "
        "inside cryogenic dilution refrigerators designed for quantum processors."
    )

    # Identical copied text
    assert checker.check_overlap(generated_text=source_text, source_texts=[source_text]).is_clean is False

    # Paraphrased text with no long n-gram match
    paraphrased = "Quantum processors function at extremely cold temperatures close to zero kelvin using specialized cooling systems."
    res = checker.check_overlap(generated_text=paraphrased, source_texts=[source_text])
    assert res.is_clean is True
    assert res.overlap_ratio < 0.20


def test_copy_checker_allows_quoted_spans():
    checker = CopyChecker(n_gram=6, max_overlap_ratio=0.25)
    source_text = "Superconducting qubits operate at millikelvin temperatures near absolute zero."
    text_with_quote = (
        'As researchers noted, "Superconducting qubits operate at millikelvin temperatures near absolute zero," '
        "which requires immense refrigeration."
    )
    res = checker.check_overlap(generated_text=text_with_quote, source_texts=[source_text])
    # Explicit quotes are allowed and discounted from verbatim penalty
    assert res.is_clean is True
