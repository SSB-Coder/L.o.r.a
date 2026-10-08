"""Unit tests for V1 Citation existence check and V2 Claim decomposition."""

from lora.grounding.cards import EvidenceCard
from lora.verify.citation import CitationExistenceChecker
from lora.verify.claims import ClaimDecomposer


def test_v1_citation_checker_valid():
    cards = [
        EvidenceCard(
            card_id="S1",
            claim_fact="Transformers replaced RNNs.",
            exact_quote="Transformers replaced recurrent neural networks in 2017.",
            source_url="https://arxiv.org/abs/1706.03762",
            title="Attention Is All You Need",
            page_or_section="p. 1",
            tier=1,
        )
    ]
    checker = CitationExistenceChecker()
    valid_text = 'Transformers "replaced recurrent neural networks in 2017" across language processing [S1 p. 1].'
    res = checker.check_citations(text=valid_text, cards=cards)
    assert res.is_valid is True
    assert res.forged_citations == []
    assert res.invalid_quotes == []


def test_v1_citation_checker_forged_citation():
    cards = [
        EvidenceCard(
            card_id="S1",
            claim_fact="Fact",
            exact_quote="Quote",
            source_url="https://example.com",
            title="Title",
            tier=2,
        )
    ]
    checker = CitationExistenceChecker()
    forged_text = "This claim references a nonexistent source [S99 p. 4]."
    res = checker.check_citations(text=forged_text, cards=cards)
    assert res.is_valid is False
    assert "S99" in res.forged_citations


def test_v1_citation_checker_fabricated_quote():
    cards = [
        EvidenceCard(
            card_id="S1",
            claim_fact="Fact",
            exact_quote="Superconducting qubits require millikelvin dilution refrigerators.",
            source_url="https://example.com",
            title="Title",
            tier=2,
        )
    ]
    checker = CitationExistenceChecker()
    fake_quote_text = 'The paper stated that "qubits operate flawlessly at room temperature without cooling" [S1].'
    res = checker.check_citations(text=fake_quote_text, cards=cards)
    assert res.is_valid is False
    assert len(res.invalid_quotes) >= 1


def test_v2_claim_decomposer_basic():
    decomposer = ClaimDecomposer()
    text = (
        "Modern deep learning uses transformers [S1]. "
        "They process 500 tokens in parallel with multi-head attention [S2 p. 3]. "
        "This substantially decreases training latency."
    )
    claims = decomposer.decompose(text)
    assert len(claims) >= 2
    assert any("transformers" in c.claim_text.lower() for c in claims)
    assert any(c.has_numbers for c in claims)
    # Check citation association
    num_claim = next(c for c in claims if "500" in c.claim_text)
    assert "S2" in num_claim.cited_sources
