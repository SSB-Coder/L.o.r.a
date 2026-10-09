"""Unit tests for RSIEngine (Recursive Self-Improvement Loop) and Rollback."""

import tempfile
from pathlib import Path

import pytest

from lora.outer_loop.genome import (
    EditValidator,
    GenomeRegistry,
    PromptGenome,
    TemplateCompiler,
)
from lora.outer_loop.loop import RSIEngine
from lora.outer_loop.optimizer import PromptOptimizerAgent
from lora.outer_loop.scorer import OuterLoopScorer


@pytest.fixture
def temp_db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "genomes_loop.db"


@pytest.fixture
def base_genome():
    return PromptGenome(
        system_prompt_base="You are L.O.R.A, a grounded research assistant.",
        refinement_rules=["Assert only claims directly grounded in evidence."],
        writer_style="concise_academic",
        version=0,
    )


def test_rsi_engine_iterative_improvement_and_acceptance(temp_db_path, base_genome):
    registry = GenomeRegistry(temp_db_path)
    compiler = TemplateCompiler()
    optimizer = PromptOptimizerAgent(edit_validator=EditValidator())
    scorer = OuterLoopScorer()

    engine = RSIEngine(
        registry=registry,
        compiler=compiler,
        optimizer=optimizer,
        scorer=scorer,
        max_iters=3,
    )

    evidence = [
        {"source_title": "Superconductivity 101", "snippet": "Cu2S impurities cause resistance drop at 104C."},
    ]

    # Mock synthesis function that improves as rules are added
    def mock_synthesizer(prompt: str) -> str:
        if "Synthesize all key definitions" in prompt:
            return "Replications proved Cu2S impurities cause the 104C resistance drop in LK-99 [1]."
        return "LK-99 might be a room temperature superconductor [1]."

    def mock_verifier(answer: str) -> dict:
        if "Cu2S impurities cause" in answer:
            return {
                "composite_faithfulness_score": 0.95,
                "supported_claims_ratio": 1.0,
                "unverified_critical_claims": [],
            }
        return {
            "composite_faithfulness_score": 0.60,
            "supported_claims_ratio": 0.60,
            "unverified_critical_claims": ["LK-99 might be a room temperature superconductor"],
        }

    trajectory = engine.run(
        initial_genome=base_genome,
        user_request="Explain LK-99 findings",
        evidence_cards=evidence,
        synthesis_fn=mock_synthesizer,
        verification_fn=mock_verifier,
    )

    assert trajectory.iterations_run >= 1
    assert trajectory.best_score >= trajectory.baseline_score
    assert trajectory.best_genome.version >= 0
    assert "Cu2S impurities" in trajectory.best_answer


def test_rsi_engine_instant_rollback_on_regression(temp_db_path, base_genome):
    registry = GenomeRegistry(temp_db_path)
    compiler = TemplateCompiler()
    optimizer = PromptOptimizerAgent(edit_validator=EditValidator())
    scorer = OuterLoopScorer()

    engine = RSIEngine(
        registry=registry,
        compiler=compiler,
        optimizer=optimizer,
        scorer=scorer,
        max_iters=3,
    )

    evidence = [{"source_title": "Paper", "snippet": "True facts"}]

    call_count = 0

    # Iteration 0 baseline is good, mutation drops quality
    def mock_synthesizer(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= 3:  # baseline runs
            return "Solid baseline facts [1]."
        return "Hallucinated garbage with zero evidence."

    def mock_verifier(answer: str) -> dict:
        if "Solid baseline" in answer:
            return {"composite_faithfulness_score": 0.90, "unverified_critical_claims": []}
        return {"composite_faithfulness_score": 0.10, "unverified_critical_claims": ["Hallucinated"]}

    trajectory = engine.run(
        initial_genome=base_genome,
        user_request="Topic",
        evidence_cards=evidence,
        synthesis_fn=mock_synthesizer,
        verification_fn=mock_verifier,
    )

    # Must rollback to baseline G0 and halt immediately!
    assert trajectory.best_genome.content_hash == base_genome.content_hash
    assert trajectory.best_genome.version == 0
    assert any(rec.decision == "rolled_back" for rec in trajectory.history)
    assert registry.get_latest_genome().content_hash == base_genome.content_hash
