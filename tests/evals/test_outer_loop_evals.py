"""Evaluation suite verifying L.O.R.A Outer Loop Recursive Self-Improvement."""

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
from lora.outer_loop.offline import BenchmarkRunner, BenchmarkTask
from lora.outer_loop.optimizer import PromptOptimizerAgent
from lora.outer_loop.scorer import OuterLoopScorer


@pytest.fixture
def base_genome():
    return PromptGenome(
        system_prompt_base="You are L.O.R.A, a grounded research assistant.",
        refinement_rules=["Assert only claims directly grounded in evidence."],
        writer_style="concise_academic",
        version=0,
    )


def test_eval_demonstrated_net_improvement(base_genome):
    """Verify that RSIEngine produces a positive net score improvement over baseline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = GenomeRegistry(Path(tmpdir) / "eval.db")
        engine = RSIEngine(
            registry=registry,
            compiler=TemplateCompiler(),
            optimizer=PromptOptimizerAgent(edit_validator=EditValidator()),
            scorer=OuterLoopScorer(),
            max_iters=4,
        )

        evidence = [
            {"source_title": "Paper 1", "snippet": "Specific institutional data from MIT and Stanford."},
        ]

        def progressive_synthesizer(prompt: str) -> str:
            # As rules are added, answer incorporates citations and institutional attribution
            if "key definitions" in prompt:
                return "MIT and Stanford verified room temperature superconductivity does not occur [1]."
            return "Superconductivity was not confirmed [1]."

        def progressive_verifier(answer: str) -> dict:
            if "MIT and Stanford" in answer:
                return {
                    "composite_faithfulness_score": 0.98,
                    "supported_claims_ratio": 1.0,
                    "unverified_critical_claims": [],
                }
            return {
                "composite_faithfulness_score": 0.70,
                "supported_claims_ratio": 0.70,
                "unverified_critical_claims": [],
            }

        trajectory = engine.run(
            initial_genome=base_genome,
            user_request="Summarize research replications",
            evidence_cards=evidence,
            synthesis_fn=progressive_synthesizer,
            verification_fn=progressive_verifier,
        )

        assert trajectory.net_improvement >= 0.0
        assert trajectory.best_score >= trajectory.baseline_score


def test_eval_100_percent_rollback_on_seeded_regression(base_genome):
    """Verify that a seeded quality drop triggers an immediate rollback to the prior best."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = GenomeRegistry(Path(tmpdir) / "rollback_eval.db")
        engine = RSIEngine(
            registry=registry,
            compiler=TemplateCompiler(),
            optimizer=PromptOptimizerAgent(),
            scorer=OuterLoopScorer(),
            max_iters=3,
        )

        evidence = [{"source_title": "Source", "snippet": "Factual text."}]
        call_num = 0

        def regressing_synthesizer(prompt: str) -> str:
            nonlocal call_num
            call_num += 1
            if call_num <= 3:
                return "High quality grounded answer [1]."
            return "Severe hallucination regression with unverified assertions."

        def regressing_verifier(answer: str) -> dict:
            if "High quality" in answer:
                return {"composite_faithfulness_score": 0.92, "unverified_critical_claims": []}
            return {"composite_faithfulness_score": 0.15, "unverified_critical_claims": ["Severe hallucination"]}

        trajectory = engine.run(
            initial_genome=base_genome,
            user_request="Query",
            evidence_cards=evidence,
            synthesis_fn=regressing_synthesizer,
            verification_fn=regressing_verifier,
        )

        assert trajectory.best_genome.version == 0
        assert trajectory.best_genome.content_hash == base_genome.content_hash
        assert any(r.decision == "rolled_back" for r in trajectory.history)
        assert registry.get_latest_genome().version == 0


def test_eval_offline_benchmark_held_out_promotion(base_genome):
    """Verify that candidate prompt promotion strictly gates on held-out tasks."""
    runner = BenchmarkRunner()

    tasks = [
        BenchmarkTask(
            task_id="t1",
            query="How do quantum computers work?",
            evidence_cards=[{"source_title": "Quantum", "snippet": "Quantum computers use qubits and superposition."}],
            split="train",
        ),
        BenchmarkTask(
            task_id="h1",
            query="How do quantum computers work?",
            evidence_cards=[{"source_title": "Quantum", "snippet": "Quantum computers use qubits and superposition."}],
            split="held_out",
        ),
    ]

    def mock_synth(p: str) -> str:
        return (
            "Quantum computers work by utilizing qubits that leverage the principles of "
            "superposition and quantum entanglement to perform complex state calculations [1]. "
            "These systems enable exponential computational speedups for specialized problems."
        )

    def mock_ver(a: str) -> dict:
        return {"composite_faithfulness_score": 0.95, "unverified_critical_claims": []}

    # Case 1: High held-out performance -> Promoted
    report1 = runner.evaluate_genome(
        genome=base_genome,
        tasks=tasks,
        synthesis_fn=mock_synth,
        verification_fn=mock_ver,
        baseline_held_out_score=0.80,
    )
    assert report1.promoted is True
    assert report1.held_out_score >= 0.85

    # Case 2: Held-out performance below baseline -> Rejected
    report2 = runner.evaluate_genome(
        genome=base_genome,
        tasks=tasks,
        synthesis_fn=mock_synth,
        verification_fn=mock_ver,
        baseline_held_out_score=0.99,  # baseline threshold higher than achieved
    )
    assert report2.promoted is False
