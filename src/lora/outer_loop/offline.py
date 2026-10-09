"""Offline Benchmark Mode evaluating prompt genomes across held-out tasks."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lora.outer_loop.genome import PromptGenome, TemplateCompiler
from lora.outer_loop.scorer import OuterLoopScorer


class BenchmarkTask(BaseModel):
    """Immutable test case for offline prompt evaluation."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    query: str
    evidence_cards: list[dict[str, Any]]
    split: str = "held_out"  # train, dev, held_out


class BenchmarkReport(BaseModel):
    """Evaluation summary comparing genome performance on test corpora."""

    model_config = ConfigDict(frozen=True)

    candidate_hash: str
    train_score: float
    held_out_score: float
    promoted: bool
    task_scores: dict[str, float] = Field(default_factory=dict)


class BenchmarkRunner:
    """Evaluates candidate prompt genomes across fixed offline test datasets."""

    def __init__(
        self,
        compiler: TemplateCompiler | None = None,
        scorer: OuterLoopScorer | None = None,
    ) -> None:
        self.compiler = compiler or TemplateCompiler()
        self.scorer = scorer or OuterLoopScorer()

    def evaluate_genome(
        self,
        genome: PromptGenome,
        tasks: list[BenchmarkTask],
        synthesis_fn: Callable[[str], str],
        verification_fn: Callable[[str], dict[str, Any]],
        baseline_held_out_score: float = 0.80,
    ) -> BenchmarkReport:
        """Run benchmark evaluation and decide promotion eligibility."""
        train_scores = []
        held_out_scores = []
        task_scores = {}

        for task in tasks:
            compiled = self.compiler.compile(genome, task.query, task.evidence_cards)
            ans = synthesis_fn(compiled)
            v_report = verification_fn(ans)
            b_down = self.scorer.calculate_score(
                answer=ans,
                user_request=task.query,
                evidence_cards=task.evidence_cards,
                verification_report=v_report,
                citations_found=["[1]"] if "[1]" in ans else [],
            )

            task_scores[task.task_id] = b_down.composite_score
            if task.split == "held_out":
                held_out_scores.append(b_down.composite_score)
            else:
                train_scores.append(b_down.composite_score)

        avg_train = sum(train_scores) / len(train_scores) if train_scores else 1.0
        avg_held_out = sum(held_out_scores) / len(held_out_scores) if held_out_scores else 1.0

        # Promotion requires strict empirical improvement on held-out tasks
        promoted = avg_held_out > baseline_held_out_score

        return BenchmarkReport(
            candidate_hash=genome.content_hash,
            train_score=round(avg_train, 4),
            held_out_score=round(avg_held_out, 4),
            promoted=promoted,
            task_scores=task_scores,
        )
