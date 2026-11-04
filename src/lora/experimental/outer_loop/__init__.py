"""L.O.R.A Outer Loop / Recursive Self-Improvement Package."""

from lora.experimental.outer_loop.genome import (
    EditValidator,
    GenomeRegistry,
    PromptGenome,
    TemplateCompiler,
)
from lora.experimental.outer_loop.loop import IterationRecord, OptimizationTrajectory, RSIEngine
from lora.experimental.outer_loop.offline import BenchmarkReport, BenchmarkRunner, BenchmarkTask
from lora.experimental.outer_loop.optimizer import (
    PromptMutationProposal,
    PromptOptimizerAgent,
    TraceAnalyzer,
    TraceDiagnostic,
)
from lora.experimental.outer_loop.scorer import OuterLoopScorer, ScoreBreakdown

__all__ = [
    "BenchmarkReport",
    "BenchmarkRunner",
    "BenchmarkTask",
    "EditValidator",
    "GenomeRegistry",
    "IterationRecord",
    "OptimizationTrajectory",
    "OuterLoopScorer",
    "PromptGenome",
    "PromptMutationProposal",
    "PromptOptimizerAgent",
    "RSIEngine",
    "ScoreBreakdown",
    "TemplateCompiler",
    "TraceAnalyzer",
    "TraceDiagnostic",
]
