"""Compatibility exports for lora.experimental.outer_loop.

Keep this package path intact so historical child imports resolve through their
forwarding modules and cannot construct duplicate model classes.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lora.experimental.outer_loop import (
        BenchmarkReport,
        BenchmarkRunner,
        BenchmarkTask,
        EditValidator,
        GenomeRegistry,
        IterationRecord,
        OptimizationTrajectory,
        OuterLoopScorer,
        PromptGenome,
        PromptMutationProposal,
        PromptOptimizerAgent,
        RSIEngine,
        ScoreBreakdown,
        TemplateCompiler,
        TraceAnalyzer,
        TraceDiagnostic,
    )

__all__ = ['BenchmarkReport', 'BenchmarkRunner', 'BenchmarkTask', 'EditValidator', 'GenomeRegistry', 'IterationRecord', 'OptimizationTrajectory', 'OuterLoopScorer', 'PromptGenome', 'PromptMutationProposal', 'PromptOptimizerAgent', 'RSIEngine', 'ScoreBreakdown', 'TemplateCompiler', 'TraceAnalyzer', 'TraceDiagnostic']


def __getattr__(name):
    from importlib import import_module

    return getattr(import_module("lora.experimental.outer_loop"), name)
