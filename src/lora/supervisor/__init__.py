"""Compatibility exports for lora.experimental.supervisor.

Keep this package path intact so historical child imports resolve through their
forwarding modules and cannot construct duplicate model classes.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lora.experimental.supervisor import (
        ActionReviewer,
        AnomalyReport,
        AnomalyWatcher,
        AnswerReviewer,
        ChairPersona,
        EditorPersona,
        EvidenceReviewer,
        FactCheckerPersona,
        GraphNode,
        NodeStatus,
        PersonaEvaluation,
        PlanReviewer,
        ReviewResult,
        RoundTableEvaluator,
        RoundTableReport,
        SafetyOfficerPersona,
        SkepticPersona,
        SqliteCheckpointer,
        StateGraph,
        StepRecord,
        SupervisorAction,
        SupervisorOrchestrator,
        SupervisorState,
        get_model_family,
    )

__all__ = ['ActionReviewer', 'AnomalyReport', 'AnomalyWatcher', 'AnswerReviewer', 'ChairPersona', 'EditorPersona', 'EvidenceReviewer', 'FactCheckerPersona', 'GraphNode', 'NodeStatus', 'PersonaEvaluation', 'PlanReviewer', 'ReviewResult', 'RoundTableEvaluator', 'RoundTableReport', 'SafetyOfficerPersona', 'SkepticPersona', 'SqliteCheckpointer', 'StateGraph', 'StepRecord', 'SupervisorAction', 'SupervisorOrchestrator', 'SupervisorState', 'get_model_family']


def __getattr__(name):
    from importlib import import_module

    return getattr(import_module("lora.experimental.supervisor"), name)
