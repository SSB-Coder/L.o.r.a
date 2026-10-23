"""Immutable state models for L.O.R.A Supervisor & State Graph.

Provides typed, frozen Pydantic definitions for node progression,
checkpoints, action sequences, and anomaly auditing.
"""

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GraphNode(StrEnum):
    """Execution graph nodes for L.O.R.A Supervisor workflow."""

    ROUTER = "router"
    PLAN_REVIEW = "plan_review"
    ACTION_GATE = "action_gate"
    ACTION_EXECUTE = "action_execute"
    FETCH = "fetch"
    EVIDENCE_REVIEW = "evidence_review"
    WRITE = "write"
    ANSWER_REVIEW = "answer_review"
    ROUND_TABLE = "round_table"
    FINALIZE = "finalize"
    ABORT = "abort"


class NodeStatus(StrEnum):
    """Status outcomes from node checkpoint evaluation."""

    PENDING = "pending"
    SUCCESS = "success"
    REVISE = "revise"
    HALTED = "halted"
    DENIED = "denied"
    ERROR = "error"


class SupervisorAction(BaseModel):
    """Planned or executed tool action monitored by the supervisor."""

    model_config = ConfigDict(frozen=True)

    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    origin: str = "desktop"
    tier: int = 0
    reasoning: str = ""


class StepRecord(BaseModel):
    """Audit log entry capturing a node execution transition."""

    model_config = ConfigDict(frozen=True)

    node: GraphNode
    status: NodeStatus
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class AnomalyReport(BaseModel):
    """Detection record for loops, scope drift, or prompt injection attempts."""

    model_config = ConfigDict(frozen=True)

    detected: bool = False
    anomaly_type: str = ""
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class SupervisorState(BaseModel):
    """Immutable state snapshot for graph execution and SQLite checkpointing."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    user_request: str
    depth_mode: str = "standard"
    run_mode: str = "quick"
    current_node: GraphNode = GraphNode.ROUTER
    step_count: int = 0
    max_steps: int = 25
    loop_count: int = 0
    max_loops: int = 3
    author_model: str = "hf.co/XHToken/Spark-X2.6-4B-GGUF:Q4_K_M"
    author_family: str = "spark"
    planned_actions: list[SupervisorAction] = Field(default_factory=list)
    executed_actions: list[SupervisorAction] = Field(default_factory=list)
    evidence_cards: list[dict[str, Any]] = Field(default_factory=list)
    draft_answer: str = ""
    claims: list[dict[str, Any]] = Field(default_factory=list)
    verification_report: dict[str, Any] = Field(default_factory=dict)
    roundtable_report: dict[str, Any] = Field(default_factory=dict)
    final_answer: str = ""
    history: list[StepRecord] = Field(default_factory=list)
    anomalies: list[AnomalyReport] = Field(default_factory=list)
    is_terminal: bool = False
    terminal_reason: str = ""
