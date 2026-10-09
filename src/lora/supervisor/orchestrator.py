"""Supervisor Orchestrator coordinating Graph, Checkpoints, and Round Table."""

import time

from lora.safety.gate import ActionGate
from lora.safety.killswitch import KillSwitch
from lora.safety.policy import PolicyEngine
from lora.supervisor.checkpoint import SqliteCheckpointer
from lora.supervisor.checkpoints import (
    ActionReviewer,
    AnomalyWatcher,
    AnswerReviewer,
    EvidenceReviewer,
    PlanReviewer,
)
from lora.supervisor.graph import StateGraph
from lora.supervisor.roundtable import RoundTableEvaluator
from lora.supervisor.state import (
    GraphNode,
    NodeStatus,
    StepRecord,
    SupervisorState,
)


class SupervisorOrchestrator:
    """Coordinates multi-agent execution, security gating, anomaly watching, and arbitration."""

    def __init__(
        self,
        checkpointer: SqliteCheckpointer | None = None,
        action_gate: ActionGate | None = None,
        judge_model: str = "granite4.2:8b",
    ) -> None:
        self.checkpointer = checkpointer or SqliteCheckpointer()
        self.policy_engine = PolicyEngine()
        self.kill_switch = KillSwitch()
        self.action_gate = action_gate or ActionGate(
            policy_engine=self.policy_engine,
            kill_switch=self.kill_switch,
        )
        self.judge_model = judge_model

        self.anomaly_watcher = AnomalyWatcher()
        self.plan_reviewer = PlanReviewer()
        self.action_reviewer = ActionReviewer(action_gate=self.action_gate)
        self.evidence_reviewer = EvidenceReviewer()
        self.answer_reviewer = AnswerReviewer()
        self.roundtable = RoundTableEvaluator()

        self.graph = StateGraph()
        self._register_nodes()

    def _register_nodes(self) -> None:
        self.graph.register_node(GraphNode.ROUTER, self._node_router)
        self.graph.register_node(GraphNode.PLAN_REVIEW, self._node_plan_review)
        self.graph.register_node(GraphNode.ACTION_GATE, self._node_action_gate)
        self.graph.register_node(GraphNode.ACTION_EXECUTE, self._node_action_execute)
        self.graph.register_node(GraphNode.FETCH, self._node_fetch)
        self.graph.register_node(GraphNode.EVIDENCE_REVIEW, self._node_evidence_review)
        self.graph.register_node(GraphNode.WRITE, self._node_write)
        self.graph.register_node(GraphNode.ANSWER_REVIEW, self._node_answer_review)
        self.graph.register_node(GraphNode.ROUND_TABLE, self._node_round_table)
        self.graph.register_node(GraphNode.FINALIZE, self._node_finalize)
        self.graph.register_node(GraphNode.ABORT, self._node_abort)

    def _node_router(self, state: SupervisorState) -> SupervisorState:
        req = state.user_request.lower()
        if any(term in req for term in ["open", "launch", "focus", "switch", "calc", "notepad"]):
            # Control flow
            return state.model_copy(
                update={
                    "current_node": GraphNode.PLAN_REVIEW,
                    "step_count": state.step_count + 1,
                    "history": [
                        *state.history,
                        StepRecord(node=GraphNode.ROUTER, status=NodeStatus.SUCCESS, timestamp=time.time()),
                    ],
                }
            )

        # Default to research / fetch flow
        return state.model_copy(
            update={
                "current_node": GraphNode.FETCH,
                "step_count": state.step_count + 1,
                "history": [
                    *state.history,
                    StepRecord(node=GraphNode.ROUTER, status=NodeStatus.SUCCESS, timestamp=time.time()),
                ],
            }
        )

    def _node_plan_review(self, state: SupervisorState) -> SupervisorState:
        res = self.plan_reviewer.review_plan(state.user_request, state.planned_actions)
        if not res.approved:
            return state.model_copy(
                update={
                    "current_node": GraphNode.ABORT,
                    "step_count": state.step_count + 1,
                    "is_terminal": True,
                    "terminal_reason": f"plan_review_{res.verdict}: {res.reason}",
                    "history": [
                        *state.history,
                        StepRecord(node=GraphNode.PLAN_REVIEW, status=NodeStatus.DENIED, details={"reason": res.reason}),
                    ],
                }
            )

        return state.model_copy(
            update={
                "current_node": GraphNode.ACTION_GATE,
                "step_count": state.step_count + 1,
                "history": [
                    *state.history,
                    StepRecord(node=GraphNode.PLAN_REVIEW, status=NodeStatus.SUCCESS),
                ],
            }
        )

    def _node_action_gate(self, state: SupervisorState) -> SupervisorState:
        for action in state.planned_actions:
            # Check repetition loop
            rep = self.anomaly_watcher.check_repetition(action, state.executed_actions)
            if rep.detected:
                return state.model_copy(
                    update={
                        "current_node": GraphNode.ABORT,
                        "step_count": state.step_count + 1,
                        "anomalies": [*state.anomalies, rep],
                        "is_terminal": True,
                        "terminal_reason": f"anomaly_{rep.anomaly_type}: {rep.message}",
                    }
                )

            # Check scope drift
            drift = self.anomaly_watcher.check_scope_drift(state.user_request, action)
            if drift.detected:
                return state.model_copy(
                    update={
                        "current_node": GraphNode.ABORT,
                        "step_count": state.step_count + 1,
                        "anomalies": [*state.anomalies, drift],
                        "is_terminal": True,
                        "terminal_reason": f"anomaly_{drift.anomaly_type}: {drift.message}",
                    }
                )

            # Evaluate with ActionReviewer
            res = self.action_reviewer.review_action(state.user_request, action)
            if not res.approved:
                return state.model_copy(
                    update={
                        "current_node": GraphNode.ABORT,
                        "step_count": state.step_count + 1,
                        "is_terminal": True,
                        "terminal_reason": f"action_gate_{res.verdict}: {res.reason}",
                    }
                )

        return state.model_copy(
            update={
                "current_node": GraphNode.ACTION_EXECUTE,
                "step_count": state.step_count + 1,
                "history": [
                    *state.history,
                    StepRecord(node=GraphNode.ACTION_GATE, status=NodeStatus.SUCCESS),
                ],
            }
        )

    def _node_action_execute(self, state: SupervisorState) -> SupervisorState:
        # Move planned actions into executed actions
        return state.model_copy(
            update={
                "current_node": GraphNode.FINALIZE,
                "step_count": state.step_count + 1,
                "executed_actions": [*state.executed_actions, *state.planned_actions],
                "planned_actions": [],
                "history": [
                    *state.history,
                    StepRecord(node=GraphNode.ACTION_EXECUTE, status=NodeStatus.SUCCESS),
                ],
            }
        )

    def _node_fetch(self, state: SupervisorState) -> SupervisorState:
        # Check cards for injection traps
        anomalies = []
        for card in state.evidence_cards:
            snippet = card.get("snippet", "")
            inj = self.anomaly_watcher.check_injection(snippet)
            if inj.detected:
                anomalies.append(inj)

        if anomalies:
            return state.model_copy(
                update={
                    "current_node": GraphNode.ABORT,
                    "step_count": state.step_count + 1,
                    "anomalies": [*state.anomalies, *anomalies],
                    "is_terminal": True,
                    "terminal_reason": "prompt_injection_detected_in_evidence",
                }
            )

        return state.model_copy(
            update={
                "current_node": GraphNode.EVIDENCE_REVIEW,
                "step_count": state.step_count + 1,
                "history": [
                    *state.history,
                    StepRecord(node=GraphNode.FETCH, status=NodeStatus.SUCCESS),
                ],
            }
        )

    def _node_evidence_review(self, state: SupervisorState) -> SupervisorState:
        res = self.evidence_reviewer.review_evidence(state.user_request, state.evidence_cards)
        if not res.approved:
            return state.model_copy(
                update={
                    "current_node": GraphNode.ABORT,
                    "step_count": state.step_count + 1,
                    "is_terminal": True,
                    "terminal_reason": f"evidence_review_{res.verdict}: {res.reason}",
                }
            )

        return state.model_copy(
            update={
                "current_node": GraphNode.WRITE,
                "step_count": state.step_count + 1,
                "history": [
                    *state.history,
                    StepRecord(node=GraphNode.EVIDENCE_REVIEW, status=NodeStatus.SUCCESS),
                ],
            }
        )

    def _node_write(self, state: SupervisorState) -> SupervisorState:
        draft = state.draft_answer or f"Synthesized research response for '{state.user_request}'."
        return state.model_copy(
            update={
                "draft_answer": draft,
                "current_node": GraphNode.ANSWER_REVIEW,
                "step_count": state.step_count + 1,
                "history": [
                    *state.history,
                    StepRecord(node=GraphNode.WRITE, status=NodeStatus.SUCCESS),
                ],
            }
        )

    def _node_answer_review(self, state: SupervisorState) -> SupervisorState:
        res = self.answer_reviewer.review_answer(
            answer=state.draft_answer,
            verification_report=state.verification_report,
            claims=state.claims,
        )

        if not res.approved:
            if state.loop_count >= state.max_loops:
                return state.model_copy(
                    update={
                        "current_node": GraphNode.ABORT,
                        "step_count": state.step_count + 1,
                        "is_terminal": True,
                        "terminal_reason": f"answer_review_max_loops_exceeded: {res.reason}",
                    }
                )
            return state.model_copy(
                update={
                    "current_node": GraphNode.ABORT,
                    "step_count": state.step_count + 1,
                    "loop_count": state.loop_count + 1,
                    "is_terminal": True,
                    "terminal_reason": f"answer_review_{res.verdict}: {res.reason}",
                }
            )

        return state.model_copy(
            update={
                "current_node": GraphNode.ROUND_TABLE,
                "step_count": state.step_count + 1,
                "history": [
                    *state.history,
                    StepRecord(node=GraphNode.ANSWER_REVIEW, status=NodeStatus.SUCCESS),
                ],
            }
        )

    def _node_round_table(self, state: SupervisorState) -> SupervisorState:
        rt_report = self.roundtable.run_roundtable(
            draft=state.draft_answer,
            claims=state.claims,
            evidence_cards=state.evidence_cards,
            depth_mode=state.depth_mode,
            run_mode=state.run_mode,
            author_model=state.author_model,
            judge_model=self.judge_model,
        )

        if rt_report.decision == "approve":
            return state.model_copy(
                update={
                    "current_node": GraphNode.FINALIZE,
                    "step_count": state.step_count + 1,
                    "roundtable_report": rt_report.model_dump(),
                    "history": [
                        *state.history,
                        StepRecord(node=GraphNode.ROUND_TABLE, status=NodeStatus.SUCCESS),
                    ],
                }
            )

        # Revise or Abstain
        return state.model_copy(
            update={
                "current_node": GraphNode.ABORT,
                "step_count": state.step_count + 1,
                "roundtable_report": rt_report.model_dump(),
                "is_terminal": True,
                "terminal_reason": f"roundtable_{rt_report.decision}: {', '.join(rt_report.veto_reasons)}",
            }
        )

    def _node_finalize(self, state: SupervisorState) -> SupervisorState:
        return state.model_copy(
            update={
                "is_terminal": True,
                "terminal_reason": "completed",
                "final_answer": state.draft_answer or "Execution completed successfully.",
                "history": [
                    *state.history,
                    StepRecord(node=GraphNode.FINALIZE, status=NodeStatus.SUCCESS),
                ],
            }
        )

    def _node_abort(self, state: SupervisorState) -> SupervisorState:
        return state.model_copy(
            update={
                "is_terminal": True,
                "history": [
                    *state.history,
                    StepRecord(node=GraphNode.ABORT, status=NodeStatus.HALTED),
                ],
            }
        )
