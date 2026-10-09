"""Unit tests for Supervisor State Graph engine."""

import tempfile
from pathlib import Path

from lora.supervisor.checkpoint import SqliteCheckpointer
from lora.supervisor.graph import StateGraph
from lora.supervisor.state import (
    GraphNode,
    NodeStatus,
    StepRecord,
    SupervisorAction,
    SupervisorState,
)


def test_graph_step_execution():
    graph = StateGraph()

    # Register custom dummy router handler
    def mock_router(state: SupervisorState) -> SupervisorState:
        return state.model_copy(
            update={
                "current_node": GraphNode.PLAN_REVIEW,
                "step_count": state.step_count + 1,
                "planned_actions": [
                    SupervisorAction(tool="web_search", params={"query": "test query"}),
                ],
                "history": [
                    *state.history,
                    StepRecord(node=GraphNode.ROUTER, status=NodeStatus.SUCCESS, timestamp=1.0),
                ],
            }
        )

    graph.register_node(GraphNode.ROUTER, mock_router)

    init_state = SupervisorState(
        session_id="test-run-1",
        user_request="Research topic",
        current_node=GraphNode.ROUTER,
    )

    next_state = graph.step(init_state)
    assert next_state.current_node == GraphNode.PLAN_REVIEW
    assert next_state.step_count == 1
    assert len(next_state.planned_actions) == 1
    assert len(next_state.history) == 1


def test_graph_max_steps_guard():
    graph = StateGraph()

    # Infinite loop node that never terminates
    def looping_node(state: SupervisorState) -> SupervisorState:
        return state.model_copy(
            update={
                "current_node": GraphNode.ROUTER,
                "step_count": state.step_count + 1,
            }
        )

    graph.register_node(GraphNode.ROUTER, looping_node)

    state = SupervisorState(
        session_id="loop-guard",
        user_request="Loop test",
        current_node=GraphNode.ROUTER,
        max_steps=5,
    )

    final_state = graph.run_to_completion(state)
    assert final_state.is_terminal is True
    assert final_state.terminal_reason == "max_steps_exceeded"
    assert final_state.step_count == 5


def test_graph_run_to_completion_with_checkpointer():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "checkpoints.db"
        checkpointer = SqliteCheckpointer(db_path)
        graph = StateGraph()

        # Step 1: ROUTER -> PLAN_REVIEW
        def handle_router(state: SupervisorState) -> SupervisorState:
            return state.model_copy(
                update={
                    "current_node": GraphNode.PLAN_REVIEW,
                    "step_count": state.step_count + 1,
                }
            )

        # Step 2: PLAN_REVIEW -> FINALIZE
        def handle_plan_review(state: SupervisorState) -> SupervisorState:
            return state.model_copy(
                update={
                    "current_node": GraphNode.FINALIZE,
                    "step_count": state.step_count + 1,
                }
            )

        # Step 3: FINALIZE -> terminal
        def handle_finalize(state: SupervisorState) -> SupervisorState:
            return state.model_copy(
                update={
                    "step_count": state.step_count + 1,
                    "is_terminal": True,
                    "terminal_reason": "completed",
                    "final_answer": "Complete answer",
                }
            )

        graph.register_node(GraphNode.ROUTER, handle_router)
        graph.register_node(GraphNode.PLAN_REVIEW, handle_plan_review)
        graph.register_node(GraphNode.FINALIZE, handle_finalize)

        init_state = SupervisorState(
            session_id="e2e-graph",
            user_request="E2E test",
            current_node=GraphNode.ROUTER,
        )

        completed = graph.run_to_completion(init_state, checkpointer=checkpointer)
        assert completed.is_terminal is True
        assert completed.terminal_reason == "completed"
        assert completed.final_answer == "Complete answer"

        # Verify checkpoints saved in SQLite
        history = checkpointer.list_checkpoints("e2e-graph")
        assert len(history) >= 3


def test_unregistered_node_halts():
    graph = StateGraph()
    state = SupervisorState(
        session_id="unregistered",
        user_request="test",
        current_node=GraphNode.FETCH,  # not registered
    )
    result = graph.step(state)
    assert result.is_terminal is True
    assert "no_handler_registered" in result.terminal_reason
