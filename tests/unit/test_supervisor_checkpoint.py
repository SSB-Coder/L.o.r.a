"""Unit tests for Supervisor SQLite checkpointer."""

import tempfile
from pathlib import Path

import pytest

from lora.supervisor.checkpoint import SqliteCheckpointer
from lora.supervisor.state import (
    GraphNode,
    NodeStatus,
    StepRecord,
    SupervisorAction,
    SupervisorState,
)


@pytest.fixture
def temp_db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "checkpoints.db"


def test_save_and_retrieve_latest_checkpoint(temp_db_path):
    checkpointer = SqliteCheckpointer(temp_db_path)
    state = SupervisorState(
        session_id="sess-001",
        user_request="Research quantum computing",
        depth_mode="standard",
        run_mode="deep",
        current_node=GraphNode.ROUTER,
        step_count=0,
    )

    step_idx = checkpointer.save_checkpoint(state)
    assert step_idx == 0

    retrieved = checkpointer.get_latest_checkpoint("sess-001")
    assert retrieved is not None
    assert retrieved.session_id == "sess-001"
    assert retrieved.user_request == "Research quantum computing"
    assert retrieved.current_node == GraphNode.ROUTER
    assert retrieved.depth_mode == "standard"


def test_multiple_checkpoints_and_history(temp_db_path):
    checkpointer = SqliteCheckpointer(temp_db_path)
    state0 = SupervisorState(
        session_id="sess-002",
        user_request="Check weather",
        current_node=GraphNode.ROUTER,
        step_count=0,
    )
    checkpointer.save_checkpoint(state0)

    # State update (immutable replacement)
    step1 = StepRecord(node=GraphNode.ROUTER, status=NodeStatus.SUCCESS, timestamp=100.0)
    state1 = state0.model_copy(
        update={
            "current_node": GraphNode.PLAN_REVIEW,
            "step_count": 1,
            "planned_actions": [
                SupervisorAction(tool="web_search", params={"query": "weather"}),
            ],
            "history": [step1],
        }
    )
    checkpointer.save_checkpoint(state1)

    latest = checkpointer.get_latest_checkpoint("sess-002")
    assert latest is not None
    assert latest.step_count == 1
    assert latest.current_node == GraphNode.PLAN_REVIEW
    assert len(latest.planned_actions) == 1
    assert latest.planned_actions[0].tool == "web_search"
    assert len(latest.history) == 1

    # Checkpoint at step 0
    at_0 = checkpointer.get_checkpoint_at_step("sess-002", 0)
    assert at_0 is not None
    assert at_0.step_count == 0
    assert at_0.current_node == GraphNode.ROUTER


def test_list_and_rollback_checkpoints(temp_db_path):
    checkpointer = SqliteCheckpointer(temp_db_path)
    base = SupervisorState(
        session_id="sess-rollback",
        user_request="Test rollback",
        current_node=GraphNode.ROUTER,
        step_count=0,
    )
    checkpointer.save_checkpoint(base)

    s1 = base.model_copy(update={"current_node": GraphNode.PLAN_REVIEW, "step_count": 1})
    checkpointer.save_checkpoint(s1)

    s2 = s1.model_copy(update={"current_node": GraphNode.FETCH, "step_count": 2})
    checkpointer.save_checkpoint(s2)

    cps = checkpointer.list_checkpoints("sess-rollback")
    assert len(cps) == 3
    assert [c["step_index"] for c in cps] == [0, 1, 2]

    # Roll back to step 1
    rolled_back = checkpointer.rollback_to_step("sess-rollback", 1)
    assert rolled_back.step_count == 1
    assert rolled_back.current_node == GraphNode.PLAN_REVIEW

    # After rollback, step 2 should be pruned
    cps_after = checkpointer.list_checkpoints("sess-rollback")
    assert len(cps_after) == 2
    assert [c["step_index"] for c in cps_after] == [0, 1]


def test_nonexistent_session(temp_db_path):
    checkpointer = SqliteCheckpointer(temp_db_path)
    assert checkpointer.get_latest_checkpoint("missing") is None
    assert checkpointer.get_checkpoint_at_step("missing", 0) is None
    assert checkpointer.list_checkpoints("missing") == []
