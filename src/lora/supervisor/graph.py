"""Execution graph and state machine for L.O.R.A Supervisor workflow."""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from lora.supervisor.checkpoint import SqliteCheckpointer
from lora.supervisor.state import GraphNode, SupervisorState

NodeHandler = Callable[[SupervisorState], SupervisorState | Any]


class StateGraph:
    """Explicit state machine coordinating nodes, step budgets, and checkpoints."""

    def __init__(self) -> None:
        self._handlers: dict[GraphNode, NodeHandler] = {}

    def register_node(self, node: GraphNode, handler: NodeHandler) -> None:
        """Register an execution handler for a specific graph node."""
        self._handlers[node] = handler

    def step(self, state: SupervisorState) -> SupervisorState:
        """Synchronously advance the graph by executing the current node's handler."""
        if state.is_terminal:
            return state

        if state.step_count >= state.max_steps:
            return state.model_copy(
                update={
                    "is_terminal": True,
                    "terminal_reason": "max_steps_exceeded",
                }
            )

        handler = self._handlers.get(state.current_node)
        if not handler:
            return state.model_copy(
                update={
                    "is_terminal": True,
                    "terminal_reason": f"no_handler_registered_for_{state.current_node.value}",
                }
            )

        # Execute handler (sync)
        if inspect.iscoroutinefunction(handler):
            # Run in existing or new event loop
            try:
                asyncio.get_running_loop()
                # Create a task if inside running loop, or block via nest
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    next_state = executor.submit(asyncio.run, handler(state)).result()
            except RuntimeError:
                next_state = asyncio.run(handler(state))
        else:
            next_state = handler(state)

        return next_state

    async def astep(self, state: SupervisorState) -> SupervisorState:
        """Asynchronously advance the graph by executing the current node's handler."""
        if state.is_terminal:
            return state

        if state.step_count >= state.max_steps:
            return state.model_copy(
                update={
                    "is_terminal": True,
                    "terminal_reason": "max_steps_exceeded",
                }
            )

        handler = self._handlers.get(state.current_node)
        if not handler:
            return state.model_copy(
                update={
                    "is_terminal": True,
                    "terminal_reason": f"no_handler_registered_for_{state.current_node.value}",
                }
            )

        if inspect.iscoroutinefunction(handler):
            next_state = await handler(state)
        else:
            next_state = handler(state)

        return next_state

    def run_to_completion(
        self,
        initial_state: SupervisorState,
        checkpointer: SqliteCheckpointer | None = None,
    ) -> SupervisorState:
        """Synchronously step through the graph until a terminal state is reached."""
        current = initial_state
        if checkpointer:
            checkpointer.save_checkpoint(current)

        while not current.is_terminal:
            current = self.step(current)
            if checkpointer:
                checkpointer.save_checkpoint(current)

        return current

    async def arun_to_completion(
        self,
        initial_state: SupervisorState,
        checkpointer: SqliteCheckpointer | None = None,
    ) -> SupervisorState:
        """Asynchronously step through the graph until a terminal state is reached."""
        current = initial_state
        if checkpointer:
            checkpointer.save_checkpoint(current)

        while not current.is_terminal:
            current = await self.astep(current)
            if checkpointer:
                checkpointer.save_checkpoint(current)

        return current
