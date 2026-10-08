"""Execution controller coordinating Planner and ToolRegistry for L.O.R.A."""

from typing import Any

from lora.core.planner import ControlPlanner
from lora.models.laya_router import LayaDecision
from lora.safety.gate import DataProvenance
from lora.tools.registry import ToolRegistry


class Controller:
    """Orchestrates action planning, gating, and execution for control intents."""

    def __init__(
        self,
        planner: ControlPlanner | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.planner = planner or ControlPlanner()
        self.tool_registry = tool_registry or ToolRegistry()

    def execute_user_request(
        self,
        user_request: str,
        decision: LayaDecision | None = None,
        origin: str = "desktop",
    ) -> dict[str, Any]:
        """Plan and execute a user control request."""
        # 1. Generate plan
        planned_actions = self.planner.plan(user_request, decision)

        results = []
        overall_success = True

        # 2. Execute each action sequentially
        for action in planned_actions:
            # Stamp provenance from user request
            provenance = {k: DataProvenance.USER for k in action.params}

            step_res = self.tool_registry.execute(
                tool_name=action.tool,
                params=action.params,
                origin=origin,
                provenance=provenance,
            )

            results.append({
                "tool": action.tool,
                "params": action.params,
                "reasoning": action.reasoning,
                "result": step_res,
            })

            if step_res.get("status") not in ["success", "partial"]:
                overall_success = False
                break

        return {
            "status": "success" if overall_success else "failed",
            "request": user_request,
            "steps": results,
            "step_count": len(results),
        }

    async def aexecute_user_request(
        self,
        user_request: str,
        decision: LayaDecision | None = None,
        origin: str = "desktop",
    ) -> dict[str, Any]:
        """Asynchronously plan and execute a user control request."""
        planned_actions = self.planner.plan(user_request, decision)
        results = []
        overall_success = True

        for action in planned_actions:
            provenance = {k: DataProvenance.USER for k in action.params}
            step_res = await self.tool_registry.aexecute(
                tool_name=action.tool,
                params=action.params,
                origin=origin,
                provenance=provenance,
            )
            results.append({
                "tool": action.tool,
                "params": action.params,
                "reasoning": action.reasoning,
                "result": step_res,
            })
            if step_res.get("status") not in ["success", "partial"]:
                overall_success = False
                break

        return {
            "status": "success" if overall_success else "failed",
            "request": user_request,
            "steps": results,
            "step_count": len(results),
        }
