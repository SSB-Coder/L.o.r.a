"""Typed Tool Registry for L.O.R.A integrating with ActionGate."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from lora.safety.gate import ActionGate
from lora.safety.policy import ActionRiskTier
from lora.tools.control.apps import AppLauncher
from lora.tools.control.windows import WindowManager
from lora.tools.files.list import list_dir
from lora.tools.files.paths import open_path
from lora.tools.files.read import read_file
from lora.tools.files.search import search_files


class ToolManifest(BaseModel):
    name: str
    description: str
    tier: ActionRiskTier
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    handler: Any = None  # Callable


class ToolRegistry:
    """Central registry of executable capabilities bounded by ActionGate."""

    def __init__(self, action_gate: ActionGate | None = None) -> None:
        self.action_gate = action_gate or ActionGate()
        self._tools: dict[str, ToolManifest] = {}
        self.app_launcher = AppLauncher()
        self.window_manager = WindowManager()
        self._register_built_in_tools()

    def register(
        self,
        name: str,
        description: str,
        tier: ActionRiskTier,
        handler: Callable[..., Any],
        parameters_schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool with its metadata and execution handler."""
        manifest = ToolManifest(
            name=name,
            description=description,
            tier=tier,
            parameters_schema=parameters_schema or {},
            handler=handler,
        )
        self._tools[name] = manifest

    def _register_built_in_tools(self) -> None:
        """Register default Phase 3 control and filesystem tools."""
        # 1. Desktop Application
        self.register(
            name="open_app",
            description="Launch an allowlisted desktop application.",
            tier=ActionRiskTier.TIER1,
            handler=self.app_launcher.open_app,
            parameters_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        )

        # 2. Web URLs and Web Apps
        self.register(
            name="open_url",
            description="Open a validated web URL or web-app shortcut in the default browser.",
            tier=ActionRiskTier.TIER1,
            handler=self.app_launcher.open_url,
            parameters_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        )

        # 3. Safe Path / File Viewer
        self.register(
            name="open_path",
            description="Open an allowlisted file or folder in Explorer or default viewer.",
            tier=ActionRiskTier.TIER1,
            handler=lambda path: open_path(path, policy_engine=self.action_gate.policy_engine),
            parameters_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

        # 4. List Directory
        self.register(
            name="list_dir",
            description="List directory items inside allowed roots.",
            tier=ActionRiskTier.TIER0,
            handler=lambda path=".": list_dir(path, policy_engine=self.action_gate.policy_engine),
            parameters_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )

        # 5. Read File
        self.register(
            name="read_file",
            description="Safely read text, markdown, CSV, JSON, or extract PDF pages.",
            tier=ActionRiskTier.TIER0,
            handler=lambda path, max_bytes=10_000_000: read_file(path, max_bytes=max_bytes, policy_engine=self.action_gate.policy_engine),
            parameters_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

        # 6. Search Files
        self.register(
            name="search_files",
            description="Search filenames and text content inside allowed roots.",
            tier=ActionRiskTier.TIER0,
            handler=lambda query, root=".": search_files(query, root=root, policy_engine=self.action_gate.policy_engine),
            parameters_schema={"type": "object", "properties": {"query": {"type": "string"}, "root": {"type": "string"}}, "required": ["query"]},
        )

        # 7. Focus Window
        self.register(
            name="focus_window",
            description="Bring an existing application window matching title regex to the front.",
            tier=ActionRiskTier.TIER1,
            handler=self.window_manager.focus_window,
            parameters_schema={"type": "object", "properties": {"title_regex": {"type": "string"}}, "required": ["title_regex"]},
        )

        # 8. List Windows
        self.register(
            name="list_windows",
            description="List all active, visible desktop application windows.",
            tier=ActionRiskTier.TIER0,
            handler=self.window_manager.list_windows,
            parameters_schema={"type": "object"},
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """List all available tools and their schema metadata."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "tier": int(t.tier),
                "schema": t.parameters_schema,
            }
            for t in self._tools.values()
        ]

    def execute(
        self,
        tool_name: str,
        params: dict[str, Any] | None = None,
        origin: str = "desktop",
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify action against ActionGate policy and execute handler if permitted."""
        tool = self._tools.get(tool_name)
        if not tool:
            return {
                "status": "error",
                "message": f"Unknown tool '{tool_name}'.",
            }

        params_dict = params or {}

        # Gate verification
        gate_result = self.action_gate.verify_and_gate(
            tool=tool_name,
            origin=origin,
            tier=tool.tier,
            params=params_dict,
            provenance=provenance,
        )

        if not gate_result.allowed:
            return {
                "status": "denied",
                "reason": gate_result.reason,
                "verdict": str(gate_result.verdict),
            }

        try:
            return tool.handler(**params_dict)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Tool execution failed: {e}",
            }
