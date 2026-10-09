import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lora.safety.gate import ActionGate, DataProvenance
from lora.safety.policy import ActionRiskTier, PolicyEngine

logger = logging.getLogger(__name__)


class McpServerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: str
    args: tuple[str, ...] = Field(default_factory=tuple)
    env: dict[str, str] = Field(default_factory=dict)
    disabled: bool = False


class McpRegistryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mcpServers: dict[str, McpServerConfig] = Field(default_factory=dict)  # noqa: N815


class McpToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    server_name: str
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    tier: ActionRiskTier = ActionRiskTier.TIER1


class McpToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: str
    success: bool
    content: str
    error: str | None = None


class McpClientManager:
    """Manages external MCP tool servers, schema validation, and deterministic ActionGate sandboxing."""

    def __init__(
        self,
        mcp_config_path: Path | None = None,
        gate: ActionGate | None = None,
    ) -> None:
        self._mcp_config_path = mcp_config_path
        self._gate = gate or ActionGate(policy_engine=PolicyEngine())
        self._tools: dict[str, McpToolDefinition] = {}
        self._handlers: dict[str, Callable[[dict[str, Any]], str]] = {}

    @property
    def tools(self) -> dict[str, McpToolDefinition]:
        return dict(self._tools)

    def load_registry(self) -> McpRegistryConfig:
        """Reads mcp.json and returns parsed McpRegistryConfig."""
        if not self._mcp_config_path or not self._mcp_config_path.is_file():
            return McpRegistryConfig()

        try:
            with open(self._mcp_config_path, encoding="utf-8") as f:
                data = json.load(f)
            return McpRegistryConfig.model_validate(data)
        except Exception as e:
            logger.warning(
                "Failed to parse MCP config %s: %s", self._mcp_config_path, e
            )
            return McpRegistryConfig()

    def register_mock_tool(
        self,
        tool_def: McpToolDefinition,
        handler: Callable[[dict[str, Any]], str],
    ) -> None:
        """Registers an in-process tool definition and execution handler for testing or builtin MCPs."""
        self._tools[tool_def.name] = tool_def
        self._handlers[tool_def.name] = handler

    def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        origin: str = "desktop",
        provenance: dict[str, Any] | None = None,
    ) -> McpToolResult:
        """Executes an MCP tool with schema validation and ActionGate policy evaluation."""
        if name not in self._tools:
            return McpToolResult(
                tool_name=name,
                success=False,
                content="",
                error=f"MCP tool '{name}' not found.",
            )

        tool_def = self._tools[name]

        # Normalize origin for policy evaluation
        norm_origin = "desktop"
        if str(origin).lower() in ("mobile", "user_remote", DataProvenance.USER_REMOTE):
            norm_origin = "mobile"

        # 1. Schema Validation (required parameters check)
        param_schema = tool_def.parameters
        if param_schema and isinstance(param_schema, dict):
            required_keys = param_schema.get("required", [])
            for key in required_keys:
                if key not in arguments:
                    return McpToolResult(
                        tool_name=name,
                        success=False,
                        content="",
                        error=f"Schema validation error: Missing required argument '{key}'.",
                    )

        # 2. Safety & Sandboxing evaluation via ActionGate
        policy_result = self._gate.verify_and_gate(
            tool=f"mcp_{name}",
            origin=norm_origin,
            tier=tool_def.tier,
            params=arguments,
            provenance=provenance,
        )

        if not policy_result.allowed:
            logger.warning(
                "ActionGate blocked execution of MCP tool %s: %s (rule: %s)",
                name,
                policy_result.reason,
                policy_result.rule_matched,
            )
            return McpToolResult(
                tool_name=name,
                success=False,
                content="",
                error=f"Security gate blocked action: {policy_result.reason}",
            )

        # 3. Path Safety check for any filesystem path argument
        path_candidate_keys = ("path", "target", "file", "dir", "folder", "dest", "src")
        for key, val in arguments.items():
            if not isinstance(val, str):
                continue
            key_lower = key.lower()
            is_path_key = any(p in key_lower for p in path_candidate_keys)
            is_path_val = ("\\" in val or "/" in val or (len(val) >= 2 and val[1] == ":"))
            if is_path_key or is_path_val:
                path_res = self._gate.policy_engine.check_path(
                    path=val,
                    origin=norm_origin,
                )
                if not path_res.allowed:
                    logger.warning(
                        "ActionGate path check blocked MCP tool %s param '%s'='%s': %s",
                        name,
                        key,
                        val,
                        path_res.reason,
                    )
                    return McpToolResult(
                        tool_name=name,
                        success=False,
                        content="",
                        error=f"Security gate blocked path ({key}): {path_res.reason}",
                    )

        # 4. Invocation
        handler = self._handlers.get(name)
        if not handler:
            return McpToolResult(
                tool_name=name,
                success=False,
                content="",
                error=f"No execution handler registered for tool '{name}'.",
            )

        try:
            result_str = handler(arguments)
            return McpToolResult(
                tool_name=name,
                success=True,
                content=result_str,
                error=None,
            )
        except Exception as e:
            logger.error("Error executing MCP tool %s: %s", name, e)
            return McpToolResult(
                tool_name=name,
                success=False,
                content="",
                error=f"Tool execution error: {e}",
            )
