import json
from pathlib import Path

from lora.safety.gate import ActionGate, DataProvenance
from lora.safety.policy import ActionRiskTier, PolicyEngine
from lora.tools.mcp import (
    McpClientManager,
    McpRegistryConfig,
    McpToolDefinition,
)


def test_mcp_config_parsing(tmp_path: Path):
    mcp_file = tmp_path / "mcp.json"
    mcp_data = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\safe"],
                "disabled": False,
            },
            "disabled_server": {
                "command": "python",
                "args": ["-m", "unused"],
                "disabled": True,
            },
        }
    }
    mcp_file.write_text(json.dumps(mcp_data), encoding="utf-8")

    manager = McpClientManager(mcp_config_path=mcp_file)
    registry = manager.load_registry()

    assert isinstance(registry, McpRegistryConfig)
    assert "filesystem" in registry.mcpServers
    assert registry.mcpServers["filesystem"].disabled is False
    assert registry.mcpServers["disabled_server"].disabled is True
    assert registry.mcpServers["filesystem"].command == "npx"


def test_mcp_tool_registration_and_sandboxing(tmp_path: Path):
    engine = PolicyEngine()
    gate = ActionGate(policy_engine=engine)
    manager = McpClientManager(gate=gate)

    # Register an in-memory MCP tool definition
    tool_def = McpToolDefinition(
        server_name="notes_server",
        name="read_note",
        description="Reads a note file",
        tier=ActionRiskTier.TIER1,
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )

    # Register tool handler
    def sample_handler(args: dict) -> str:
        return f"Content of {args['path']}"

    manager.register_mock_tool(tool_def, sample_handler)

    # Executing safe desktop user action should succeed
    res = manager.execute_tool(
        "read_note",
        {"path": str(Path.home() / "Documents" / "note.txt")},
        origin="desktop",
    )
    assert res.success is True
    assert "Content of" in res.content

    # Executing protected System32 path should be blocked
    bad_res = manager.execute_tool(
        "read_note",
        {"path": "C:\\Windows\\System32\\cmd.exe"},
        origin="desktop",
    )
    assert bad_res.success is False
    assert (
        "security" in bad_res.error.lower()
        or "blocked" in bad_res.error.lower()
        or "path" in bad_res.error.lower()
    )

    # Executing protected path using non-standard parameter name (e.g. dest_filepath)
    tool_def_custom = McpToolDefinition(
        server_name="backup_server",
        name="backup_file",
        description="Backs up file",
        tier=ActionRiskTier.TIER1,
        parameters={"type": "object", "properties": {"dest_filepath": {"type": "string"}}},
    )
    manager.register_mock_tool(tool_def_custom, lambda args: "ok")
    bad_custom_res = manager.execute_tool(
        "backup_file",
        {"dest_filepath": "C:\\Windows\\System32\\backup.txt"},
        origin="desktop",
    )
    assert bad_custom_res.success is False
    assert "blocked" in bad_custom_res.error.lower() or "path" in bad_custom_res.error.lower()

    # Executing with UNTRUSTED taint should be denied by ActionGate
    taint_res = manager.execute_tool(
        "read_note",
        {"path": str(Path.home() / "Documents" / "note.txt")},
        origin="desktop",
        provenance={"source": DataProvenance.UNTRUSTED},
    )
    assert taint_res.success is False
    assert "untrusted" in taint_res.error.lower() or "blocked" in taint_res.error.lower()


def test_mcp_schema_validation(tmp_path: Path):
    manager = McpClientManager()
    tool_def = McpToolDefinition(
        server_name="calc_server",
        name="add_numbers",
        description="Adds two integers",
        tier=ActionRiskTier.TIER0,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    )
    manager.register_mock_tool(tool_def, lambda args: str(args["a"] + args["b"]))

    # Valid schema call
    res = manager.execute_tool("add_numbers", {"a": 5, "b": 10}, origin="desktop")
    assert res.success is True
    assert res.content == "15"

    # Missing required argument
    bad_res = manager.execute_tool("add_numbers", {"a": 5}, origin="desktop")
    assert bad_res.success is False
    assert "validation" in bad_res.error.lower() or "missing" in bad_res.error.lower()
