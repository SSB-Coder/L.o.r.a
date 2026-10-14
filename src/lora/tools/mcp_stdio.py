"""Real MCP (Model Context Protocol) connection over stdio, so LORA can use external tool servers
declared in ~/.lora/mcp.json (filesystem, GitHub, databases, ...).

Every tool is registered into McpClientManager, which validates the arguments and runs each call through the
ActionGate. Tool output is untrusted data. Tools that could delete or run commands are never registered
(LORA's zero-deletion rule), and tools that look like they change things are Tier 2.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
from typing import Any

from lora.safety.policy import ActionRiskTier
from lora.tools.mcp import McpClientManager, McpServerConfig, McpToolDefinition

logger = logging.getLogger(__name__)
PROTOCOL = "2024-11-05"
FORBIDDEN_TOOL = re.compile(r"delete|remove|\brm\b|unlink|drop|truncate|kill|format|exec|shell|run_command|terminal|eval|sudo", re.IGNORECASE)
MUTATING_TOOL = re.compile(r"write|create|edit|update|move|rename|copy|upload|send|post|push|commit|insert|set_|put|append|save", re.IGNORECASE)


class McpError(Exception):
    pass


class McpStdioServer:
    """One MCP server process speaking newline-delimited JSON-RPC on stdin/stdout."""

    def __init__(self, name: str, cfg: McpServerConfig, timeout: float = 30.0) -> None:
        self.name, self.cfg, self.timeout = name, cfg, timeout
        self._proc: subprocess.Popen | None = None
        self._inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._id = 0
        self.tools: list[dict[str, Any]] = []

    def start(self) -> None:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        self._proc = subprocess.Popen(
            [self.cfg.command, *self.cfg.args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1, env={**os.environ, **self.cfg.env}, creationflags=flags,
        )
        threading.Thread(target=self._read, daemon=True).start()
        self._request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "lora", "version": "0.1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.tools = list(self._request("tools/list", {}).get("tools", []))

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            try:
                self._inbox.put(json.loads(line))
            except Exception:
                continue
        self._inbox.put({"closed": True})

    def _send(self, obj: dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._id += 1
            rid = self._id
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            while True:
                try:
                    msg = self._inbox.get(timeout=self.timeout)
                except queue.Empty as exc:
                    raise McpError(f"{self.name}: no answer to {method}") from exc
                if msg.get("closed"):
                    raise McpError(f"{self.name}: server exited")
                if msg.get("id") != rid:
                    continue  # notifications / other traffic
                if "error" in msg:
                    raise McpError(f"{self.name}: {msg['error'].get('message', msg['error'])}")
                return msg.get("result", {})

    def call(self, tool: str, arguments: dict[str, Any]) -> str:
        result = self._request("tools/call", {"name": tool, "arguments": arguments})
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        if result.get("isError"):
            raise McpError(text or "tool reported an error")
        return text

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()


def _clean_description(text: str) -> str:
    """Tool descriptions come from the server: strip control characters and keep them short."""
    return re.sub(r"[\x00-\x1f]+", " ", text or "")[:140].strip()


class McpHub:
    """Connects every enabled server in mcp.json and registers its tools with the manager."""

    def __init__(self, manager: McpClientManager) -> None:
        self.manager = manager
        self.servers: dict[str, McpStdioServer] = {}
        self.errors: dict[str, str] = {}
        self.skipped: list[str] = []
        self._started = False

    def connect_all(self) -> None:
        if self._started:
            return
        self._started = True
        for name, cfg in self.manager.load_registry().mcpServers.items():
            if cfg.disabled:
                continue
            server = McpStdioServer(name, cfg)
            try:
                server.start()
            except Exception as exc:
                self.errors[name] = f"{type(exc).__name__}: {exc}"
                logger.warning("MCP server %s could not start: %s", name, exc)
                server.close()
                continue
            self.servers[name] = server
            for tool in server.tools:
                raw = str(tool.get("name", ""))
                if not raw or FORBIDDEN_TOOL.search(raw):
                    self.skipped.append(f"{name}.{raw}")
                    continue
                tier = ActionRiskTier.TIER2 if MUTATING_TOOL.search(raw) else ActionRiskTier.TIER1
                exposed = f"{name}__{raw}"
                self.manager.register_mock_tool(
                    McpToolDefinition(
                        server_name=name, name=exposed, description=_clean_description(str(tool.get("description", ""))),
                        parameters=tool.get("inputSchema") or {}, tier=tier,
                    ),
                    lambda args, _s=server, _t=raw: _s.call(_t, args),
                )
            logger.info("MCP server %s connected with %d tools (%d blocked).", name, len(server.tools), len(self.skipped))

    @property
    def tool_names(self) -> list[str]:
        return list(self.manager.tools)

    def close(self) -> None:
        for s in self.servers.values():
            s.close()
