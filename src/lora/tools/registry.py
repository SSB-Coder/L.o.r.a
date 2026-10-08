"""Typed Tool Registry for L.O.R.A integrating with ActionGate."""

import asyncio
import concurrent.futures
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from lora.safety.gate import ActionGate, DataProvenance
from lora.safety.policy import ActionRiskTier, PolicyEngine
from lora.tools.browser.ask_ai import AskOtherAiTool
from lora.tools.browser.driver import PlaywrightBrowserDriver
from lora.tools.browser.pdf_fetcher import PdfFetcher
from lora.tools.control.apps import AppLauncher
from lora.tools.control.windows import WindowManager
from lora.tools.files.list import list_dir
from lora.tools.files.paths import open_path
from lora.tools.files.read import read_file
from lora.tools.files.search import search_files
from lora.tools.web.search import WebSearchDispatcher


class ToolManifest(BaseModel):
    name: str
    description: str
    tier: ActionRiskTier
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    handler: Any = None  # Callable


class ToolRegistry:
    """Central registry of executable capabilities bounded by ActionGate."""

    def __init__(self, action_gate: ActionGate | None = None) -> None:
        self.action_gate = action_gate or ActionGate(policy_engine=PolicyEngine())
        self._tools: dict[str, ToolManifest] = {}
        self.app_launcher = AppLauncher()
        self.window_manager = WindowManager()
        self.web_search_dispatcher = WebSearchDispatcher()
        self.browser_driver = PlaywrightBrowserDriver()
        self.pdf_fetcher = PdfFetcher()
        self.ask_other_ai_tool = AskOtherAiTool(driver=self.browser_driver)
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
        """Register default Phase 3 control and Phase 4 web/browser tools."""
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

        # 9. Web and Knowledge Search
        self.register(
            name="web_search",
            description="Search free scholarly knowledge and web sources (arXiv, Wikipedia, Crossref, PubMed, GitHub, etc.).",
            tier=ActionRiskTier.TIER0,
            handler=self.web_search_dispatcher.search,
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "source": {"type": "string", "default": "auto"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        )

        # 10. Fetch Web Page
        self.register(
            name="fetch_page",
            description="Fetch and extract clean article text from a web page using headless browser.",
            tier=ActionRiskTier.TIER0,
            handler=self.browser_driver.fetch_page,
            parameters_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        )

        # 11. PDF Fetch
        self.register(
            name="pdf_fetch",
            description="Safely download and extract text from an external PDF document.",
            tier=ActionRiskTier.TIER0,
            handler=self.pdf_fetcher.fetch_pdf,
            parameters_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        )

        # 12. Ask Other AI
        self.register(
            name="ask_other_ai",
            description="Ask an external AI chatbot (ChatGPT, Claude) a question via DOM automation.",
            tier=ActionRiskTier.TIER1,
            handler=self.ask_other_ai_tool.ask,
            parameters_schema={
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["service", "prompt"],
            },
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

    async def aexecute(
        self,
        tool_name: str,
        params: dict[str, Any] | None = None,
        origin: str = "desktop",
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Asynchronously verify action against ActionGate policy and execute handler."""
        tool = self._tools.get(tool_name)
        if not tool:
            return {
                "status": "error",
                "message": f"Unknown tool '{tool_name}'.",
            }

        params_dict = params or {}

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
            # Pass provenance to ask_other_ai if expected
            if tool_name == "ask_other_ai" and provenance:
                prov_val = provenance.get("provenance")
                if prov_val == "untrusted":
                    params_dict["provenance"] = DataProvenance.UNTRUSTED

            res = tool.handler(**params_dict)
            if asyncio.iscoroutine(res):
                return await res
            return res
        except Exception as e:
            return {
                "status": "error",
                "message": f"Tool execution failed: {e}",
            }

    def execute(
        self,
        tool_name: str,
        params: dict[str, Any] | None = None,
        origin: str = "desktop",
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synchronously verify action against ActionGate policy and execute handler."""
        tool = self._tools.get(tool_name)
        if not tool:
            return {
                "status": "error",
                "message": f"Unknown tool '{tool_name}'.",
            }

        params_dict = params or {}

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
            if tool_name == "ask_other_ai" and provenance:
                prov_val = provenance.get("provenance")
                if prov_val == "untrusted":
                    params_dict["provenance"] = DataProvenance.UNTRUSTED

            res = tool.handler(**params_dict)
            if asyncio.iscoroutine(res):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        return pool.submit(asyncio.run, res).result()
                else:
                    return asyncio.run(res)
            return res
        except Exception as e:
            return {
                "status": "error",
                "message": f"Tool execution failed: {e}",
            }
