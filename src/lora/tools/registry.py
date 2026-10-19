"""Typed Tool Registry for L.O.R.A integrating with ActionGate."""

import asyncio
import concurrent.futures
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from lora.agent.notes import create_note
from lora.grounding.media_vector_db import MediaVectorDB
from lora.safety.gate import ActionGate, DataProvenance, has_untrusted_taint
from lora.safety.policy import ActionRiskTier, PolicyEngine
from lora.storage.export import ReportExporter
from lora.tools.browser.ask_ai import AskOtherAiTool
from lora.tools.browser.driver import PlaywrightBrowserDriver
from lora.tools.browser.pdf_fetcher import PdfFetcher
from lora.tools.control.apps import AppLauncher
from lora.tools.control.windows import WindowManager
from lora.tools.files.known import known_folder_roots
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

    def __init__(
        self,
        action_gate: ActionGate | None = None,
        media_db: MediaVectorDB | None = None,
    ) -> None:
        if action_gate is None:
            roots = list(PolicyEngine.DEFAULT_ALLOWED_ROOTS)
            roots += [r for r in known_folder_roots() if r not in roots]
            action_gate = ActionGate(policy_engine=PolicyEngine(allowed_roots=roots))
        self.action_gate = action_gate
        self._tools: dict[str, ToolManifest] = {}
        self.app_launcher = AppLauncher()
        self.window_manager = WindowManager()
        self.web_search_dispatcher = WebSearchDispatcher()
        self.browser_driver = PlaywrightBrowserDriver()
        self.pdf_fetcher = PdfFetcher()
        self.ask_other_ai_tool = AskOtherAiTool(driver=self.browser_driver)
        self.media_db = media_db or MediaVectorDB()
        self.report_exporter = ReportExporter()
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

        # 12. Ask Other AI (Tier 2 external browser typing per Plan §4.3.1 & §5.1)
        self.register(
            name="ask_other_ai",
            description="Ask an external AI chatbot (ChatGPT, Claude, Gemini, Perplexity) a question via DOM automation with prompt engineering.",
            tier=ActionRiskTier.TIER2,
            handler=self.ask_other_ai_tool.ask,
            parameters_schema={
                "type": "object",
                "properties": {
                    "service": {"type": "string", "enum": ["chatgpt", "claude", "gemini", "perplexity"]},
                    "prompt": {"type": "string"},
                    "optimize_prompt": {"type": "boolean", "default": True},
                    "style": {"type": "string", "default": "general"},
                    "deep_optimize": {"type": "boolean", "default": False},
                },
                "required": ["service", "prompt"],
            },
        )

        # 12b. Create a note (.md / .txt only, new files only, never overwrites)
        self.register(
            name="create_note",
            description="Create a new .md or .txt file in the LORA documents folder. Never overwrites or deletes.",
            tier=ActionRiskTier.TIER2,
            handler=lambda title, content, ext="md": create_note(
                title, content, ext, policy=self.action_gate.policy_engine
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "ext": {"type": "string", "enum": ["md", "txt"], "default": "md"},
                },
                "required": ["title", "content"],
            },
        )

        # 13. Add File / Media to FAISS Vector DB
        self.register(
            name="add_media_file",
            description="Ingest a file or media asset (PDF, Markdown, text, image, audio) into the FAISS vector database using EmbeddingGemma-2.",
            tier=ActionRiskTier.TIER1,
            handler=lambda path, tags=None: self.media_db.add_file(path, tags=tags),
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Local file path to index"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional category tags"},
                },
                "required": ["path"],
            },
        )

        # 14. Remove File / Media from FAISS Vector DB
        self.register(
            name="remove_media_file",
            description="Remove a file or media asset and rebuild the FAISS vector database.",
            tier=ActionRiskTier.TIER1,
            handler=lambda path_or_id: self.media_db.remove_file(path_or_id),
            parameters_schema={
                "type": "object",
                "properties": {
                    "path_or_id": {"type": "string", "description": "File path, filename, or file_id to remove"},
                },
                "required": ["path_or_id"],
            },
        )

        # 15. List Indexed Files and Media
        self.register(
            name="list_media_files",
            description="List all indexed files, documents, and media assets in the FAISS vector database.",
            tier=ActionRiskTier.TIER0,
            handler=self.media_db.list_files,
            parameters_schema={"type": "object"},
        )

        # 16. Search Media Vector DB
        self.register(
            name="search_media_db",
            description="Semantic vector search across indexed user files and media using EmbeddingGemma-2 and FAISS.",
            tier=ActionRiskTier.TIER0,
            handler=lambda query, top_k=5: self.media_db.search(query=query, top_k=top_k),
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Semantic search query"},
                    "top_k": {"type": "integer", "default": 5, "description": "Number of passages to return"},
                },
                "required": ["query"],
            },
        )

        # 17. Report on Files and Media
        self.register(
            name="report_on_media",
            description="Synthesize an evidence-backed research report reading from indexed files and media in the FAISS vector database.",
            tier=ActionRiskTier.TIER0,
            handler=lambda query="", top_k=6: self.media_db.report_on_media(query=query, top_k=top_k),
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "default": "", "description": "Topic or questions to report on"},
                    "top_k": {"type": "integer", "default": 6},
                },
            },
        )

        # 18. Export Report (Markdown & HTML styled reports in ~/LORA/outputs)
        self.register(
            name="export_report",
            description="Export research notes, answers, or documents to formatted Markdown or dark-aqueous HTML in ~/LORA/outputs.",
            tier=ActionRiskTier.TIER2,
            handler=self._handle_export_report,
            parameters_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Title of the report"},
                    "content": {"type": "string", "description": "Markdown text content of the report"},
                    "format": {"type": "string", "enum": ["markdown", "html", "both"], "default": "markdown"},
                    "filename": {"type": "string", "description": "Optional custom filename without extension"},
                },
                "required": ["title", "content"],
            },
        )

    def _handle_export_report(
        self,
        title: str,
        content: str,
        format: str = "markdown",
        filename: str | None = None,
    ) -> dict[str, Any]:
        paths = []
        fmt = (format or "markdown").lower()
        if fmt in ("markdown", "md", "both"):
            p_md = self.report_exporter.export_markdown(title=title, content=content, filename=filename)
            paths.append(str(p_md))
        if fmt in ("html", "both"):
            p_html = self.report_exporter.export_html(title=title, markdown_content=content, filename=filename)
            paths.append(str(p_html))
        return {
            "status": "success",
            "message": f"Exported report '{title}' to {', '.join(paths)}",
            "paths": paths,
        }

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
                if has_untrusted_taint(provenance):
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
                if has_untrusted_taint(provenance):
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
