"""Planner for L.O.R.A Control Intents."""

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from lora.models.laya_router import LayaDecision
from lora.tools.control.apps import AppLauncher


class PlannedAction(BaseModel):
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


class ControlPlanner:
    """Plans desktop, application, and file control actions from user requests."""

    def __init__(self) -> None:
        self.app_launcher = AppLauncher()
        self.standard_paths = {
            "downloads": str(Path.home() / "Downloads"),
            "documents": str(Path.home() / "Documents"),
            "desktop": str(Path.home() / "Desktop"),
        }

    def _fast_path_plan(self, text: str) -> list[PlannedAction] | None:
        """Heuristic fast-path resolution for common single-step control requests."""
        clean = text.strip().lower()

        # 1. Window listing
        if any(w in clean for w in ["list windows", "what windows are open", "show open windows"]):
            return [PlannedAction(tool="list_windows", params={}, reasoning="User requested listing open windows")]

        # 2. Window focus
        focus_match = re.search(r"(?:focus|switch to|bring up)\s+(?:the\s+)?(.+?)(?:\s+window)?$", clean)
        if focus_match:
            target = focus_match.group(1).strip()
            return [PlannedAction(
                tool="focus_window",
                params={"title_regex": target},
                reasoning=f"User requested switching to window '{target}'",
            )]

        # 3. Ask External AI (ChatGPT, Claude, Gemini, Perplexity)
        ask_ai_match = re.search(r"^ask\s+(claude|chatgpt|gemini|perplexity)\s+(?:to\s+)?(.+)", text.strip(), re.IGNORECASE)
        if ask_ai_match:
            svc = ask_ai_match.group(1).strip().lower()
            prompt = ask_ai_match.group(2).strip()
            return [PlannedAction(
                tool="ask_other_ai",
                params={"service": svc, "prompt": prompt},
                reasoning=f"User requested asking {svc} external AI",
            )]

        # 4. Search specific knowledge provider (e.g. "search arxiv for quantum", "search github for lora")
        search_spec_match = re.search(r"^search\s+(arxiv|wikipedia|pubmed|github|crossref|hackernews|openlibrary|searxng|duckduckgo)\s+(?:for|about)?\s*(.+)", text.strip(), re.IGNORECASE)
        if search_spec_match:
            src = search_spec_match.group(1).strip().lower()
            query = search_spec_match.group(2).strip()
            return [PlannedAction(
                tool="web_search",
                params={"query": query, "source": src},
                reasoning=f"User requested searching {src} for '{query}'",
            )]

        # 5. General Web Search (e.g. "search web for X", "search for X")
        search_web_match = re.search(r"^search\s+(?:(?:the\s+)?(?:web|online|internet)\s+)?for\s+(.+)", text.strip(), re.IGNORECASE)
        if search_web_match:
            query = search_web_match.group(1).strip()
            return [PlannedAction(
                tool="web_search",
                params={"query": query, "source": "auto"},
                reasoning=f"User requested web search for '{query}'",
            )]

        # 6. Fetch Page / Webpage (case-preserving URL extraction)
        fetch_match = re.search(r"^(?:fetch|read|extract)\s+(?:webpage|page|article|url)\s+(https?://\S+)", text.strip(), re.IGNORECASE)
        if fetch_match:
            url = fetch_match.group(1).strip()
            return [PlannedAction(
                tool="fetch_page",
                params={"url": url},
                reasoning=f"User requested reading webpage {url}",
            )]

        # 7. Fetch PDF (case-preserving URL extraction)
        pdf_match = re.search(r"^(?:fetch|read|download)\s+pdf\s+(https?://\S+)", text.strip(), re.IGNORECASE)
        if pdf_match:
            url = pdf_match.group(1).strip()
            return [PlannedAction(
                tool="pdf_fetch",
                params={"url": url},
                reasoning=f"User requested fetching PDF {url}",
            )]

        # 8. Open path (Downloads, Documents, Desktop)
        for folder_name, folder_path in self.standard_paths.items():
            if f"open my {folder_name}" in clean or f"open {folder_name}" in clean:
                return [PlannedAction(
                    tool="open_path",
                    params={"path": folder_path},
                    reasoning=f"User requested opening system {folder_name} folder",
                )]

        # 4. Open web apps (Claude, ChatGPT, Wikipedia, etc.)
        for web_key in self.app_launcher.web_apps:
            if f"open {web_key}" in clean or f"launch {web_key}" in clean:
                # Check if there is a query (e.g. "open wikipedia on quantum computing")
                query = ""
                on_match = re.search(rf"open {web_key}\s+(?:on|about|for|in)\s+(.+)", clean)
                if on_match:
                    query = on_match.group(1).strip()

                target = self.app_launcher.resolve_web_app(web_key, query=query)
                if target:
                    return [PlannedAction(
                        tool="open_url",
                        params={"url": target["url"]},
                        reasoning=f"User requested opening web service {target['name']}",
                    )]

        # 5. Open desktop app (Notepad, VS Code, Calculator, etc.)
        for app_key in self.app_launcher.applications:
            aliases = [app_key] + self.app_launcher.applications[app_key].get("aliases", [])
            for alias in aliases:
                if f"open {alias}" in clean or f"launch {alias}" in clean or f"start {alias}" in clean:
                    return [PlannedAction(
                        tool="open_app",
                        params={"name": app_key},
                        reasoning=f"User requested launching {app_key}",
                    )]

        return None

    def plan(self, user_request: str, decision: LayaDecision | None = None) -> list[PlannedAction]:
        """Generate an execution plan of structured actions."""
        # Check fast path first for instantaneous sub-10ms response
        fast_plan = self._fast_path_plan(user_request)
        if fast_plan:
            return fast_plan

        # Fallback / compound action parsing
        clean = user_request.strip().lower()
        actions: list[PlannedAction] = []

        # Compound check: "open downloads and list files"
        if "downloads" in clean:
            actions.append(PlannedAction(
                tool="open_path",
                params={"path": self.standard_paths["downloads"]},
                reasoning="Opening Downloads folder",
            ))
            if "list" in clean:
                actions.append(PlannedAction(
                    tool="list_dir",
                    params={"path": self.standard_paths["downloads"]},
                    reasoning="Listing files in Downloads folder",
                ))

        if actions:
            return actions

        # Default fallback: attempt opening application
        first_word = user_request.split()[-1] if user_request else ""
        return [PlannedAction(
            tool="open_app",
            params={"name": first_word},
            reasoning=f"Attempting app resolution for '{first_word}'",
        )]
