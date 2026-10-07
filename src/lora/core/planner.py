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

        # 3. Open path (Downloads, Documents, Desktop)
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
