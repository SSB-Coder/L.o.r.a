"""Capability Gating by Origin for L.O.R.A Gateway."""

from lora.safety.policy import ActionRiskTier, PolicyEngine, PolicyResult

MOBILE_RESEARCH_TOOLS: set[str] = {
    "web_search",
    "fetch_page",
    "pdf_fetch",
    "knowledge_query",
    "notebook_query",
    "rewrite_depth",
    # Legacy aliases kept for backwards-compatible tests
    "search_web",
    "fetch_pdf",
    "read_web_page",
}

DESKTOP_CONTROL_TOOLS: set[str] = {
    "open_app",
    "open_url",
    "open_path",
    "list_dir",
    "read_file",
    "search_files",
    "focus_window",
    "list_windows",
    "ask_other_ai",
    "create_note",
    "create_artifact",
    "generate_visual",
    "browser_type",
    "mouse_click",
    "keyboard_press",
    "export_report",
}


class OriginCapabilityGate:
    """Enforces origin-based capability boundaries (Mobile = Tier 0 research only; Desktop = full policy)."""

    def __init__(self, policy_engine: PolicyEngine | None = None) -> None:
        self.policy = policy_engine or PolicyEngine()

    def evaluate(
        self,
        tool: str,
        origin: str,
        tier: int | ActionRiskTier = ActionRiskTier.TIER0,
    ) -> PolicyResult:
        """Evaluate whether the requested tool is permitted for the given origin."""
        return self.policy.check_action(tool=tool, origin=origin, tier=tier)


def get_allowed_tools_for_origin(origin: str) -> set[str]:
    """Return the set of tool names permitted for the given listener origin."""
    if origin.lower() == "mobile":
        return set(MOBILE_RESEARCH_TOOLS)
    if origin.lower() == "desktop":
        return MOBILE_RESEARCH_TOOLS | DESKTOP_CONTROL_TOOLS
    return set()
