"""Origin capability gating for L.O.R.A Gateway.

Enforces that remote mobile clients have access STRICTLY to Tier 0 research tools.
Local filesystem tools (read_file, list_dir, search_files, open_path) and desktop
UI controls (open_app, list_windows, focus_window) are completely stripped from mobile
execution graphs.
"""

from __future__ import annotations

# Strict allowlist of Tier 0 research tools permitted for mobile origin
MOBILE_RESEARCH_TOOLS: frozenset[str] = frozenset({
    "search_web",
    "fetch_pdf",
    "read_web_page",
    "ask_other_ai",
})

# Full tool set permitted for authenticated desktop listener
DESKTOP_CONTROL_TOOLS: frozenset[str] = frozenset({
    "search_web",
    "fetch_pdf",
    "read_web_page",
    "ask_other_ai",
    "read_file",
    "list_dir",
    "search_files",
    "open_path",
    "open_app",
    "open_url",
    "focus_window",
    "list_windows",
})


def get_allowed_tools_for_origin(origin: str) -> frozenset[str]:
    """Return the set of tools accessible to a given connection origin."""
    if origin.lower() == "mobile":
        return MOBILE_RESEARCH_TOOLS
    return DESKTOP_CONTROL_TOOLS
