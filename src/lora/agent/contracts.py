"""Shared turn results, traces, and agent descriptions; no UI or model loading."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

ENGINE_DOWN = (
    "I can't reach my local engine right now. Make sure Ollama is running — "
    "I'll be ready as soon as it is."
)

NO_MODELS = "My local engine is running but has no models installed yet."


class EngineUnavailable(Exception):  # noqa: N818 - established public API
    """The local model could not be reached (details are logged, not shown)."""


@dataclass(frozen=True)
class SubAgentSpec:
    id: str
    name: str
    role: str
    description: str
    default_model_role: str = "writer"


SUBAGENTS: dict[str, SubAgentSpec] = {
    "supervisor": SubAgentSpec(
        id="supervisor",
        name="Supervisor & Laya Router Agent",
        role="Orchestrator & Decision Router",
        description="Classifies intent, depth, and risk via Laya on CPU, selects the optimal model & skill, and coordinates specialist sub-agents.",
        default_model_role="planner",
    ),
    "system_navigator": SubAgentSpec(
        id="system_navigator",
        name="System & Web Navigator Agent",
        role="Control & Navigation",
        description="Launches apps, opens site-specific topic pages (Wikipedia, YouTube, arXiv), and delegates to external AIs.",
        default_model_role="writer",
    ),
    "research": SubAgentSpec(
        id="research",
        name="Deep Research & Retrieval Agent",
        role="Web & Local RAG",
        description="Plans search queries, fetches web & scholarly sources, and synthesizes cited answers with [S1] grounding.",
        default_model_role="writer",
    ),
    "workspace_code": SubAgentSpec(
        id="workspace_code",
        name="Workspace & Code Agent",
        role="Files, Code & ToolLoop",
        description="Searches and reads local files, analyzes pasted attachments/images, and runs multi-step tool loops.",
        default_model_role="complex",
    ),
    "artifact_studio": SubAgentSpec(
        id="artifact_studio",
        name="Artifact & Visual Studio Agent",
        role="File & Image Generation",
        description="Creates standalone code/markdown/data files and generates or edits inline SVG/PNG diagrams and images.",
        default_model_role="complex",
    ),
    "critic_verifier": SubAgentSpec(
        id="critic_verifier",
        name="Source Reference Reviewer",
        role="Source References",
        description="Reviews supplied source references without assigning factual confidence scores.",
        default_model_role="writer",
    ),
}


@dataclass
class AgentExecutionTrace:
    """Tracks Laya routing, specialist sub-agents, and skills that participated in a turn for UI visibility."""

    active_skill: str | None = None
    model_used: str = ""
    model_mode: str = "auto"
    escalated: bool = False
    laya_decision: dict[str, Any] = field(default_factory=dict)
    agents_invoked: list[dict[str, str]] = field(default_factory=list)

    def record(self, agent_id: str, action: str, status: str = "completed") -> None:
        spec = SUBAGENTS.get(agent_id)
        name = spec.name if spec else agent_id
        self.agents_invoked.append(
            {
                "id": agent_id,
                "name": name,
                "action": action,
                "status": status,
            }
        )

    def to_meta(self) -> dict[str, Any]:
        return {
            "active_skill": self.active_skill or "auto",
            "model_used": self.model_used,
            "model_mode": self.model_mode,
            "escalated": self.escalated,
            "laya": self.laya_decision,
            "agents": list(self.agents_invoked),
        }


@dataclass
class TurnResult:
    text: str = ""
    sources: list[dict[str, str]] = field(default_factory=list)
    kind: str = "chat"
    failure: str | None = None
    spoken: str | None = None  # when set, TTS says this instead of the visible text
    meta: dict[str, Any] = field(
        default_factory=dict
    )  # verification / sub-agent / artifact metadata for the UI


TextCb = Callable[[str, bool], None]
