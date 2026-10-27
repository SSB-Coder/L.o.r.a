"""L.O.R.A Multi-Agent Runtime: coordinates routing, cited research, document synthesis, skills, and artifacts.

Qt-free so it can be tested on its own. The UI and TTS hang off the callbacks:
  on_status(text)        short progress line ("Searching the web…")
  on_text(full, speak)   the visible reply so far; ``speak=False`` keeps TTS quiet for that update
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

from lora.agent.actions import ActionWorkflows
from lora.agent.answers import AnswerWorkflows
from lora.agent.commands import (
    Command,
    capability_reply,
    is_deep,
    is_research,
    is_smalltalk,
    looks_like_filename,
    looks_like_url,
    note_title,
    parse_command,
    parse_natural_intent_llm,
    quick_reply,
    resolve_folder,
)
from lora.agent.contracts import (
    ENGINE_DOWN,
    NO_MODELS,
    SUBAGENTS,
    AgentExecutionTrace,
    EngineUnavailable,
    SubAgentSpec,
    TextCb,
    TurnResult,
)
from lora.agent.deep import pick_judge_model
from lora.agent.fuzzy import best_key
from lora.agent.notes import create_artifact, generate_visual, process_attachments
from lora.agent.prompting import ResearchPlan, make_plan
from lora.agent.research import Source
from lora.agent.source_check import source_metadata
from lora.agent.toolloop import ToolLoop, looks_agentic
from lora.agent.verification import VerifierLLM, cards_from_sources, verify_answer
from lora.dialogue import (
    ConversationHistory,
    build_messages,
    build_note_messages,
    build_research_messages,
    strip_think,
)
from lora.models.laya_router import LayaDecision, LayaDecisionRouter
from lora.net import async_client
from lora.safety.gate import DataProvenance

# Preserve the original runtime import surface while implementations live in focused modules.
__all__ = [
    "AgentExecutionTrace",
    "AgentRuntime",
    "Any",
    "Awaitable",
    "Callable",
    "Command",
    "ConversationHistory",
    "DataProvenance",
    "ENGINE_DOWN",
    "EngineUnavailable",
    "LayaDecision",
    "LayaDecisionRouter",
    "NO_MODELS",
    "Path",
    "ResearchPlan",
    "SUBAGENTS",
    "Source",
    "SubAgentSpec",
    "TextCb",
    "ToolLoop",
    "TurnResult",
    "VerifierLLM",
    "aclosing",
    "async_client",
    "asyncio",
    "best_key",
    "build_messages",
    "build_note_messages",
    "build_research_messages",
    "capability_reply",
    "cards_from_sources",
    "classify_agent_route",
    "create_artifact",
    "dataclass",
    "field",
    "generate_visual",
    "inspect",
    "is_deep",
    "is_research",
    "is_smalltalk",
    "logger",
    "logging",
    "looks_agentic",
    "looks_like_filename",
    "looks_like_url",
    "make_plan",
    "note_title",
    "os",
    "parse_command",
    "parse_natural_intent_llm",
    "pick_judge_model",
    "process_attachments",
    "quick_reply",
    "quote_plus",
    "re",
    "resolve_folder",
    "source_metadata",
    "strip_think",
    "time",
    "urlparse",
    "verify_answer",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


_ARTIFACT_INTENT_RE = re.compile(
    r"\b(?:create|make|generate|write|save|export|build|draw|render|plot|design)\s+"
    r"(?:me\s+)?(?:a\s+|an\s+|the\s+|new\s+)*"
    r"(?:python\s+script|script|code\s+file|html\s+file|csv\s+file|json\s+file|svg|diagram|chart|plot|flowchart|image|picture|illustration|visual)\b",
    re.IGNORECASE,
)


def classify_agent_route(
    query: str,
    has_command: bool,
    is_research_query: bool,
    is_agentic_query: bool,
    has_attachments: bool = False,
    skill_agent: str = "auto",
) -> str:
    """Determine the primary specialist agent route when not explicitly overridden."""
    if skill_agent and skill_agent != "auto" and skill_agent in SUBAGENTS:
        return skill_agent
    if has_command:
        return "system_navigator"
    if _ARTIFACT_INTENT_RE.search(query or ""):
        return "artifact_studio"
    if is_research_query:
        return "research"
    if is_agentic_query or has_attachments:
        return "workspace_code"
    return "chat"


async def _deny(_text: str) -> bool:
    return False


class AgentRuntime(AnswerWorkflows, ActionWorkflows):
    def __init__(
        self,
        ollama_url: str,
        user_default_model: str | None,
        history: ConversationHistory,
        registry_factory: Callable[[], Any],
        profile_fn: Callable[[], str] = lambda: "",
        research_fn: Callable[..., Any] | None = None,
        media_db: Any = None,
        allow_actions: bool = True,
        mcp_config_path: Any = None,
        skills_dir: Any = None,
        agents_dir: Any = None,
        notebook_db_path: Path | None = None,
        laya_router: LayaDecisionRouter | None = None,
        embedding_engine: Any = None,
    ) -> None:
        self.ollama_url = ollama_url
        self.user_default_model = user_default_model
        self.manual_model: str | None = (
            None  # None = Auto (Smart Routing); otherwise explicit Ollama model tag
        )
        self.active_skill: str | None = (
            None  # None = Auto trigger matching; otherwise explicit skill name
        )
        self.history = history
        self._registry_factory = registry_factory
        self._registry: Any = None
        self._profile_fn = profile_fn
        self._research_fn = research_fn
        self.diag_extras: Callable[[], dict[str, Any]] = lambda: {}
        self.media_db = media_db
        self.embedding_engine = (
            embedding_engine
            if embedding_engine is not None
            else getattr(media_db, "embedding_engine", None)
        )
        self._should_abort: Callable[[], str | None] = lambda: None
        self.allow_actions = allow_actions  # False for phone jobs: chat and research only
        self.allow_typing = allow_actions  # explicit desktop command plus exact confirmation only
        self.confirm: Callable[[str], Awaitable[bool]] = (
            _deny  # user OK for sensitive actions (typing)
        )
        self.typer: Any = None
        self.persona: str | None = (
            None  # ~/.lora/agents/persona.md overrides the built-in personality
        )
        self.skills: Any = None  # built-in SKILL.md + ~/.lora/skills/*
        self.mcp_hub: Any = None
        self._mcp_config_path = mcp_config_path
        self._mcp_lock = __import__("threading").Lock()
        self.laya_router = (
            laya_router if laya_router is not None else LayaDecisionRouter(lazy_load=True)
        )
        self.last_laya_decision: LayaDecision | None = None
        self._notebook_db_path = notebook_db_path
        self._load_personalization(skills_dir, agents_dir)
        self.retry_delay = float(os.environ.get("LORA_RETRY_DELAY", "1.5"))  # seconds
        self.last_answer = ""
        self.last_question = ""
        self.last_sources: list[dict[str, str]] = []
        self.last_raw_sources: list[Source] = []
        self.last_model_used = ""
        self.on_model_changed: Callable[[str, list[str]], None] | None = None
        self.on_skill_changed: Callable[[str], None] | None = None
        self.on_voice_changed_cb: Callable[[str], None] | None = None
        self.on_verification: Callable[[dict[str, Any]], None] | None = None
        self.on_power_command: Callable[[str], None] | None = None

    def on_sources_changed(self) -> None:
        """Clear cached conversation context when a FAISS media source is removed so deleted sources are never echoed."""
        self.history.clear()
        self.last_sources.clear()
        self.last_raw_sources.clear()

    # ------------------------------------------------------- model & skill controls
    def set_model(self, model_name: str | None) -> str:
        clean = (model_name or "").strip()
        if not clean or clean.lower() in ("auto", "smart", "default"):
            self.manual_model = None
            return "Model routing set to **Auto (Smart Complexity Routing)**."
        self.manual_model = clean
        return f"Switched active model to **{clean}**."

    def set_skill(self, skill_name: str | None) -> str:
        clean = (skill_name or "").strip().lower()
        if not clean or clean in ("auto", "none", "off"):
            self.active_skill = None
            return "Skill mode set to **Auto (Intent Matching)**."
        if self.skills is not None:
            sk = self.skills.get_skill(clean)
            if sk is not None:
                self.active_skill = sk.name
                return f"Activated skill **{sk.name}** ({sk.description})."
        self.active_skill = clean
        return f"Activated skill **{clean}**."

    def list_skills(self) -> list[dict[str, Any]]:
        if self.skills is None:
            return []
        try:
            loaded = self.skills.skills or self.skills.load_skills()
            return [
                {
                    "name": s.name,
                    "description": s.description,
                    "agent": s.agent,
                    "default_depth": s.default_depth,
                    "builtin": s.builtin,
                }
                for s in loaded.values()
            ]
        except Exception:
            return []

    @staticmethod
    def list_subagents() -> list[dict[str, str]]:
        return [
            {"id": a.id, "name": a.name, "role": a.role, "description": a.description}
            for a in SUBAGENTS.values()
        ]

    async def available_models(self) -> list[str]:
        from lora.models.ollama_client import OllamaClient

        try:
            return await OllamaClient(base_url=self.ollama_url).installed_models()
        except Exception:
            return []

    # ------------------------------------------------------- personalization & MCP
    def _load_personalization(self, skills_dir: Any, agents_dir: Any) -> None:
        from pathlib import Path as _Path

        try:
            if agents_dir and (_Path(agents_dir) / "persona.md").is_file():
                raw_p = (
                    (_Path(agents_dir) / "persona.md").read_text(encoding="utf-8")[:2000].strip()
                )
                # Strip legacy "Think JARVIS to their Tony Stark" from any persisted persona.md
                raw_p = re.sub(
                    r"Think\s+JARVIS\s+to\s+their\s+Tony\s+Stark\s*:?\s*",
                    "",
                    raw_p,
                    flags=re.IGNORECASE,
                ).strip()
                self.persona = raw_p or None
            from lora.skills.loader import SkillLoader

            self.skills = SkillLoader(
                _Path(skills_dir) if skills_dir else None, include_builtin=True
            )
            self.skills.load_skills()
        except Exception as exc:
            logger.warning("Could not load skills/persona: %s", exc)

    def _skill_for(self, q: str) -> Any:
        try:
            return (
                self.skills.match_skill(q, explicit_skill=self.active_skill)
                if self.skills is not None
                else None
            )
        except Exception:
            return None

    def _persona_kwargs(self, q: str) -> dict[str, Any]:
        skill = None if is_smalltalk(q) else self._skill_for(q)
        return {
            "persona": self.persona,
            "addon": (skill.system_prompt_addon or "") if skill else "",
        }

    def start_mcp(self) -> None:
        """Connect the servers in ~/.lora/mcp.json (blocking; call from a background thread). Safe to call twice."""
        if not self.allow_actions or os.environ.get("LORA_ENABLE_MCP", "0") != "1":
            logger.info("MCP startup requires desktop scope and LORA_ENABLE_MCP=1")
            return
        with self._mcp_lock:
            if self.mcp_hub is not None or not self._mcp_config_path:
                return
            from lora.tools.mcp import McpClientManager
            from lora.tools.mcp_stdio import McpHub

            manager = McpClientManager(
                mcp_config_path=self._mcp_config_path, gate=self._get_registry().action_gate
            )
            hub = McpHub(manager)
            hub.connect_all()
            self.mcp_hub = hub

    def mcp_tool_defs(self) -> dict[str, Any]:
        return self.mcp_hub.manager.tools if self.mcp_hub is not None else {}

    def _mentions_mcp(self, q: str) -> bool:
        names = {d.server_name.lower() for d in self.mcp_tool_defs().values()}
        return bool(names) and any(re.search(rf"\b{re.escape(n)}\b", q.lower()) for n in names)

    async def call_mcp(self, name: str, args: dict[str, Any], tainted: bool) -> dict[str, Any]:
        if self.mcp_hub is None or name not in self.mcp_tool_defs():
            return {"status": "error", "message": f"Unknown tool '{name}'."}
        prov = {k: (DataProvenance.UNTRUSTED if tainted else DataProvenance.USER) for k in args}
        res = await asyncio.to_thread(
            self.mcp_hub.manager.execute_tool, name, args, "desktop", prov
        )
        if res.success:
            return {"status": "success", "text": res.content}
        return {
            "status": "denied" if "gate" in (res.error or "").lower() else "error",
            "reason": res.error,
            "message": res.error,
        }

    # ---------------------------------------------------------------- public
    async def run_turn(
        self,
        query: str,
        on_status: Callable[[str], None],
        on_text: TextCb,
        mode: str | None = None,
        should_abort: Callable[[], str | None] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        answer_depth: str | None = None,
    ) -> TurnResult:
        """Route a turn; all research modes use cited synthesis and reference checks."""
        q = query.strip()
        self._should_abort = should_abort or (lambda: None)
        if self._registry is not None:
            self._registry.action_gate.reset_task()

        # Clock/date and capability answers need neither a model nor web research.
        if not attachments:
            canned = quick_reply(q) or capability_reply(q)
            if canned:
                self.history.add(q, canned)
                on_text(canned, True)
                return TurnResult(text=canned, kind="smalltalk")

        images_b64, attachment_context, image_paths = process_attachments(attachments)
        skill = self._skill_for(q)

        # 1. Route every utterance through Laya Decision Router on CPU (0 MB GPU VRAM)
        route = self.laya_router.predict
        if os.environ.get("LORA_NEURAL_ROUTING", "0") != "1" and isinstance(
            self.laya_router, LayaDecisionRouter
        ):
            route = self.laya_router.predict_fast
        decision = route(q)
        self.last_laya_decision = decision

        trace = AgentExecutionTrace(
            active_skill=skill.name if skill else None,
            model_mode=self.manual_model or "auto",
            laya_decision=decision.model_dump(),
        )
        trace.record(
            "supervisor",
            f"Laya ({decision.intent} · {decision.depth_mode} · {decision.action_risk})",
        )

        try:
            cmd = parse_command(q) if not attachments else None
            summary_request = bool(
                re.search(
                    r"\b(?:summari[sz]e|summary|overview)\b.*\b(?:my|these|added|uploaded)\s+(?:sources|files|documents)\b",
                    q,
                    re.IGNORECASE,
                )
            )
            has_local_sources = (
                self.allow_actions
                and self.media_db is not None
                and bool(getattr(self.media_db, "file_count", 0))
            )
            casual = is_smalltalk(q)
            explicit_web = bool(
                re.search(r"\b(?:online|web|internet|latest|news)\b", q, re.IGNORECASE)
            )
            mine = (
                await self._user_sources(q)
                if has_local_sources
                and cmd is None
                and not casual
                and not attachments
                and not summary_request
                and not explicit_web
                else []
            )
            local_context_request = bool(
                has_local_sources
                and cmd is None
                and not explicit_web
                and not casual
                and not attachments
                and not looks_agentic(q)
            )
            if (
                cmd is None
                and not attachments
                and self.allow_actions
                and decision.intent == "control"
                and not is_research(q)
            ):
                cmd = await parse_natural_intent_llm(
                    q,
                    lambda msgs: self._collect(msgs, role="controller"),
                )
            if (
                cmd is not None
                and not self.allow_actions
                and cmd.kind not in ("rewrite_depth", "list_notebooks")
            ):
                msg = "That kind of request only works from the desktop app."
                on_text(msg, True)
                return TurnResult(text=msg, kind="command")

            if summary_request and self.allow_actions and not attachments:
                result = await self._summarize_sources(q, on_status, on_text)
            else:
                result = (
                    await self._run_command(cmd, q, on_status, on_text, trace=trace)
                    if cmd is not None
                    else None
                )
            if result is not None:
                pass
            elif local_context_request and (
                mode in ("standard", "deep", "deep_with_refinement")
                or re.search(r"\b(?:fact[- ]?check|verif(?:y|ied|ication))\b", q, re.IGNORECASE)
            ):
                result = await self._run_research(
                    q,
                    on_status,
                    on_text,
                    mode or "standard",
                    trace=trace,
                    decision=decision,
                    local_sources=mine,
                    answer_depth=answer_depth,
                )
            elif (
                not casual
                and not local_context_request
                and (
                    is_research(q)
                    or (is_deep(q) and len(q.split()) >= 5)
                    or (skill and skill.agent == "research")
                    or (
                        decision.intent == "research"
                        and not local_context_request
                        and not (
                            self.allow_actions
                            and (
                                looks_agentic(q)
                                or (skill and skill.agent in ("workspace_code", "artifact_studio"))
                            )
                        )
                    )
                )
            ):
                trace.record("research", f"Multi-source retrieval (topic={decision.topic_type})")
                skill_mode = (
                    skill.default_depth
                    if self.active_skill
                    and skill
                    and skill.default_depth in ("quick", "deep", "deep_with_refinement")
                    else None
                )
                explicit_verify = bool(
                    re.search(
                        r"\b(?:fact[- ]?check|verif(?:y|ied|ication)|double[- ]?check)\b",
                        q,
                        re.IGNORECASE,
                    )
                )
                run_mode = (
                    mode
                    or skill_mode
                    or ("deep" if is_deep(q) else "standard" if explicit_verify else "quick")
                )
                result = await self._run_research(
                    q,
                    on_status,
                    on_text,
                    run_mode,
                    trace=trace,
                    decision=decision,
                    answer_depth=answer_depth,
                )
            elif self.allow_actions and (
                looks_agentic(q)
                or self._mentions_mcp(q)
                or (skill and skill.agent in ("workspace_code", "artifact_studio"))
                or (
                    image_paths
                    and re.search(
                        r"\b(edit|modify|grayscale|invert|caption|annotate)\b", q, re.IGNORECASE
                    )
                )
            ):
                trace.record("workspace_code", "Multi-step ToolLoop execution")
                result = await self._run_agent(
                    q,
                    on_status,
                    on_text,
                    extra_context=attachment_context,
                    image_paths=image_paths,
                    trace=trace,
                )
            else:
                route_id = classify_agent_route(
                    q,
                    False,
                    False,
                    False,
                    has_attachments=bool(attachments),
                    skill_agent=skill.agent if skill else "auto",
                )
                if route_id != "chat":
                    trace.record(route_id, "Contextual synthesis")
                result = await self._run_chat(
                    q,
                    on_text,
                    images=images_b64,
                    attachment_context=attachment_context,
                    trace=trace,
                    user_sources=mine,
                    require_sources=local_context_request,
                )
        except EngineUnavailable as exc:
            logger.warning("Engine unavailable: %s", exc)
            msg = NO_MODELS if "no models" in str(exc) else ENGINE_DOWN
            return TurnResult(text=msg, kind="error", failure=msg)
        except Exception as exc:
            if type(exc).__name__ == "JobAborted" or "JobAborted" in str(type(exc)):
                logger.info("Turn cancelled by user stop: %s", exc)
                raise  # Keep cancellation/preemption distinct from a completed job.
            logger.exception("Turn failed")
            msg = "Something went wrong on my side. Try that again in a moment."
            return TurnResult(text=msg, kind="error", failure=f"{type(exc).__name__}: {exc}")

        if (
            result.kind in ("chat", "research", "agent", "note")
            and result.text
            and not result.failure
        ):
            self.history.add(q, result.text)
            self.last_answer, self.last_question, self.last_sources = result.text, q, result.sources
            trace.model_used = self.last_model_used
            if result.meta:
                result.meta.setdefault("agent_trace", trace.to_meta())
                result.meta.setdefault("laya_decision", decision.model_dump())
        return result

    async def run_job(
        self,
        text: str,
        run_mode: str,
        should_abort: Callable[[], str | None],
        notebook_id: str | None = None,
        depth: str | None = None,
        checkpoint: dict[str, Any] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """A phone job: same pipeline, answer returned as data for GET /v1/jobs/{id}."""
        from lora.agent.jobs import JobAborted

        self._abort_exc = JobAborted
        if checkpoint and checkpoint.get("last_sources") and not self.last_raw_sources:
            self.last_raw_sources = list(checkpoint["last_sources"])
        if notebook_id and not self.last_raw_sources:
            try:
                from lora.storage.notebook import DEFAULT_NOTEBOOK_DB, NotebookStore

                store = NotebookStore(db_path=self._notebook_db_path or DEFAULT_NOTEBOOK_DB)
                await store.initialize()
                nb = await store.get_notebook(notebook_id)
                if nb is not None:
                    self.last_answer = nb.answer
                    self.last_question = nb.title
                    self.last_raw_sources = [
                        Source(
                            id=c.card_id,
                            title=c.title,
                            url=c.source_url,
                            text=c.exact_quote,
                            snippet=c.claim_fact,
                        )
                        for c in nb.cards
                    ]
            except Exception as exc:
                logger.debug("Could not load notebook %s for job: %s", notebook_id, exc)
        prev_len = [0]

        def _on_job_text(full_text: str, _speak: bool = True) -> None:
            if on_delta is not None and len(full_text) > prev_len[0]:
                delta_chunk = full_text[prev_len[0] :]
                prev_len[0] = len(full_text)
                try:
                    on_delta(delta_chunk)
                except Exception:
                    pass

        effective_mode = run_mode
        if (
            depth
            and depth.lower() in ("eli5", "simple", "standard", "detailed", "expert")
            and self.last_raw_sources
            and text.lower().startswith("rewrite")
        ):
            res = await self._do_rewrite_depth(depth, lambda _s: None, _on_job_text)
        else:
            res = await self.run_turn(
                text,
                lambda _s: None,
                _on_job_text,
                mode=effective_mode,
                should_abort=should_abort,
                answer_depth=depth,
            )
        if res.failure and res.kind == "error":
            raise RuntimeError(res.failure)
        return {
            "answer": res.text,
            "sources": res.sources,
            "verification": res.meta,
            "kind": res.kind,
            "run_mode": effective_mode,
        }

    def _check_abort(self) -> None:
        from lora.safety.killswitch import KillSwitch

        if KillSwitch.get_global().is_halted():
            reason = KillSwitch.get_global().halt_reason or "Emergency kill switch activated"
            exc = getattr(self, "_abort_exc", None)
            if exc is None:
                from lora.agent.jobs import JobAborted as exc  # noqa: N813 - local exception type
            raise exc(reason)
        reason = self._should_abort()
        if reason:
            exc = getattr(self, "_abort_exc", None)
            if exc is None:
                from lora.agent.jobs import JobAborted as exc  # noqa: N813 - local exception type
            raise exc(reason)

    # ----------------------------------------------------------------- model
    async def _plan(self, role: str = "writer"):
        from lora.models.ollama_client import OllamaClient
        from lora.models.selection import plan_chat

        client = OllamaClient(base_url=self.ollama_url)
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                plan = await plan_chat(
                    client,
                    user_default=self.user_default_model,
                    role=role,
                    manual_model=self.manual_model,
                )
                if plan is None:
                    raise EngineUnavailable("no models installed")
                self.last_model_used = plan.model
                return client, plan
            except EngineUnavailable:
                raise
            except Exception as exc:
                last_exc = exc
                logger.info("Ollama not ready (attempt %d): %s", attempt + 1, exc)
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        raise EngineUnavailable(str(last_exc))

    async def _collect(
        self,
        messages: list[dict[str, Any]],
        plan_pair: Any = None,
        model: str | None = None,
        keep_alive: str | None = None,
        role: str = "writer",
        options_override: dict[str, Any] | None = None,
    ) -> str:
        """Run the model to completion without showing the text (critic, rewrite, tool loop)."""
        client, plan = plan_pair or await self._plan(role=role)
        chosen = model or plan.model
        opts = dict(plan.options)
        if options_override:
            # Never override num_ctx to a different value on the same model (prevents Ollama VRAM reload thrashing),
            # but allow smaller num_ctx when switching to a different judge/verifier model so 8B models fit in 6GB VRAM.
            for k, v in options_override.items():
                if k != "num_ctx" or "num_ctx" not in opts or chosen != plan.model:
                    opts[k] = v
        self.last_model_used = chosen
        logger.info("Model (%s): %s (options=%s think=%s)", role, chosen, opts, plan.think)
        text = ""
        async with aclosing(
            client.stream_chat(
                chosen,
                messages,
                options=opts,
                think=plan.think,
                keep_alive=keep_alive or plan.keep_alive,
            )
        ) as stream:
            async for delta in stream:
                self._check_abort()
                text += delta
        return strip_think(text)

    async def _stream(
        self,
        messages: list[dict[str, Any]],
        on_text: TextCb,
        speak: bool = True,
        plan_pair: Any = None,
        role: str = "writer",
    ) -> str:
        from lora.models.selection import pick_escalation_model

        client, plan = plan_pair or await self._plan(role=role)
        chosen_model = plan.model
        self.last_model_used = chosen_model
        logger.info(
            "Model (%s): %s (options=%s think=%s)", role, chosen_model, plan.options, plan.think
        )
        answer = ""
        for attempt in range(2):
            try:
                async with aclosing(
                    client.stream_chat(
                        chosen_model,
                        messages,
                        options=plan.options,
                        think=plan.think,
                        keep_alive=plan.keep_alive,
                    )
                ) as stream:
                    async for delta in stream:
                        self._check_abort()
                        answer += delta
                        shown = strip_think(answer, streaming=True)
                        if shown:
                            on_text(shown, speak)
                if strip_think(answer):
                    break
                if attempt == 0 and not self.manual_model:
                    installed = await client.installed_models()
                    esc = pick_escalation_model(installed, chosen_model)
                    if esc:
                        logger.info("Auto-escalating empty reply from %s -> %s", chosen_model, esc)
                        chosen_model = esc
                        self.last_model_used = esc
                        answer = ""
                        continue
                break
            except Exception as exc:
                if type(exc).__name__ == "JobAborted":
                    raise
                if answer or attempt == 1:
                    raise EngineUnavailable(str(exc)) from exc
                logger.info("Stream failed before first token, retrying: %s", exc)
                await asyncio.sleep(self.retry_delay)
        final = strip_think(answer)
        if not final:
            raise EngineUnavailable("model returned an empty reply")
        return final

    async def run_rsi_outer_loop(
        self,
        question: str,
        sources: list[Source],
        max_iters: int = 5,
    ) -> dict[str, Any]:
        """Run the formal Phase 10 RSIEngine (PromptGenome evolution + rollback) over retrieved sources."""
        import tempfile

        from lora.outer_loop.genome import (
            EditValidator,
            GenomeRegistry,
            PromptGenome,
            TemplateCompiler,
        )
        from lora.outer_loop.loop import RSIEngine
        from lora.outer_loop.optimizer import PromptOptimizerAgent
        from lora.outer_loop.scorer import OuterLoopScorer

        client, plan = await self._plan(role="writer")
        installed = await client.installed_models()
        verifier_model, _ = (
            pick_judge_model(installed, plan.model) if installed else (plan.model, False)
        )
        cards = [{"source_title": s.title, "snippet": (s.text or s.snippet)[:400]} for s in sources]

        base_genome = PromptGenome(
            system_prompt_base="You are L.O.R.A, a grounded research assistant.",
            refinement_rules=[
                "Assert only claims directly grounded in evidence.",
                "Cite sources inline as [S1].",
            ],
            writer_style="concise_academic",
            version=0,
        )
        registry = GenomeRegistry(Path(tempfile.gettempdir()) / "lora_rsi_genomes.db")
        engine = RSIEngine(
            registry=registry,
            compiler=TemplateCompiler(),
            optimizer=PromptOptimizerAgent(edit_validator=EditValidator()),
            scorer=OuterLoopScorer(),
            max_iters=max_iters,
        )
        import threading

        from lora.agent.jobs import JobAborted
        from lora.agent.turns import TurnHandle, run_coroutine, run_owned_thread

        loop = asyncio.get_running_loop()
        stopped = threading.Event()
        child_lock = threading.Lock()
        active_child = [None]
        parent_abort = self._should_abort

        def call_owned(factory):
            child = TurnHandle()
            with child_lock:
                if stopped.is_set() or parent_abort():
                    raise JobAborted("cancelled")
                active_child[0] = child
            try:
                return run_coroutine(loop, factory(), 60.0, child)
            finally:
                with child_lock:
                    if active_child[0] is child:
                        active_child[0] = None

        def stop_refinement():
            stopped.set()
            with child_lock:
                child = active_child[0]
            if child is not None:
                child.cancel()

        def synth_sync(compiled_prompt: str) -> str:
            return call_owned(
                lambda: self._collect(
                    [{"role": "user", "content": compiled_prompt}], (client, plan)
                )
            )

        def verify_sync(ans: str) -> dict[str, Any]:
            v = call_owned(
                lambda: verify_answer(
                    ans, sources, VerifierLLM(client, verifier_model, plan.options)
                )
            )
            return {
                "composite_faithfulness_score": v.score,
                "supported_claims_ratio": (v.supported / v.total) if v.total else 0.0,
                "unverified_critical_claims": v.unverified,
            }

        traj = await run_owned_thread(
            engine.run,
            base_genome,
            question,
            cards,
            synth_sync,
            verify_sync,
            on_cancel=stop_refinement,
        )
        return {
            "best_answer": traj.best_answer,
            "best_score": traj.best_score,
            "baseline_score": traj.baseline_score,
            "net_improvement": traj.net_improvement,
            "iterations_run": traj.iterations_run,
            "genome_version": traj.best_genome.version,
        }

    # -------------------------------------------------------------- commands
    def _get_registry(self) -> Any:
        if self._registry is None:
            self._registry = self._registry_factory()
        return self._registry

    async def _tool(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._tool_with_provenance(
            name, params, {k: DataProvenance.USER for k in params}
        )

    async def _tool_with_provenance(
        self, name: str, params: dict[str, Any], prov: dict[str, Any]
    ) -> dict[str, Any]:
        reg = self._get_registry()
        from lora.gateway.gating import get_allowed_tools_for_origin

        origin = "desktop" if self.allow_actions else "mobile"
        if name not in get_allowed_tools_for_origin(origin):
            return {"status": "denied", "reason": "Tool excluded from this origin's graph"}
        if name in {"create_note", "export_report", "ask_other_ai"}:
            import json

            if not await self.confirm(
                f"Allow {name}?\n\n{json.dumps(params, ensure_ascii=False, indent=2)}"
            ):
                return {"status": "denied", "reason": "User confirmation required"}
            return await reg.aexecute(
                name, dict(params), origin=origin, provenance=prov, confirmed=True
            )
        return await reg.aexecute(name, dict(params), origin=origin, provenance=prov)

    async def _web_search_tool(self, query: str) -> dict[str, Any]:
        import httpx

        from lora.agent.research import HEADERS, search_with_report

        async with async_client(HEADERS, httpx.Timeout(12.0, connect=6.0)) as client:
            hits, report = await search_with_report(query, client, limit=6)
        if not hits:
            return {
                "status": "error",
                "message": "The web search returned nothing ("
                + "; ".join(report["errors"][:2])
                + ").",
            }
        return {
            "status": "success",
            "text": "\n".join(f"{h.title} — {h.url}\n  {h.snippet}" for h in hits),
        }

    async def _fetch_page_tool(self, url: str) -> dict[str, Any]:
        from lora.agent.research import HEADERS, fetch_text

        async with async_client(HEADERS, 15.0) as client:
            text = await fetch_text(url, client)
        return (
            {"status": "success", "text": text}
            if text
            else {"status": "error", "message": "Couldn't read that page."}
        )

    async def _run_agent(
        self,
        q: str,
        on_status: Callable[[str], None],
        on_text: TextCb,
        extra_context: str = "",
        image_paths: list[str] | None = None,
        trace: AgentExecutionTrace | None = None,
    ) -> TurnResult:
        if image_paths and re.search(
            r"\b(edit|modify|grayscale|invert|caption|annotate)\b", q, re.IGNORECASE
        ):
            on_status("Editing image artifact…")
            res = generate_visual(
                title=note_title(q, "Edited Image"),
                spec=q,
                source_image_path=image_paths[0],
                policy=self._get_registry().action_gate.policy_engine,
            )
            if res.get("status") == "success" and res.get("artifact"):
                art = res["artifact"]
                msg = f"I created the modified image **{art['name']}** (`{art['path']}`)."
                on_text(msg, True)
                return TurnResult(text=msg, kind="agent", meta={"artifacts": [art]})

        loop = ToolLoop(self)
        final = await loop.run(
            q, on_status, on_text, self.history.messages(), extra_context=extra_context
        )
        on_text(final, True)
        meta: dict[str, Any] = {}
        if loop.artifacts:
            meta["artifacts"] = loop.artifacts
        if loop.steps_taken and trace:
            for st in loop.steps_taken:
                trace.record("workspace_code", f"Tool: {st['tool']} ({st['status']})")
        return TurnResult(text=final, kind="agent", meta=meta)

    # -------------------------------------------------------- typing into editors

    # ----------------------------------------------------------------- notes & artifacts

    # ---------------------------------------------------------------- ask AI
