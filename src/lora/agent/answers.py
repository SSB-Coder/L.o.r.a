"""Conversation, grounded research, and local-source answer workflows.

Methods retain the AgentRuntime contract and operate on its owned turn state.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any

from lora.agent.contracts import AgentExecutionTrace, TextCb, TurnResult
from lora.agent.prompting import ResearchPlan, make_plan
from lora.agent.research import Source
from lora.agent.source_check import source_metadata
from lora.dialogue import ConversationHistory, build_messages, build_research_messages
from lora.models.laya_router import LayaDecision

logger = logging.getLogger("lora.agent.runtime")


class AnswerWorkflows:
    async def _user_sources(self, q: str, start_index: int = 1) -> list[Source]:
        """Passages from the files the user added under 'Add sources' that are relevant to this question."""
        if not self.allow_actions:
            return []  # Mobile must never consult even an already-indexed local file.
        db = self.media_db
        if db is None or not getattr(db, "file_count", 0):
            return []
        try:
            hits = await asyncio.to_thread(db.search, q, 4, 0.40)
        except Exception as exc:
            logger.info("Searching the user's sources failed: %s", exc)
            return []
        out: list[Source] = []
        for h in hits:
            n = start_index + len(out)
            name = h.get("filename", "your file")
            page = f", page {h['page_number']}" if h.get("page_number") else ""
            if h.get("section"):
                page += f", section: {h['section']}"
            out.append(
                Source(f"S{n}", f"Your file: {name}{page}", "", text=str(h.get("text", ""))[:2200])
            )
        return out

    async def _run_chat(
        self,
        q: str,
        on_text: TextCb,
        images: list[str] | None = None,
        attachment_context: str = "",
        trace: AgentExecutionTrace | None = None,
        user_sources: list[Source] | None = None,
        require_sources: bool = False,
    ) -> TurnResult:
        if images:
            role = "vision"
        else:
            role = "writer"
        if trace and role in ("vision", "complex"):
            trace.escalated = True

        mine = user_sources if user_sources is not None else await self._user_sources(q)
        if require_sources and not mine:
            text = "I couldn't find a relevant passage in your added sources. Try naming the document or giving me a little more detail."
            on_text(text, True)
            return TurnResult(
                text=text, kind="chat", meta={"retrieval": "local", "verdict": "unchecked"}
            )
        if mine:
            self.last_raw_sources = list(mine)
            block = "\n\n".join(f"[{s.id}] {s.title}\n{s.text}" for s in mine)
            if attachment_context:
                block = f"{block}\n\nAttached Context:\n{attachment_context}"
            msgs = build_research_messages(q, block, self.history, **self._persona_kwargs(q))
            if images:
                msgs[-1]["images"] = images
            text = await self._stream(msgs, on_text, speak=False, role=role)
            on_text(text, True)
            meta = source_metadata(text, mine)
            meta["retrieval"] = "local"
            return TurnResult(
                text=text,
                sources=[
                    {
                        "id": s.id,
                        "title": s.title,
                        "url": "",
                        "domain": "your files",
                        "quote": s.text[:600],
                    }
                    for s in mine
                ],
                kind="chat",
                meta=meta,
            )
        msgs = build_messages(
            q,
            self.history,
            self._profile_fn(),
            images=images,
            attachment_context=attachment_context,
            **self._persona_kwargs(q),
        )
        text = await self._stream(msgs, on_text, role=role)
        return TurnResult(text=text, kind="chat")

    def _emit_verification_preview(
        self, meta: dict[str, Any], trace: AgentExecutionTrace | None = None
    ) -> None:
        if trace:
            meta.setdefault("agent_trace", trace.to_meta())
        if callable(self.on_verification):
            try:
                self.on_verification(meta)
            except Exception:
                logger.debug("Optional runtime callback failed; continuing the turn", exc_info=True)

    async def _run_research(
        self,
        q: str,
        on_status: Callable[[str], None],
        on_text: TextCb,
        run_mode: str = "standard",
        trace: AgentExecutionTrace | None = None,
        decision: LayaDecision | None = None,
        local_sources: list[Source] | None = None,
        answer_depth: str | None = None,
    ) -> TurnResult:
        deep = run_mode in ("deep", "deep_with_refinement")
        started = time.perf_counter()
        if self._research_fn is None:
            from lora.agent.research import research as _research

            self._research_fn = _research
        # Keep the writer resident throughout research.
        plan_pair = await self._plan(role="writer")
        on_status("Planning the search…")
        rplan: ResearchPlan = (
            ResearchPlan(question=q, queries=[])
            if local_sources is not None
            else await make_plan(
                lambda msgs: self._collect(msgs, plan_pair=plan_pair, role="planner"),
                q,
                self.history.messages(),
            )
        )
        logger.info(
            "Prompt engineer: %r -> question=%r queries=%s", q, rplan.question, rplan.queries
        )
        sig_params = {}
        try:
            sig_params = inspect.signature(self._research_fn).parameters
            takes_queries = "queries" in sig_params
            takes_topic = "topic_type" in sig_params
        except (TypeError, ValueError):
            takes_queries = False
            takes_topic = False
        r_kwargs: dict[str, Any] = {}
        if takes_queries:
            r_kwargs["queries"] = rplan.queries
        if takes_topic and decision is not None:
            r_kwargs["topic_type"] = decision.topic_type
        if "embedding_engine" in sig_params and self.embedding_engine is not None:
            r_kwargs["embedding_engine"] = self.embedding_engine
        from lora.agent.research import ResearchResult

        planned_at = time.perf_counter()
        found = (
            ResearchResult(q, list(local_sources))
            if local_sources is not None
            else await self._research_fn(rplan.question, on_status, **r_kwargs)
        )
        quarantine_count = getattr(found, "quarantined_sources", 0)
        self._check_abort()
        # Local questions already pass their own sources. Never append arbitrary
        # local matches to a web search, including files added while it runs.
        found.sources = list(found.sources)
        retrieved_at = time.perf_counter()
        self.last_raw_sources = list(found.sources)
        if not found.sources:
            on_status("No usable sources found.")
            refusal = (
                "I could not find usable sources to substantiate this query. "
                "To prevent ungrounded claims or hallucinations, I am abstaining from generating an answer. "
                "Please provide relevant reference documents or refine your search query."
            )
            on_text(refusal, False)
            return TurnResult(
                text=refusal,
                sources=[],
                kind="abstention",
                meta={"verdict": "abstain", "score": 0.0, "reason": "empty_sources", "stages": []},
            )

        sources_ui = [
            {
                "id": s.id,
                "title": s.title,
                "url": s.url,
                "domain": s.domain or "your files",
                "tier": s.tier,
                "is_preprint": s.is_preprint,
                "quote": (s.text or s.snippet)[:600],
            }
            for s in found.sources
        ]
        if hasattr(self, "on_sources") and callable(self.on_sources):
            try:
                self.on_sources(sources_ui)
            except Exception:
                logger.debug("Optional runtime callback failed; continuing the turn", exc_info=True)
        on_status("Writing it up…")
        kwargs = self._persona_kwargs(q)
        if answer_depth in ("eli5", "simple", "standard", "detailed", "expert"):
            kwargs["addon"] += (
                f"\nRequested answer style: {answer_depth}. Match the explanation to this style."
            )
        expanded = deep or answer_depth in ("detailed", "expert")
        msgs = build_research_messages(
            rplan.question,
            found.prompt_block(),
            self.history,
            concise=False if expanded else None,
            **kwargs,
        )
        draft = await self._stream(msgs, on_text, speak=False, plan_pair=plan_pair)
        written_at = time.perf_counter()
        self._check_abort()

        # A research answer stays on the warm writer. Citation integrity is a
        # structural check, never a factual confidence score or a repair loop.
        meta = source_metadata(draft, found.sources)
        meta["run_mode"] = run_mode
        if quarantine_count:
            draft += (
                f"\n\n{quarantine_count} sources containing embedded instructions were excluded."
            )
        meta["timings"] = {
            "planning": round(planned_at - started, 3),
            "retrieval": round(retrieved_at - planned_at, 3),
            "writing": round(written_at - retrieved_at, 3),
            "reference_check": round(time.perf_counter() - written_at, 3),
            "retrieval_stages": getattr(found, "timings", {}),
        }
        logger.info("Research timings (seconds): %s", meta["timings"])
        self._emit_verification_preview(meta, trace)
        on_text(draft, False)
        spoken = None
        if deep:
            from lora.voice.speech_stream import split_complete

            spoken = " ".join(split_complete(draft + " ")[0][:2]) or None
        return TurnResult(text=draft, sources=sources_ui, kind="research", meta=meta, spoken=spoken)

    async def _summarize_sources(
        self, question: str, on_status: Callable[[str], None], on_text: TextCb
    ) -> TurnResult:
        """Summarize document-order batches, including later sections, without browsing."""
        if self.media_db is None or not self.media_db.file_count:
            text = "Add a document under Sources first, then I can summarize it."
            on_text(text, True)
            return TurnResult(text=text, kind="chat")
        on_status("Reading your documents.")
        snapshot = await asyncio.to_thread(self.media_db.summary_batches)
        batches = snapshot["batches"]
        if not batches:
            text = "Your sources contain no readable text to summarize."
            on_text(text, False)
            return TurnResult(text=text, kind="chat")
        plan_pair = await self._plan(role="writer")
        sources = {}
        for batch in batches:
            sources.setdefault(
                batch["id"], Source(batch["id"], batch["title"], "", text=batch["text"])
            )
        self.last_raw_sources = list(sources.values())
        instruction = (
            question
            + "\nSummarize the actual content: central ideas, arguments, examples, and conclusions. "
            "Distinguish the author's opinions from established facts. Keep each file identifiable and cite it. "
            "Do not return index statistics or a list of sample quotations."
        )
        notes = []
        for i, batch in enumerate(batches, 1):
            self._check_abort()
            on_status(f"Summarizing your sources: {i}/{len(batches)} sections.")
            block = f"[{batch['id']}] {batch['title']}\n{batch['text']}"
            messages = build_research_messages(
                instruction
                if len(batches) == 1
                else instruction
                + "\nThis is one document section. Capture its points in at most 180 words.",
                block,
                ConversationHistory(),
                concise=False,
                **self._persona_kwargs(question),
            )
            if len(batches) == 1:
                answer = await self._stream(messages, on_text, speak=False, plan_pair=plan_pair)
            else:
                note = await self._collect(messages, plan_pair=plan_pair)
                notes.append(f"[{batch['id']}] {batch['title']}\n{note[:3000]}")
        # Reduce bounded groups so large collections cannot overflow the fixed
        # writer window. Every section contributes; avoid sampling just the lead.
        while sum(map(len, notes)) > 11000:
            reduced = []
            for start in range(0, len(notes), 3):
                self._check_abort()
                messages = build_research_messages(
                    "Combine these section notes in at most 180 words. Preserve conclusions, caveats and source IDs.",
                    "\n\n".join(notes[start : start + 3]),
                    ConversationHistory(),
                    concise=False,
                )
                reduced.append((await self._collect(messages, plan_pair=plan_pair))[:3000])
            notes = reduced
        if len(batches) > 1:
            on_status("Bringing the document summaries together.")
            messages = build_research_messages(
                instruction,
                "\n\n".join(notes),
                ConversationHistory(),
                concise=False,
                **self._persona_kwargs(question),
            )
            answer = await self._stream(messages, on_text, speak=False, plan_pair=plan_pair)
        if snapshot["partial"]:
            answer += (
                f"\n\nThis summary covers {snapshot['included_chunks']} of {snapshot['total_chunks']} indexed passages. "
                "The collection exceeded this turn's limit; ask about a specific document for more detail."
            )
        meta = source_metadata(answer, sources.values())
        meta.update(
            {
                "retrieval": "local",
                "summary_coverage": {
                    key: snapshot[key] for key in ("included_chunks", "total_chunks", "partial")
                },
            }
        )
        on_text(answer, False)
        ui_sources = [
            {"id": source.id, "title": source.title, "url": "", "quote": source.text[:600]}
            for source in sources.values()
        ]
        return TurnResult(text=answer, sources=ui_sources, kind="research", meta=meta)
