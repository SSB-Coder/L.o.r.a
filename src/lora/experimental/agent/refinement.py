"""Explicit recursive prompt-refinement experiment, outside ordinary research."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from lora.agent.research import Source


async def run_rsi_outer_loop(
    self,
    question: str,
    sources: list[Source],
    max_iters: int = 5,
    verify_fn=None,
) -> dict[str, Any]:
    """Run the formal Phase 10 RSIEngine (PromptGenome evolution + rollback) over retrieved sources."""
    import tempfile

    from lora.experimental.agent.deep import pick_judge_model
    from lora.experimental.agent.verification import VerifierLLM, verify_answer

    verify_answer = verify_fn or verify_answer

    from lora.experimental.outer_loop.genome import (
        EditValidator,
        GenomeRegistry,
        PromptGenome,
        TemplateCompiler,
    )
    from lora.experimental.outer_loop.loop import RSIEngine
    from lora.experimental.outer_loop.optimizer import PromptOptimizerAgent
    from lora.experimental.outer_loop.scorer import OuterLoopScorer

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
