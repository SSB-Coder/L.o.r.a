"""Background worker for the gateway job queue (phone requests).

Jobs run in the mobile-safe mode of the agent: chat and web research (with fact checking according to the
job's run_mode), never opening apps, files or typing. Desktop requests keep priority: the worker waits while a
desktop turn is running and re-queues the job if a desktop job asks it to yield.
Also broadcasts real-time WebSocket events (`state`, `queue`, `answer`, `error`) to connected mobile clients.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from lora.gateway.queue import JobItem, JobQueue, JobStatus

logger = logging.getLogger(__name__)


class JobAborted(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class JobWorker(threading.Thread):
    def __init__(
        self,
        queue: JobQueue,
        runtime_factory: Callable[[str], Any],
        desktop_busy: Callable[[], bool] = lambda: False,
        poll_s: float = 0.5,
        broadcaster: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(daemon=True, name="lora-job-worker")
        self.queue, self._runtime_factory = queue, runtime_factory
        self._loop = asyncio.new_event_loop()
        self._desktop_busy, self._poll = desktop_busy, poll_s
        self._broadcaster = broadcaster
        self._stop_evt = threading.Event()
        self.processed = 0

    def stop(self) -> None:
        self._stop_evt.set()

    def _call(self, coro: Any) -> Any:
        return self._loop.run_until_complete(coro)

    def _emit_ws(self, event: dict[str, Any]) -> None:
        if self._broadcaster is None:
            return
        try:
            self._call(self._broadcaster(event))
        except Exception:
            pass

    def run(self) -> None:
        asyncio.set_event_loop(self._loop)
        runtimes: dict[str, Any] = {}
        while not self._stop_evt.is_set():
            try:
                if self._desktop_busy():
                    self._stop_evt.wait(self._poll)
                    continue
                job = self._call(self.queue.pop_next())
            except Exception:
                logger.exception("Job worker poll failed")
                self._stop_evt.wait(self._poll)
                continue
            if job is None:
                self._stop_evt.wait(self._poll)
                continue
            if job.cancel_requested or job.status == JobStatus.CANCELLED:
                continue
            origin = str(job.origin)
            if origin not in runtimes:
                runtimes[origin] = self._runtime_factory(origin)
            self.process(job, runtimes[origin])

    # ---------------------------------------------------------------- one job
    def _current(self, job: JobItem) -> JobItem:
        return self.queue.get_job(job.job_id) or job

    def process(self, job: JobItem, runtime: Any) -> None:
        started = time.monotonic()
        running = dataclasses.replace(job, status=JobStatus.RUNNING)
        self.queue.set_active_job(running)
        self._emit_ws({"type": "queue", "position": 0})
        self._emit_ws({"type": "state", "state": "thinking"})

        def should_abort() -> str | None:
            cur = self._current(job)
            if cur.cancel_requested:
                return "cancelled"
            if cur.preempt_requested or self._desktop_busy():
                return "preempted"
            return None

        try:
            result = self._call(runtime.run_job(job.text, job.run_mode, should_abort))
            done = dataclasses.replace(running, status=JobStatus.COMPLETED, result=result)
            logger.info("Job %s done in %.1fs (%s)", job.job_id, time.monotonic() - started, job.run_mode)
            if isinstance(result, dict):
                ver = result.get("verification") or {}
                checked = ver.get("checked") or {}
                score = float(checked.get("score", 0.95))
                conf = "HIGH" if score >= 0.85 else ("MEDIUM" if score >= 0.65 else "LOW")
                citations = [
                    {
                        "id": s.get("id", f"S{i+1}"),
                        "title": s.get("title", ""),
                        "source_url": s.get("url", ""),
                        "quote": s.get("domain", ""),
                        "tier": 1,
                    }
                    for i, s in enumerate(result.get("sources") or [])
                    if isinstance(s, dict)
                ]
                self._emit_ws({
                    "type": "answer",
                    "job_id": job.job_id,
                    "text": result.get("answer", ""),
                    "citations": citations,
                    "confidence": conf,
                    "unverified": checked.get("unverified") or [],
                })
        except JobAborted as exc:
            if exc.reason == "preempted":  # yield to the desktop and run again afterwards
                self.queue.set_active_job(None)
                try:
                    self._call(self.queue.enqueue(dataclasses.replace(job, status=JobStatus.QUEUED, preempt_requested=False)))
                except Exception:
                    logger.exception("Could not re-queue preempted job")
                return
            done = dataclasses.replace(running, status=JobStatus.CANCELLED)
            self._emit_ws({"type": "state", "state": "idle"})
        except Exception as exc:
            logger.exception("Job %s failed", job.job_id)
            done = dataclasses.replace(running, status=JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            self._emit_ws({"type": "error", "job_id": job.job_id, "message": str(exc)})
        self.queue.set_active_job(done)  # records the final state for GET /v1/jobs/{id}
        self.queue.set_active_job(None)
        self.processed += 1
