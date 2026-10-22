"""Background worker for the gateway job queue (phone requests).

Jobs run in the mobile-safe mode of the agent: chat and cited web research with lightweight reference checks, never opening apps, files or typing. Desktop requests keep priority: the worker waits while a
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


class JobAborted(Exception):  # noqa: N818 - public cancellation API
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
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self._active_task: asyncio.Task | None = None

    def stop(self) -> None:
        self._stop_evt.set()
        task = self._active_task
        if task is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(task.cancel)

    def _call(self, coro: Any) -> Any:
        task = self._loop.create_task(coro)
        self._active_task = task
        try:
            return self._loop.run_until_complete(task)
        finally:
            self._active_task = None

    def _emit_ws(self, event: dict[str, Any], origin: str = "mobile") -> None:
        if self._broadcaster is None:
            return
        payload = {"origin": origin, **event}
        try:
            self._call(self._broadcaster(payload))
        except (Exception, asyncio.CancelledError):
            pass

    def run(self) -> None:
        asyncio.set_event_loop(self._loop)
        runtimes: dict[str, Any] = {}
        try:
            while not self._stop_evt.is_set():
                try:
                    if self._desktop_busy():
                        self._stop_evt.wait(self._poll)
                        continue
                    job = self._call(self.queue.pop_next())
                except asyncio.CancelledError:
                    break
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
                try:
                    if origin not in runtimes:
                        runtimes[origin] = self._runtime_factory(origin)
                    self.process(job, runtimes[origin])
                except asyncio.CancelledError:
                    if self._stop_evt.is_set():
                        break
                    logger.warning("Job worker operation cancelled")
                except Exception as exc:
                    logger.exception("Job runtime failed; worker remains available")
                    self.queue.set_active_job(dataclasses.replace(job, status=JobStatus.FAILED, error=str(exc)))
                    self.queue.set_active_job(None)

        finally:
            try:
                from lora.agent.turns import drain_pending_tasks

                self._loop.run_until_complete(drain_pending_tasks())
                for runtime in runtimes.values():
                    registry = getattr(runtime, "_registry", None)
                    if registry is not None:
                        try:
                            self._loop.run_until_complete(registry.close())
                        except Exception:
                            logger.exception("Job registry cleanup failed")
            finally:
                self._loop.close()

    # ---------------------------------------------------------------- one job
    def _current(self, job: JobItem) -> JobItem:
        return self.queue.get_job(job.job_id) or job

    def process(self, job: JobItem, runtime: Any) -> None:
        started = time.monotonic()
        running = dataclasses.replace(job, status=JobStatus.RUNNING)
        self.queue.set_active_job(running)
        job_origin = str(job.origin.value if hasattr(job.origin, "value") else job.origin)
        self._emit_ws({"type": "queue", "position": 0}, origin=job_origin)
        self._emit_ws({"type": "state", "state": "thinking"}, origin=job_origin)

        def should_abort() -> str | None:
            cur = self._current(job)
            if cur.cancel_requested:
                return "cancelled"
            if cur.preempt_requested or self._desktop_busy():
                return "preempted"
            return None

        try:
            sig = getattr(runtime.run_job, "__code__", None)
            arg_names = sig.co_varnames if sig else ()
            job_kwargs: dict[str, Any] = {}
            if "notebook_id" in arg_names:
                job_kwargs["notebook_id"] = job.notebook_id
            if "depth" in arg_names:
                job_kwargs["depth"] = job.depth
            if "checkpoint" in arg_names and job.job_id in self.checkpoints:
                job_kwargs["checkpoint"] = self.checkpoints[job.job_id]
            if "on_delta" in arg_names:
                job_kwargs["on_delta"] = lambda delta: self._emit_ws(
                    {"type": "partial_text", "job_id": job.job_id, "text": delta},
                    origin=job_origin,
                )
            result = self._call(runtime.run_job(job.text, job.run_mode, should_abort, **job_kwargs))
            if self._current(job).cancel_requested:
                raise JobAborted("cancelled")
            self.checkpoints.pop(job.job_id, None)
            done = dataclasses.replace(running, status=JobStatus.COMPLETED, result=result)
            logger.info("Job %s done in %.1fs (%s)", job.job_id, time.monotonic() - started, job.run_mode)
            if isinstance(result, dict):
                from lora.agent.source_check import mobile_answer

                self._emit_ws({"type": "answer", "job_id": job.job_id, **mobile_answer(result)}, origin=job_origin)
        except asyncio.CancelledError:
            done = dataclasses.replace(running, status=JobStatus.CANCELLED)
        except JobAborted as exc:
            if exc.reason == "preempted":  # yield to the desktop and run again afterwards
                self.checkpoints[job.job_id] = {
                    "last_sources": list(getattr(runtime, "last_raw_sources", []) or []),
                    "last_question": getattr(runtime, "last_question", job.text),
                }
                self.queue.set_active_job(None)
                try:
                    self._call(self.queue.enqueue(dataclasses.replace(job, status=JobStatus.QUEUED, preempt_requested=False)))
                except Exception:
                    logger.exception("Could not re-queue preempted job")
                return
            done = dataclasses.replace(running, status=JobStatus.CANCELLED)
            self._emit_ws({"type": "state", "state": "idle"}, origin=job_origin)
        except Exception as exc:
            logger.exception("Job %s failed", job.job_id)
            done = dataclasses.replace(running, status=JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            self._emit_ws({"type": "error", "job_id": job.job_id, "message": str(exc)}, origin=job_origin)
        self.queue.set_active_job(done)  # records the final state for GET /v1/jobs/{id}
        self.queue.set_active_job(None)
        self.checkpoints.pop(job.job_id, None)
        self.processed += 1
