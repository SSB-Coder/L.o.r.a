"""Background worker for the gateway job queue (phone requests). Until now nothing consumed it.

Jobs run in the mobile-safe mode of the agent: chat and web research (with fact checking according to the
job's run_mode), never opening apps, files or typing. Desktop requests keep priority: the worker waits while a
desktop turn is running and re-queues the job if a desktop job asks it to yield.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from collections.abc import Callable
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
    ) -> None:
        super().__init__(daemon=True, name="lora-job-worker")
        self.queue, self._runtime_factory = queue, runtime_factory
        self._loop = asyncio.new_event_loop()
        self._desktop_busy, self._poll = desktop_busy, poll_s
        self._stop_evt = threading.Event()
        self.processed = 0

    def stop(self) -> None:
        self._stop_evt.set()

    def _call(self, coro: Any) -> Any:
        return self._loop.run_until_complete(coro)

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
        except JobAborted as exc:
            if exc.reason == "preempted":  # yield to the desktop and run again afterwards
                self.queue.set_active_job(None)
                try:
                    self._call(self.queue.enqueue(dataclasses.replace(job, status=JobStatus.QUEUED, preempt_requested=False)))
                except Exception:
                    logger.exception("Could not re-queue preempted job")
                return
            done = dataclasses.replace(running, status=JobStatus.CANCELLED)
        except Exception as exc:
            logger.exception("Job %s failed", job.job_id)
            done = dataclasses.replace(running, status=JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
        self.queue.set_active_job(done)  # records the final state for GET /v1/jobs/{id}
        self.queue.set_active_job(None)
        self.processed += 1
