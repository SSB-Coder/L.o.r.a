"""Single-GPU job queue with desktop priority and node-boundary preemption for L.O.R.A Gateway."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class JobStatus(StrEnum):
    """Execution status of a gateway job."""

    QUEUED = "queued"
    RUNNING = "running"
    PREEMPTED = "preempted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobOrigin(StrEnum):
    """Source listener of the incoming request."""

    DESKTOP = "desktop"
    MOBILE = "mobile"


@dataclass(frozen=True)
class JobItem:
    """An individual request queued for processing."""

    job_id: str
    origin: JobOrigin
    text: str
    notebook_id: str | None = None
    depth: str | None = None
    run_mode: str = "quick"
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.monotonic)
    preempt_requested: bool = False
    cancel_requested: bool = False
    result: Any | None = None
    error: str | None = None


class JobQueue:
    """Thread-safe FIFO queue with desktop priority and node-boundary preemption."""

    def __init__(self) -> None:
        self._queue: list[JobItem] = []
        self._jobs: dict[str, JobItem] = {}
        self._active_job: JobItem | None = None
        self._lock = asyncio.Lock()

    async def enqueue(self, job: JobItem) -> int:
        """Enqueue a job with priority rules: Desktop jobs jump ahead of mobile jobs."""
        async with self._lock:
            self._jobs[job.job_id] = job

            if job.origin == JobOrigin.DESKTOP:
                # Insert ahead of pending mobile jobs, keeping FIFO for desktop jobs
                insert_idx = len(self._queue)
                for i, existing in enumerate(self._queue):
                    if existing.origin == JobOrigin.MOBILE:
                        insert_idx = i
                        break
                self._queue.insert(insert_idx, job)

                # If a mobile job is currently active on the GPU, signal preemption immutably
                if self._active_job is not None and self._active_job.origin == JobOrigin.MOBILE:
                    preempted = dataclasses.replace(self._active_job, preempt_requested=True)
                    self._active_job = preempted
                    self._jobs[preempted.job_id] = preempted
            else:
                self._queue.append(job)

            # Return 1-based position in queue
            return self._queue.index(job) + 1

    def get_queue_position(self, job_id: str) -> int:
        """Get 1-based queue position (0 if active/running, -1 if finished/not found)."""
        if self._active_job is not None and self._active_job.job_id == job_id:
            return 0

        for i, job in enumerate(self._queue):
            if job.job_id == job_id:
                return i + 1
        return -1

    def get_job(self, job_id: str) -> JobItem | None:
        """Retrieve job by id."""
        return self._jobs.get(job_id)

    def set_active_job(self, job: JobItem | None) -> None:
        """Update currently running job on the GPU."""
        self._active_job = job
        if job is not None:
            self._jobs[job.job_id] = job

    @property
    def active_job(self) -> JobItem | None:
        """Get currently running job on the GPU."""
        return self._active_job

    async def pop_next(self) -> JobItem | None:
        """Pop next highest-priority job from queue."""
        async with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    async def request_stop(
        self, job_id: str, requester_origin: JobOrigin
    ) -> tuple[bool, str]:
        """Cancel a job, enforcing origin-scoped cancellation boundaries."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False, f"Job '{job_id}' not found"

            # Enforce scoping: Mobile clients cannot cancel desktop jobs
            if requester_origin == JobOrigin.MOBILE and job.origin == JobOrigin.DESKTOP:
                return False, "Mobile origin cannot cancel desktop jobs"

            cancelled_job = dataclasses.replace(
                job, cancel_requested=True, status=JobStatus.CANCELLED
            )
            self._jobs[job_id] = cancelled_job

            if job in self._queue:
                idx = self._queue.index(job)
                self._queue[idx] = cancelled_job
                self._queue.remove(cancelled_job)

            if self._active_job is not None and self._active_job.job_id == job_id:
                self._active_job = cancelled_job

            return True, f"Job '{job_id}' cancelled"
