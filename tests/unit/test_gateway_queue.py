"""Unit tests for Gateway Single-GPU Queue and Desktop Preemption (tests/unit/test_gateway_queue.py)."""

import pytest

from lora.gateway.queue import JobItem, JobOrigin, JobQueue, JobStatus


@pytest.mark.asyncio
async def test_job_queue_desktop_priority_ordering():
    queue = JobQueue()

    # Enqueue mobile job 1
    m1 = JobItem(job_id="mob-1", origin=JobOrigin.MOBILE, text="Search quantum")
    pos_m1 = await queue.enqueue(m1)
    assert pos_m1 == 1

    # Enqueue mobile job 2
    m2 = JobItem(job_id="mob-2", origin=JobOrigin.MOBILE, text="Search biology")
    pos_m2 = await queue.enqueue(m2)
    assert pos_m2 == 2

    # Enqueue desktop job - should jump ahead of mobile jobs in queue!
    d1 = JobItem(job_id="desk-1", origin=JobOrigin.DESKTOP, text="Open Spotify")
    pos_d1 = await queue.enqueue(d1)
    assert pos_d1 == 1

    # Queue positions after desktop insertion:
    assert queue.get_queue_position("desk-1") == 1
    assert queue.get_queue_position("mob-1") == 2
    assert queue.get_queue_position("mob-2") == 3


@pytest.mark.asyncio
async def test_job_queue_scoped_cancellation():
    queue = JobQueue()

    d1 = JobItem(job_id="desk-1", origin=JobOrigin.DESKTOP, text="Desktop job")
    m1 = JobItem(job_id="mob-1", origin=JobOrigin.MOBILE, text="Mobile job")

    await queue.enqueue(d1)
    await queue.enqueue(m1)

    # Mobile requester CANNOT cancel desktop job
    ok, msg = await queue.request_stop("desk-1", requester_origin=JobOrigin.MOBILE)
    assert ok is False
    assert "cannot cancel" in msg.lower() or "forbidden" in msg.lower()
    assert queue.get_job("desk-1").status != JobStatus.CANCELLED

    # Mobile requester CAN cancel mobile job
    ok, msg = await queue.request_stop("mob-1", requester_origin=JobOrigin.MOBILE)
    assert ok is True
    assert queue.get_job("mob-1").status == JobStatus.CANCELLED

    # Desktop requester CAN cancel desktop job
    ok, msg = await queue.request_stop("desk-1", requester_origin=JobOrigin.DESKTOP)
    assert ok is True
    assert queue.get_job("desk-1").status == JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_job_queue_node_boundary_preemption_signal():
    import dataclasses

    queue = JobQueue()

    # Start mobile job
    m1 = JobItem(job_id="mob-running", origin=JobOrigin.MOBILE, text="Deep research")
    await queue.enqueue(m1)
    m1_running = dataclasses.replace(m1, status=JobStatus.RUNNING)
    queue.set_active_job(m1_running)

    assert queue.active_job.preempt_requested is False

    # Desktop job arrives while mobile is running
    d1 = JobItem(job_id="desk-arrive", origin=JobOrigin.DESKTOP, text="Urgent voice command")
    await queue.enqueue(d1)

    # Active mobile job must have received preemption signal
    assert queue.active_job.preempt_requested is True
    assert queue.get_job("mob-running").preempt_requested is True


@pytest.mark.asyncio
async def test_job_queue_pop_next():
    queue = JobQueue()

    m1 = JobItem(job_id="mob-1", origin=JobOrigin.MOBILE, text="Mobile query")
    d1 = JobItem(job_id="desk-1", origin=JobOrigin.DESKTOP, text="Desktop query")

    await queue.enqueue(m1)
    await queue.enqueue(d1)

    # First popped should be the higher priority desktop job
    next_job = await queue.pop_next()
    assert next_job is not None
    assert next_job.job_id == "desk-1"

    # Second popped should be the mobile job
    next_job2 = await queue.pop_next()
    assert next_job2 is not None
    assert next_job2.job_id == "mob-1"

    # Now queue is empty
    assert await queue.pop_next() is None
