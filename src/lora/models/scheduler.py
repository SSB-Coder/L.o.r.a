"""One model call at a time, shared across gateway threads and event loops."""

import asyncio
import functools
import inspect
import threading
from contextlib import asynccontextmanager

_GPU_LOCK = threading.Lock()


@asynccontextmanager
async def model_slot():
    # Non-blocking acquire avoids leaving an orphan thread holding the lock if
    # a waiting task is cancelled. A streaming call holds the slot to completion.
    while not _GPU_LOCK.acquire(blocking=False):
        await asyncio.sleep(0.02)
    try:
        yield
    finally:
        _GPU_LOCK.release()


def serialized_model_call(fn):
    if inspect.isasyncgenfunction(fn):
        @functools.wraps(fn)
        async def stream(*args, **kwargs):
            async with model_slot():
                async for item in fn(*args, **kwargs):
                    yield item
        return stream

    @functools.wraps(fn)
    async def call(*args, **kwargs):
        async with model_slot():
            return await fn(*args, **kwargs)
    return call
