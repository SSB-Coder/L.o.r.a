"""Own desktop turns until their asyncio task has actually finished cleanup.

Cancellation of a concurrent Future is not evidence that its asyncio task has
finished. The separate completion event below is set by the task's done callback.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


async def drain_pending_tasks() -> None:
    """Finish task cancellation before the owner closes its event loop."""
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.get_running_loop().shutdown_asyncgens()


async def run_owned_thread(function: Callable, *args, on_cancel: Callable[[], None]):
    """Drain a cooperative worker before releasing its owning asyncio turn."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        on_cancel()
        try:
            await task
        except Exception:
            logger.debug("Cancelled worker finished cleanup", exc_info=True)
        raise


class TurnHandle:
    def __init__(self, turn_id: int = 0) -> None:
        self.turn_id = turn_id
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.submitted = threading.Event()
        self._lock = threading.Lock()
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def cancel(self) -> None:
        self.cancelled.set()
        with self._lock:
            task, loop = self._task, self._loop
        if task is not None and loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(task.cancel)

    def finish_if_unsubmitted(self) -> None:
        if not self.submitted.is_set():
            self.finished.set()


class TurnCoordinator:
    """One active desktop turn, plus identity checks for queued UI callbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: TurnHandle | None = None
        self._sequence = 0

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active is not None and not self._active.finished.is_set()

    def begin(self) -> TurnHandle | None:
        with self._lock:
            if self._active is not None and not self._active.finished.is_set():
                return None
            self._sequence += 1
            self._active = TurnHandle(self._sequence)
            return self._active

    def is_current(self, handle: TurnHandle) -> bool:
        with self._lock:
            return self._active is handle and not handle.cancelled.is_set()

    def owns(self, handle: TurnHandle) -> bool:
        with self._lock:
            return self._active is handle

    def cancel(self) -> None:
        with self._lock:
            handle = self._active
        if handle is not None:
            handle.cancel()


def run_coroutine(
    loop: asyncio.AbstractEventLoop,
    coroutine: Coroutine[Any, Any, Any],
    timeout: float,
    handle: TurnHandle | None = None,
) -> Any:
    """Wait from a worker thread; timeouts cancel the actual task, not just a proxy.

    Callers must keep ownership while ``handle.finished`` is false. Synchronous
    native work may delay cancellation; never start another turn to bypass it.
    """
    handle = handle or TurnHandle()
    result: concurrent.futures.Future = concurrent.futures.Future()

    def done(task: asyncio.Task) -> None:
        try:
            if task.cancelled():
                result.set_exception(concurrent.futures.CancelledError())
            else:
                error = task.exception()
                if error is not None:
                    result.set_exception(error)
                else:
                    result.set_result(task.result())
        finally:
            # Completion means all coroutine finally blocks have run.
            handle.finished.set()

    def start() -> None:
        task = loop.create_task(coroutine)
        with handle._lock:
            handle._task, handle._loop = task, loop
        task.add_done_callback(done)
        if handle.cancelled.is_set():
            task.cancel()

    handle.submitted.set()
    try:
        loop.call_soon_threadsafe(start)
    except RuntimeError:
        coroutine.close()
        handle.finished.set()
        raise
    try:
        value = result.result(timeout=timeout)
        handle.finished.wait(1.0)
        return value
    except concurrent.futures.TimeoutError:
        handle.cancel()
        if not handle.finished.wait(2.0):
            logger.warning("Turn %d cancellation pending; retaining ownership", handle.turn_id)
        raise
