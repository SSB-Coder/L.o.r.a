"""Opt-in Windows test runner when sandbox policy blocks asyncio's TCP self-pipe.

Uses the real IOCP event loop, tasks, cancellation and transports. A 10ms poll
deadline replaces the socket used only to wake the loop from another thread.
Production startup and ordinary pytest never import or enable this adapter.
"""

import asyncio
import gc
import sys
import threading
import os
import tempfile
from pathlib import Path


def configure() -> None:
    if sys.platform != "win32":
        return
    os.environ["LORA_SANDBOX_TESTS"] = "1"
    temp = Path(__file__).resolve().parents[1] / 'scratch' / f'sandbox-temp-{os.getpid()}'
    temp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temp)
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--no-sandbox --disable-gpu')

    class SandboxLoop(asyncio.ProactorEventLoop):
        def _make_self_pipe(self):
            self._ssock = self._csock = None

        def _close_self_pipe(self):
            pass

        def _loop_self_reading(self, future=None):
            pass

        def _run_once(self):
            if not self._scheduled or self._scheduled[0]._when > self.time() + 0.01:
                self.call_later(0.01, lambda: None)
            super()._run_once()

    class SandboxPolicy(asyncio.WindowsProactorEventLoopPolicy):
        def new_event_loop(self):
            # Proactor's main-thread constructor also installs a socket fd for
            # OS signals. Construct on a helper thread to omit that test-only
            # signal-wakeup facility; IOCP task/transport behavior is unchanged.
            created = []
            def create():
                created.append(SandboxLoop())
            thread = threading.Thread(target=create)
            # Qt objects left in cycles must be collected on the GUI thread,
            # never incidentally while allocating the helper event loop.
            collecting = gc.isenabled()
            if threading.current_thread() is threading.main_thread():
                gc.collect()
            gc.disable()
            try:
                thread.start()
                thread.join()
            finally:
                if collecting:
                    gc.enable()
            return created[0]

    asyncio.set_event_loop_policy(SandboxPolicy())


if __name__ == "__main__":
    configure()
    import pytest
    raise SystemExit(pytest.main(sys.argv[1:]))
