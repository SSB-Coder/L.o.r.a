"""UiDispatcher must run callables on the GUI thread, from any thread."""

import sys
import threading
import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from lora.ui.threadsafe import UiDispatcher  # noqa: E402


@pytest.fixture(scope="module")
def core_app():
    # QApplication (not QCoreApplication): other UI tests in the same session create widgets.
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def _spin(app, predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        app.processEvents()
        time.sleep(0.005)


def test_post_from_worker_thread_runs_on_gui_thread(core_app):
    ui = UiDispatcher()
    seen = []
    gui_thread = threading.current_thread()

    def worker():
        ui.post(lambda: seen.append(threading.current_thread()))

    threading.Thread(target=worker).start()
    _spin(core_app, lambda: seen)
    assert seen == [gui_thread]


def test_exception_in_callback_does_not_break_dispatcher(core_app):
    ui = UiDispatcher()
    seen = []

    def boom():
        raise RuntimeError("boom")

    threading.Thread(target=lambda: (ui.post(boom), ui.post(lambda: seen.append(1)))).start()
    _spin(core_app, lambda: seen)
    assert seen == [1]


def test_plain_singleshot_from_thread_never_fires(core_app):
    """Documents WHY the dispatcher exists (this is the bug that dropped every reply)."""
    fired = []
    threading.Thread(target=lambda: QTimer.singleShot(0, lambda: fired.append(1))).start()
    _spin(core_app, lambda: fired, timeout=0.5)
    assert fired == []
