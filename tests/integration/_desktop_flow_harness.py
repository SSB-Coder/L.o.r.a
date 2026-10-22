"""Harness: runs the REAL run_desktop_app against a fake Ollama in offscreen Qt and prints a JSON result.

Used by test_desktop_query_flow.py (in a subprocess: it owns the QApplication and event loop).
"""
import json, os, sys, threading, time, argparse, tempfile
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["LORA_RETRY_DELAY"] = "0"
sys.path.insert(0, os.environ["LORA_SRC"])
mode = sys.argv[1]  # ok | down | think

from _controlled_desktop import configure
seen_payloads = configure(mode, voice=False)
url = "http://127.0.0.1:11434"

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv)

import lora.main as lm
from lora.ui.hud import LoraDesktopHUD

log = []; hud_ref = []
ui_errors = []
class UiErrors(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            ui_errors.append(record.getMessage())
logging.getLogger("lora.ui.threadsafe").addHandler(UiErrors())
class RecHUD(LoraDesktopHUD):
    def __init__(self, *a, **k): super().__init__(*a, **k); hud_ref.append(self)
    def stream_begin(self, q, label=""): log.append(("begin", threading.current_thread().name, label)); super().stream_begin(q, label)
    def stream_update(self, t): log.append(("update", threading.current_thread().name, t)); super().stream_update(t)
    def present_answer(self, q, a, sources=None, confidence=0.95, label="", streamed=False, meta=None):
        log.append(("present", threading.current_thread().name, a, label, streamed, sources))
        super().present_answer(q, a, sources, confidence, label, streamed, meta)
        QTimer.singleShot(150, lambda: None)
    def set_voice_state(self, st):
        log.append(("state", st)); super().set_voice_state(st)

tmp = Path(tempfile.mkdtemp())
cfg_mgr, user_cfg = lm.init_platform(base_dir=tmp)
user_cfg = user_cfg.model_copy(update={"models": user_cfg.models.model_copy(update={"ollama_url": url})})
args = argparse.Namespace(no_audio=True, headless=False, port=0, host="127.0.0.1", offline=False, config_dir=tmp)

t0 = [time.monotonic()]
done = {"v": False}
def finish_check():
    if mode == "sleepbusy" and time.monotonic() - t0[0] > 5:
        app.quit()
        return
    if mode == "interrupt" and len([e for e in log if e[0] == "present"]) == 1 and not done.get("followup"):
        done["followup"] = True
        hud_ref[0].query_submitted.emit("Good night", "")
        return
    if mode == "interrupt" and len([e for e in log if e[0] == "present"]) < 2:
        return
    if any(e[0] == "present" for e in log) and any(e[:2] == ("state", "idle") for e in log[[i for i,e in enumerate(log) if e[0]=="present"][0]:]):
        app.quit()
    elif time.monotonic() - t0[0] > 45:
        log.append(("TIMEOUT",)); app.quit()

def submit():
    t0[0] = time.monotonic()
    hud_ref[0].query_submitted.emit("tell me a joke about cats", "")
    if mode == "interrupt":
        QTimer.singleShot(250, lambda: hud_ref[0].query_submitted.emit("stop", ""))
    if mode == "sleepbusy":
        def sleep_now():
            log.append(("sleep-requested",))
            hud_ref[0].sleep_toggled.emit(True)
        QTimer.singleShot(250, sleep_now)
    t = QTimer(); t.setInterval(100); t.timeout.connect(finish_check); t.start(); done["t"] = t
QTimer.singleShot(500, submit)

with patch("lora.main.LoraDesktopHUD", RecHUD), \
     patch("lora.main.start_gateway_thread", return_value=None), \
     patch("lora.grounding.media_vector_db.MediaVectorDB", MagicMock(return_value=MagicMock(file_count=0))):
    rc = lm.run_desktop_app(cfg_mgr, user_cfg, args)

main_thread = threading.main_thread().name
present = [e for e in log if e[0] == "present"]
updates = [e for e in log if e[0] == "update"]
payload = seen_payloads[-1] if seen_payloads else {}
print("RESULT:" + json.dumps({
    "timeout": any(e[0] == "TIMEOUT" for e in log),
    "ui_calls_on_gui_thread": all(e[1] == main_thread for e in log if e[0] in ("begin", "update", "present")),
    "n_updates": len(updates),
    "last_update": updates[-1][2] if updates else None,
    "answer": present[0][2] if present else None,
    "label": present[0][3] if present else None,
    "streamed": present[0][4] if present else None,
    "sources": present[0][5] if present else None,
    "idle_after": bool(present) and log[-1][:2] == ("state", "idle"),
    "model": payload.get("model"), "think": payload.get("think"),
    "options": payload.get("options"), "keep_alive": payload.get("keep_alive"),
    "roles": [m["role"] for m in payload.get("messages", [])],
    "answers": [e[2] for e in present],
    "ui_errors": ui_errors,
    "states_after_sleep": [e[1] for e in log[next((i for i,e in enumerate(log) if e[0]=='sleep-requested'), len(log)):] if e[0]=='state'],
}))
