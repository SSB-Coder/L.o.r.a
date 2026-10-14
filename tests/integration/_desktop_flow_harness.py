"""Harness: runs the REAL run_desktop_app against a fake Ollama in offscreen Qt and prints a JSON result.

Used by test_desktop_query_flow.py (in a subprocess: it owns the QApplication and event loop).
"""
import json, os, sys, threading, time, argparse, tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["LORA_RETRY_DELAY"] = "0"
sys.path.insert(0, os.environ["LORA_SRC"])
mode = sys.argv[1]  # ok | down | think

seen_payloads = []

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({"models": [{"name": "SparkLLM/Spark-X2.5-4B:latest", "size": 1}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); payload = json.loads(self.rfile.read(n))
        if self.path.endswith("/api/generate"):  # model warm-up at startup
            body = b'{"done": true}'
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        seen_payloads.append(payload)
        self.send_response(200); self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        toks = ["<think>hmm</think>", "Right ", "away, ", "boss. ", "Anything else?"] if mode == "think" else ["Right ", "away, ", "boss. ", "Anything else?"]
        for t in toks:
            line = json.dumps({"message": {"role": "assistant", "content": t}, "done": False}).encode() + b"\n"
            self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n"); self.wfile.flush(); time.sleep(0.12)
        line = json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}).encode() + b"\n"
        self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n0\r\n\r\n"); self.wfile.flush()

srv = ThreadingHTTPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
if mode != "down":
    threading.Thread(target=srv.serve_forever, daemon=True).start()
else:
    srv.server_close()  # nothing listening: connections are refused
url = f"http://127.0.0.1:{port}"

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv)

import lora.main as lm
from lora.ui.hud import LoraDesktopHUD

log = []; hud_ref = []
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

done = {"v": False}
def finish_check():
    if any(e[0] == "present" for e in log) and any(e[:2] == ("state", "idle") for e in log[[i for i,e in enumerate(log) if e[0]=="present"][0]:]):
        app.quit()
    elif time.monotonic() - t0 > 20:
        log.append(("TIMEOUT",)); app.quit()
t0 = time.monotonic()
def submit():
    hud_ref[0].query_submitted.emit("explain gradient descent briefly", "")
    t = QTimer(); t.setInterval(100); t.timeout.connect(finish_check); t.start(); done["t"] = t
QTimer.singleShot(500, submit)

with patch("lora.main.LoraDesktopHUD", RecHUD), \
     patch("lora.main.start_gateway_thread", return_value=None), \
     patch("lora.grounding.media_vector_db.MediaVectorDB", MagicMock()):
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
}))
