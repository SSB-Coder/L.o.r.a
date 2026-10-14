"""Harness: hands-free voice + push-to-talk through the REAL run_desktop_app (fake mic thread, scripted STT,
fake Ollama). Prints a JSON result; used by test_desktop_voice_flow.py in a subprocess.
"""
import json, os, sys, threading, time, argparse, tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
sys.modules["sounddevice"] = MagicMock()  # sandbox has no PortAudio; importing the real one is slow and fails

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["LORA_RETRY_DELAY"] = "0"
os.environ["LORA_TTS_FAKE"] = "1"
sys.path.insert(0, os.environ["LORA_SRC"])
mode = sys.argv[1]  # voice | nomic

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({"models": [{"name": "gemma4:e4b"}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        if self.path.endswith("/api/generate"):
            body = b'{"done": true}'; self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        self.send_response(200); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        for obj in ({"message": {"content": "Sure thing."}, "done": False}, {"message": {"content": ""}, "done": True}):
            line = json.dumps(obj).encode() + b"\n"; self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
srv = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}"

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv)

import lora.main as lm
from lora.ui.hud import LoraDesktopHUD
import lora.voice.capture as capmod, lora.voice.stt as sttmod

transcripts = []          # scripted STT output, consumed in order
class FakeTranscriber:
    def __init__(self, *a, **k): pass
    def transcribe(self, audio, sample_rate=16000):
        text = transcripts.pop(0) if transcripts else ""
        return MagicMock(text=text)

class FakeBuffer:
    def __init__(self): self.data = np.zeros(0, dtype=np.float32)
    def clear(self): self.data = np.zeros(0, dtype=np.float32)
    def read_last_n_seconds(self, s): return self.data[-int(s * 16000):]

class FakeCapture:
    registry = []
    def __init__(self, *a, **k):
        self.cbs = []; self.buffer = FakeBuffer(); FakeCapture.registry.append(self)
        self.ok = (mode != "nomic"); self.device_name = "Fake USB Mic" if self.ok else ""
    def register_chunk_callback(self, cb): self.cbs.append(cb)
    def start(self): return self.ok
    @property
    def is_streaming(self): return self.ok
    def seconds_since_audio(self): return 0.05 if self.ok else None
    def status_text(self): return self.device_name if self.ok else "no input device found (PortAudioError: none)"
    def stop(self): pass
    def feed(self, chunk):
        for cb in self.cbs: cb(chunk)

rng = np.random.default_rng(1)
def speak(cap, seconds=0.7):
    for _ in range(int(seconds / 0.032)): cap.feed((rng.standard_normal(512) * 0.1).astype(np.float32)); time.sleep(0.002)
def silence(cap, seconds=1.0):
    for _ in range(int(seconds / 0.032)): cap.feed(np.zeros(512, dtype=np.float32)); time.sleep(0.002)
def utterance(cap): speak(cap); silence(cap); time.sleep(0.7)

queries, msgs, mic, states = [], [], [], []
hud_ref = []
class RecHUD(LoraDesktopHUD):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); hud_ref.append(self)
    def append_message(self, s, t):
        msgs.append((s, t)); (queries.append(t) if s == 'You' else None); super().append_message(s, t)
    def set_mic_status(self, ok, text): mic.append((ok, text)); super().set_mic_status(ok, text)
    def set_voice_state(self, st): states.append(st); super().set_voice_state(st)

tmp = Path(tempfile.mkdtemp()); os.chdir(tmp)       # no speaker profile in cwd
cfg_mgr, user_cfg = lm.init_platform(base_dir=tmp)
user_cfg = user_cfg.model_copy(update={"models": user_cfg.models.model_copy(update={"ollama_url": url})})
args = argparse.Namespace(no_audio=False, headless=False, port=0, host="127.0.0.1", offline=False, config_dir=tmp)

def scenario():
    cap = FakeCapture.registry[-1]
    def run():
        try:
            if mode == "voice":
                transcripts.extend(["we should get lunch at noon", "Hey LORA what's the plan", "hey laura", "tell me a joke"])
                utterance(cap)   # (a) ordinary talk -> must be ignored
                results["a_ignored"] = list(queries) == []
                utterance(cap)   # (b) wake phrase + query in one go
                time.sleep(1.2)  # let the reply finish (inflight clears)
                results["b_query"] = list(queries)
                utterance(cap)   # (c) wake phrase alone -> armed
                results["c_state_armed"] = states[-1] == "listening"
                utterance(cap)   # (d) next utterance becomes the query
                time.sleep(1.2)
                results["d_queries"] = list(queries)
                # (e) push-to-talk: hold, audio arrives, release
                transcripts.append("open the report")
                time.sleep(0.5)
                results["state_before_ptt"] = hud_ref[0]._current_voice_state; results["states_tail"] = states[-6:]
                hud_ref[0].ptt_pressed.emit(); time.sleep(0.1)
                results["state_after_down"] = hud_ref[0]._current_voice_state
                time.sleep(0.8); cap.buffer.data = (rng.standard_normal(16000) * 0.1).astype(np.float32)
                hud_ref[0].ptt_released.emit(); time.sleep(1.5)
                results["e_queries"] = list(queries)
                time.sleep(2.2)  # mic heartbeat
            else:
                hud_ref[0].ptt_pressed.emit(); time.sleep(0.5); hud_ref[0].ptt_released.emit(); time.sleep(0.8)
                time.sleep(2.3)
        finally:
            QTimer.singleShot(0, app.quit) if False else app.quit_requested.emit()
    threading.Thread(target=run, daemon=True).start()

from PySide6.QtCore import QObject, Signal
class Q(QObject): quit_requested = Signal()
qo = Q(); app.quit_requested = qo.quit_requested; qo.quit_requested.connect(app.quit)
results = {}
QTimer.singleShot(600, scenario)

with patch("lora.main.LoraDesktopHUD", RecHUD), patch("lora.main.start_gateway_thread", return_value=None), \
     patch("lora.grounding.media_vector_db.MediaVectorDB", MagicMock()), \
     patch.object(capmod, "AudioCapture", FakeCapture), patch.object(sttmod, "FasterWhisperTranscriber", FakeTranscriber):
    lm.run_desktop_app(cfg_mgr, user_cfg, args)

print("RESULT:" + json.dumps({"queries": queries, "results": results, "mic": mic[-2:],
                         "messages": [m for m in msgs if m[0] != "You"]}))
