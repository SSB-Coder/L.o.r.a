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

from _controlled_desktop import configure
seen_payloads = configure(mode, voice=True)
url = "http://127.0.0.1:11434"

# Reproduce quitting while one native-style synthesis call is blocked and
# additional sentences are queued. Closing the backend must wake the call.
shutdown_entered, shutdown_closed = threading.Event(), threading.Event()
speech_calls, speech_streams, terminal_errors, shutdown_order = [], [], [], []
if mode == "shutdownspeech":
    import logging
    from lora.voice.isolation import SpeechWorkerClosedError
    from lora.voice.speech_stream import SpeechStreamer

    class RecordingStreamer(SpeechStreamer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            speech_streams.append(self)

    class ClosingKokoro:
        def __init__(self, *args, **kwargs): pass
        def preload(self): return True
        def diagnosis(self): return {"ok": True, "summary": "Controlled voice ready", "fix": ""}
        def synthesize(self, text, **kwargs):
            speech_calls.append(text)
            shutdown_entered.set()
            if not shutdown_closed.wait(15):
                raise RuntimeError("Test backend was never closed")
            raise SpeechWorkerClosedError("tts worker closed during synthesize")
        def shutdown(self):
            shutdown_order.append(all(stream._stop.is_set() for stream in speech_streams))
            shutdown_closed.set()

    class ErrorCollector(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR:
                terminal_errors.append(record.getMessage())

    logging.getLogger().addHandler(ErrorCollector())
    patch("lora.voice.tts.worker.KokoroTtsWorkerProcess", ClosingKokoro).start()
    patch("lora.voice.speech_stream.SpeechStreamer", RecordingStreamer).start()

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv)

import lora.main as lm
from lora.ui.hud import LoraDesktopHUD
import lora.voice.capture as capmod, lora.voice.stt as sttmod

transcripts = []          # scripted STT output, consumed in order
transcribed = []
transcribed_at = []
answered_at = []
class FakeTranscriber:
    def __init__(self, *a, **k): pass
    def preload(self): pass
    def transcribe(self, audio, sample_rate=16000):
        text = transcripts.pop(0) if transcripts else ""
        transcribed.append(text)
        transcribed_at.append(time.monotonic())
        return __import__("types").SimpleNamespace(text=text, confidence=1.0, avg_logprob=0.0, low_confidence=False)

class FakeBuffer:
    def __init__(self): self.data = np.zeros(0, dtype=np.float32)
    def clear(self): self.data = np.zeros(0, dtype=np.float32)
    def read_last_n_seconds(self, s): return self.data[-int(s * 16000):]

class FakeCapture:
    registry = []
    def __init__(self, *a, **k):
        self.cbs = []; self.buffer = FakeBuffer(); FakeCapture.registry.append(self)
        self.ok = (mode != "nomic"); self.device_name = "Fake USB Mic" if self.ok else ""
        self.starts = self.stops = 0
    def register_chunk_callback(self, cb): self.cbs.append(cb)
    def start(self):
        self.starts += 1
        return self.ok
    @property
    def is_streaming(self): return self.ok
    def seconds_since_audio(self): return 0.05 if self.ok else None
    def status_text(self): return self.device_name if self.ok else "no input device found (PortAudioError: none)"
    def stop(self): self.stops += 1
    def feed(self, chunk):
        for cb in self.cbs: cb(chunk)

rng = np.random.default_rng(1)
def speak(cap, seconds=0.7):
    for _ in range(int(seconds / 0.032)): cap.feed((rng.standard_normal(512) * 0.1).astype(np.float32)); time.sleep(0.002)
def silence(cap, seconds=1.0):
    for _ in range(int(seconds / 0.032)): cap.feed(np.zeros(512, dtype=np.float32)); time.sleep(0.002)
def utterance(cap): speak(cap); silence(cap)

def wait_for(predicate, description, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError(f"Timed out waiting for {description}; transcripts={transcribed}, queries={queries}, states={states}")

queries, msgs, mic, states = [], [], [], []
hud_ref = []
class RecHUD(LoraDesktopHUD):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); hud_ref.append(self)
    def append_message(self, s, t):
        msgs.append((s, t)); (queries.append(t) if s == 'You' else None); super().append_message(s, t)
    def set_mic_status(self, ok, text): mic.append((ok, text)); super().set_mic_status(ok, text)
    def set_voice_state(self, st): states.append(st); super().set_voice_state(st)
    def present_answer(self, *args, **kwargs):
        answered_at.append(time.monotonic())
        super().present_answer(*args, **kwargs)

tmp = Path(tempfile.mkdtemp()); os.chdir(tmp)       # no speaker profile in cwd
cfg_mgr, user_cfg = lm.init_platform(base_dir=tmp)
user_cfg = user_cfg.model_copy(update={"models": user_cfg.models.model_copy(update={"ollama_url": url})})
args = argparse.Namespace(no_audio=False, headless=False, port=0, host="127.0.0.1", offline=False, config_dir=tmp)

def scenario():
    cap = FakeCapture.registry[-1]
    def run():
        try:
            if mode == "shutdownspeech":
                hud_ref[0].query_submitted.emit("tell me a joke", "")
                wait_for(shutdown_entered.is_set, "synthesis in flight during exit")
                wait_for(lambda: len(answered_at) == 1, "remaining sentences queued before exit")
                results["synthesis_was_pending"] = True
            elif mode == "sleepwake":
                hud_ref[0].sleep_toggled.emit(True)
                wait_for(lambda: states[-1:] == ["sleeping"], "sleep mode")
                time.sleep(.7)
                transcripts.extend(["background conversation", "Hey Laura tell me a joke"])
                utterance(cap)
                wait_for(lambda: len(transcribed) == 1, "sleep chatter classification")
                results["sleep_chatter_ignored"] = queries == []
                utterance(cap)
                wait_for(lambda: queries == ["tell me a joke"], "addressed voice wake")
                results["capture_stops_before_exit"] = cap.stops
            elif mode == "busy":
                transcripts.extend(["Hey Laura tell me a joke", "Hey Lora how are you"])
                utterance(cap)
                wait_for(lambda: len(queries) == 1, "first busy query")
                time.sleep(.45)  # wake-cue echo window, while the scripted model is still executing
                utterance(cap)
                time.sleep(.15)
                results["no_stt_during_execution"] = len(transcribed) == 1
                wait_for(lambda: len(queries) == 2 and len(answered_at) == 2, "deferred query execution")
                results["deferred_after_answer"] = transcribed_at[1] >= answered_at[0]
                results["capture_starts"] = cap.starts
                results["capture_stops_before_exit"] = cap.stops
            elif mode == "voice":
                transcripts.extend(["we should get lunch at noon", "Hey LORA what's the plan", "hey laura", "tell me a joke"])
                utterance(cap)   # (a) ordinary talk -> must be ignored
                wait_for(lambda: len(transcribed) == 1, "background chatter classification")
                time.sleep(.1)
                results["a_ignored"] = list(queries) == []
                utterance(cap)   # (b) wake phrase + query in one go
                wait_for(lambda: len(queries) == 1 and states[-1:] == ["idle"], "inline wake query and completed reply")
                results["b_query"] = list(queries)
                time.sleep(.5)  # post-playback microphone echo cooldown
                utterance(cap)   # (c) wake phrase alone -> armed
                wait_for(lambda: states[-1:] == ["listening"], "armed follow-up listening")
                time.sleep(.4)  # wake cue echo cooldown
                results["c_state_armed"] = states[-1] == "listening"
                utterance(cap)   # (d) next utterance becomes the query
                wait_for(lambda: len(queries) == 2 and states[-1:] == ["idle"], "follow-up query and completed reply")
                results["d_queries"] = list(queries)
                # (e) push-to-talk: hold, audio arrives, release
                transcripts.append("open the report")
                time.sleep(0.5)
                results["state_before_ptt"] = hud_ref[0]._current_voice_state; results["states_tail"] = states[-6:]
                hud_ref[0].ptt_pressed.emit(); time.sleep(0.1)
                results["state_after_down"] = hud_ref[0]._current_voice_state
                time.sleep(0.8); cap.buffer.data = (rng.standard_normal(16000) * 0.1).astype(np.float32)
                hud_ref[0].ptt_released.emit()
                wait_for(lambda: len(queries) == 3, "push-to-talk delivery")
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
QTimer.singleShot(0, lambda: QTimer.singleShot(1100, scenario))  # delay after startup finishes

with patch("lora.main.LoraDesktopHUD", RecHUD), patch("lora.main.start_gateway_thread", return_value=None), \
     patch("lora.grounding.media_vector_db.MediaVectorDB", MagicMock(return_value=MagicMock(file_count=0))), \
     patch.object(capmod, "AudioCapture", FakeCapture), patch.object(sttmod, "FasterWhisperTranscriber", FakeTranscriber):
    lm.run_desktop_app(cfg_mgr, user_cfg, args)

if mode == "shutdownspeech":
    results.update({"speech_calls": len(speech_calls), "terminal_errors": terminal_errors,
                    "speech_stopped_before_worker_closed": shutdown_order == [True],
                    "speech_threads_joined": all(not thread.is_alive() for stream in speech_streams for thread in stream._threads)})

print("RESULT:" + json.dumps({"queries": queries, "results": results, "mic": mic[-2:],
                         "messages": [m for m in msgs if m[0] != "You"]}))
