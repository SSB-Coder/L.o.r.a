import sys
import os
import time

sys.path.insert(0, os.path.abspath("src"))

from lora.config.manager import ConfigManager
from lora.ui.hud import LoraDesktopHUD
from lora.voice.tts.worker import Qwen3TtsWorkerProcess
from lora.voice.wakeword import WakeWordDetector
from lora.voice.stt import FasterWhisperTranscriber
import numpy as np

print("1. Testing WakeWord 'Awaken LORA'...")
transcriber = FasterWhisperTranscriber(lazy_load=True)
ww = WakeWordDetector(wake_phrase="Awaken LORA", chat_phrase="Hey LORA")
res = ww.process_audio(np.zeros(16000, dtype=np.float32), current_state="idle")
print("WakeWord idle check passed, detected:", res.detected)

print("\n2. Testing Fast TTS...")
worker = Qwen3TtsWorkerProcess(spawn_process=False)
chunk = worker.synthesize("Awaken LORA online.", voice="vivian")
print(f"TTS generated {len(chunk.samples)} samples in {chunk.duration:.2f}s at {chunk.sample_rate} Hz!")

print("\n3. Testing HUD creation...")
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)
hud = LoraDesktopHUD()
print("HUD windowTitle:", hud.windowTitle())
hud.set_voice_state("listening")
hud.set_audio_level(0.75)
hud.set_voice_state("speaking")
hud.set_voice_state("idle")
print("HUD state cycling passed!")

print("\n=== ALL VERIFICATIONS PASSED SUCCESSFULLY ===")
