import os
import sys
import numpy as np

# Ensure src in path
sys.path.insert(0, os.path.abspath("src"))

print("=== 1. TESTING STT (FasterWhisper) ===")
try:
    from lora.voice.stt import FasterWhisperTranscriber
    transcriber = FasterWhisperTranscriber(lazy_load=False)
    # Test on silence
    silence = np.zeros(16000, dtype=np.float32)
    res = transcriber.transcribe(silence)
    print("STT success! Result on silence:", repr(res.text))
except Exception as e:
    print("STT FAILED:", e)
    import traceback
    traceback.print_exc()

print("\n=== 2. TESTING TTS (Qwen3-TTS) ===")
try:
    from lora.voice.tts.worker import Qwen3TtsWorkerProcess
    tts = Qwen3TtsWorkerProcess(
        models_dir=r"C:\Models\Qwen3-TTS-12Hz-1.7B-CustomVoice",
        tokenizer_dir=r"C:\Models\Qwen3-TTS-Tokenizer-12Hz",
        default_voice="vivian",
        spawn_process=False,
    )
    chunk = tts.synthesize("Hello, testing speech.", voice="vivian")
    if chunk:
        print(f"TTS success! Generated {len(chunk.samples)} samples at {chunk.sample_rate} Hz (duration: {len(chunk.samples)/chunk.sample_rate:.2f}s)")
    else:
        print("TTS returned None!")
except Exception as e:
    print("TTS FAILED:", e)
    import traceback
    traceback.print_exc()

print("\n=== 3. TESTING WAKE WORD ===")
try:
    from lora.voice.wakeword import WakeWordDetector
    ww = WakeWordDetector()
    print("WakeWordDetector initialized successfully")
except Exception as e:
    print("Wake word FAILED:", e)
    import traceback
    traceback.print_exc()
