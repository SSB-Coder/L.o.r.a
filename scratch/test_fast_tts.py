import time
import subprocess
import tempfile
from pathlib import Path
import wave
import numpy as np

def synthesize_fast(text: str, voice: str = "Microsoft Zira Desktop"):
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    
    clean_text = text.replace('"', '""').replace("`", "")
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        f"Add-Type -AssemblyName System.Speech; "
        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SelectVoice('{voice}'); "
        f"$s.SetOutputToWaveFile('{tmp_wav}'); "
        f"$s.Speak('{clean_text}'); "
        f"$s.Dispose()"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    with wave.open(tmp_wav, 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    Path(tmp_wav).unlink(missing_ok=True)
    t1 = time.perf_counter()
    print(f"Generated {len(samples)} samples at {sr} Hz in {t1 - t0:.3f} seconds!")
    return samples, sr

if __name__ == "__main__":
    synthesize_fast("Hello, I am LORA. Layer Oriented Research Assistant online and ready.")
