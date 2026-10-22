import json
import sys
import time
from collections import deque
from pathlib import Path
import numpy as np
from lora.voice.capture import AudioBuffer
from lora.voice.vad import VoiceActivityDetector, SpeechState

rng=np.random.default_rng(12)
frames=[rng.normal(0,.01,512).astype(np.float32) for _ in range(1000)]
old=deque(maxlen=160000)
started=time.perf_counter()
for frame in frames:
    old.extend(frame.flatten().astype(np.float32).tolist())
old_ms=(time.perf_counter()-started)*1000
new=AudioBuffer()
started=time.perf_counter()
for frame in frames:
    new.write(frame)
new_ms=(time.perf_counter()-started)*1000
np.testing.assert_allclose(new.read_last_n_seconds(10),np.array(old,dtype=np.float32))
vad=VoiceActivityDetector(onnx_model_path='silero',threshold=.5)
t=np.arange(16000*8)/16000
noise=(.02*np.sin(2*np.pi*120*t)+rng.normal(0,.002,len(t))).astype(np.float32)
speech_frames=0
for start in range(0,len(noise)-511,512):
    state,_=vad.process_chunk(noise[start:start+512])
    speech_frames+=state!=SpeechState.SILENCE
    if state==SpeechState.ENDED:
        vad.get_and_reset_utterance()
result={'kind':'synthetic-fan-and-audio-ring-smoke','fan_seconds':8,'false_speech_frames':speech_frames,
        'vad_provider':vad._silero_vad.session.get_providers() if vad._silero_vad else [],
        'write_frames':1000,'old_ring_write_ms':round(old_ms,3),'numpy_ring_write_ms':round(new_ms,3),
        'old_ring_approx_bytes':sys.getsizeof(old)+len(old)*sys.getsizeof(0.0),'numpy_ring_bytes':new._buffer.nbytes,
        'limits':'Synthetic fan only; not a human-speaker accuracy or full application RAM/latency benchmark.'}
Path('docs/validation/audio-smoke.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
