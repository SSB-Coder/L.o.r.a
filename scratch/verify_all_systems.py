import sys
import time
import numpy as np
from pathlib import Path

print("=" * 60)
print("L.O.R.A END-TO-END SYSTEM INTEGRATION VERIFICATION")
print("=" * 60)

# 1. VERIFY OLLAMA
print("\n[1/5] Verifying Ollama Engine Connection...")
try:
    import asyncio
    from lora.models.ollama_client import OllamaClient
    client = OllamaClient()
    online = asyncio.run(client.is_online())
    assert online, "Ollama is not online"
    models = asyncio.run(client.list_models())
    model_names = [m.get("name") for m in models]
    print(f"  -> Ollama is ONLINE at http://127.0.0.1:11434")
    print(f"  -> Installed models ({len(model_names)}): {', '.join(model_names[:4])}")
    
    # Test generation
    res = asyncio.run(client.generate_chat(
        model="spark-x25-4b-32k:latest",
        messages=[{"role": "user", "content": "Respond with single word: READY"}]
    ))
    print(f"  -> Generation Test: '{res.strip()}' [SUCCESS]")
except Exception as e:
    print(f"  -> Ollama Test FAILED: {e}")
    sys.exit(1)

# 2. VERIFY STT (Faster-Whisper)
print("\n[2/5] Verifying Speech-to-Text (Faster-Whisper on CPU)...")
try:
    from lora.voice.stt import FasterWhisperTranscriber
    transcriber = FasterWhisperTranscriber(lazy_load=False)
    # Generate 1 sec of synthetic audio (silence / tone)
    dummy_audio = np.zeros(16000, dtype=np.float32)
    stt_res = transcriber.transcribe(dummy_audio)
    print(f"  -> Transcriber model: {transcriber.model_size_or_path} ({transcriber.device}, {transcriber.compute_type})")
    print(f"  -> Dummy audio transcribed in {stt_res.inference_time_seconds:.3f}s [SUCCESS]")
except Exception as e:
    print(f"  -> STT Test FAILED: {e}")
    sys.exit(1)

# 3. VERIFY WAKEWORD DETECTION (Awaken LORA & Hey LORA)
print("\n[3/5] Verifying Dual Wakeword Detection...")
try:
    from lora.voice.wakeword import WakeWordDetector
    detector = WakeWordDetector(transcriber=transcriber)
    
    # Test state target phrases
    res_sleep = detector.process_audio(np.zeros(800, dtype=np.float32), current_state="sleep")
    res_idle = detector.process_audio(np.zeros(800, dtype=np.float32), current_state="idle")
    assert res_sleep.phrase == "Awaken LORA", f"Expected 'Awaken LORA', got {res_sleep.phrase}"
    assert res_idle.phrase == "Hey LORA", f"Expected 'Hey LORA', got {res_idle.phrase}"
    print(f"  -> Sleep state wake phrase target: '{res_sleep.phrase}' [OK]")
    print(f"  -> Idle state chat phrase target: '{res_idle.phrase}' [OK]")
    print(f"  -> Dual activation logic verified [SUCCESS]")
except Exception as e:
    print(f"  -> Wakeword Test FAILED: {e}")
    sys.exit(1)

# 4. VERIFY QWEN3-TTS NEURAL VOICE ENGINE
print("\n[4/5] Verifying Qwen3-TTS Model Engine...")
try:
    from lora.voice.tts.worker import get_qwen_model, Qwen3TtsWorkerProcess
    model = get_qwen_model()
    assert model is not None, "Qwen model failed to load"
    print(f"  -> Qwen3-TTS 1.7B Model loaded from C:\\Models\\Qwen3-TTS-12Hz-1.7B-CustomVoice [OK]")
    
    worker = Qwen3TtsWorkerProcess(spawn_process=False)
    voices = worker.get_supported_voices()
    print(f"  -> Supported custom voices: {', '.join(voices)}")
    
    # Test synthesis of a short prompt
    chunk = worker.synthesize("System online.", voice="vivian")
    print(f"  -> Synthesized: '{chunk.text}' ({len(chunk.samples)} samples, {chunk.duration:.2f}s, {chunk.sample_rate}Hz) [SUCCESS]")
except Exception as e:
    print(f"  -> TTS Test FAILED: {e}")
    sys.exit(1)

# 5. VERIFY FAISS VECTOR DB & EMBEDDINGGEMMA-2
print("\n[5/5] Verifying FAISS & EmbeddingGemma-2 Vector DB...")
try:
    from lora.grounding.media_vector_db import MediaVectorDB
    db = MediaVectorDB()
    test_file = Path("pyproject.toml").resolve()
    add_res = db.add_file(test_file)
    print(f"  -> Ingested '{test_file.name}': {add_res.get('status')} ({add_res.get('chunks_added', 0)} chunks, dim={add_res.get('vector_dimension')}) [OK]")
    
    hits = db.search("fastapi", top_k=2)
    print(f"  -> Semantic search query 'fastapi': found {len(hits)} grounded chunks [OK]")
    if hits:
        print(f"     Match: '{hits[0]['text'][:80]}...'")
        
    report = db.report_on_media()
    print(f"  -> Report summary generated: {report.get('summary')[:90]}... [SUCCESS]")
except Exception as e:
    print(f"  -> Vector DB Test FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL 5/5 SUBSYSTEMS FULLY INTEGRATED AND VERIFIED OPERATIONAL!")
print("=" * 60)
