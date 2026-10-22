"""Controlled dependencies for real desktop lifecycle/controller integration.

Runs real Qt dispatch, runtime routing, HTTP parsing and turn coordination, with
scripted inference/audio and no native Chromium, microphone or private weights.
Socket transport and rendered WebGL remain separate acceptance measurements.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np


def configure(mode, voice=False):
    # Inherit the opt-in restricted-loop adapter in subprocesses as well.
    if os.environ.get("LORA_SANDBOX_TESTS") == "1":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from run_sandbox_tests import configure as configure_loop
        configure_loop()

    os.environ["LORA_SPEAKER_VERIFY_ENABLED"] = "0"
    os.environ["LORA_IN_PROCESS_SPEECH"] = "1"  # controlled fake backends, never native workers
    sys.modules["sounddevice"] = MagicMock()
    import lora.models.laya_router as router
    import lora.ui.hud as hud
    from lora.voice.tts.worker import TtsAudioChunk

    # These patches are subprocess-local and deliberately never affect startup.
    hud.WEB_ENGINE_AVAILABLE = False
    router.laya = None
    payloads = []

    class Tokens(httpx.AsyncByteStream):
        async def __aiter__(self):
            tokens = ["Right ", "away, ", "friend. ", "Anything else?"]
            if mode == "shutdownspeech":
                tokens = ["First sentence. Second sentence. Third sentence. "]
            elif voice:
                tokens = ["Sure ", "thing."]
            elif mode == "think":
                tokens.insert(0, "<think>hmm</think>")
            for token in tokens:
                await asyncio.sleep(.8 if mode in ("busy", "interrupt", "sleepbusy") else .08)
                yield (json.dumps({"message": {"content": token}, "done": False}) + "\n").encode()
            yield b'{"message":{"content":""},"done":true}\n'

    def respond(request):
        if mode == "down":
            raise httpx.ConnectError("scripted engine unavailable", request=request)
        if request.url.path.endswith("/api/tags"):
            return httpx.Response(200, json={"models": [{"name": "SparkLLM/Spark-X2.5-4B:latest"}]})
        if request.url.path.endswith("/api/generate"):
            return httpx.Response(200, json={"done": True})
        if request.url.path.endswith("/api/chat"):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, stream=Tokens())
        raise AssertionError(f"Unexpected network call: {request.method} {request.url}")

    original_client = httpx.AsyncClient
    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(respond)
        return original_client(*args, **kwargs)
    patch("httpx.AsyncClient", client).start()

    class SilentKokoro:
        def __init__(self, *args, **kwargs):
            pass
        def preload(self):
            return True
        def diagnosis(self):
            return {"ok": True, "summary": "Controlled Kokoro stub", "fix": ""}
        def synthesize(self, text, **kwargs):
            return TtsAudioChunk(job_id="test", text=text, samples=np.zeros(0, dtype=np.float32),
                                 sample_rate=24000, duration=0.)
        def shutdown(self):
            pass

    patch("lora.voice.tts.worker.KokoroTtsWorkerProcess", SilentKokoro).start()
    patch("lora.voice.hotkey.PushToTalkListener.start", return_value=None).start()
    patch("lora.voice.hotkey.PushToTalkListener.stop", return_value=None).start()
    patch("lora.ui.audio_cues.AudioCueManager.play_cue", return_value=None).start()
    if voice:
        patch("lora.voice.vad.VoiceActivityDetector.evaluate_chunk",
              lambda self, chunk: float(np.max(np.abs(chunk)) > .01)).start()
    return payloads
