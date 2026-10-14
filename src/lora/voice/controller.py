"""Voice Pipeline Controller and HUD State Machine for L.O.R.A."""

import asyncio
import logging
import threading
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import numpy as np

from lora.core.controller import Controller
from lora.voice.capture import AudioCapture
from lora.voice.hotkey import PushToTalkListener
from lora.voice.stt import FasterWhisperTranscriber
from lora.voice.tts.engine import TtsEngine
from lora.voice.vad import SpeechState, VoiceActivityDetector
from lora.voice.wakeword import WakeWordDetector

logger = logging.getLogger("lora.voice")


class VoiceState(StrEnum):
    """Voice HUD state machine states."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    ACTING = "acting"
    SPEAKING = "speaking"


class VoiceController:
    """Orchestrates wake word detection, VAD, STT, core action execution, and TTS output."""

    def __init__(
        self,
        capture: AudioCapture | None = None,
        vad: VoiceActivityDetector | None = None,
        wakeword: WakeWordDetector | None = None,
        stt: FasterWhisperTranscriber | None = None,
        tts: TtsEngine | None = None,
        core_controller: Controller | None = None,
        hotkey_listener: PushToTalkListener | None = None,
        state_callback: Callable[[VoiceState], None] | None = None,
    ) -> None:
        self.capture = capture or AudioCapture()
        self.vad = vad or VoiceActivityDetector()
        self.wakeword = wakeword or WakeWordDetector()
        self.stt = stt or FasterWhisperTranscriber()
        self.tts = tts or TtsEngine()
        self.core_controller = core_controller or Controller()
        self.state_callback = state_callback

        self.state = VoiceState.IDLE
        self._running = False
        self._lock = threading.Lock()
        self._loop_task: asyncio.Task[None] | None = None

        # Setup Push-to-Talk
        if hotkey_listener is not None:
            self.hotkey_listener = hotkey_listener
            if self.hotkey_listener.on_activate is None:
                self.hotkey_listener.on_activate = self._on_ptt_activate
            if self.hotkey_listener.on_deactivate is None:
                self.hotkey_listener.on_deactivate = self._on_ptt_deactivate
        else:
            self.hotkey_listener = PushToTalkListener(
                on_activate=self._on_ptt_activate,
                on_deactivate=self._on_ptt_deactivate,
            )

        # Connect capture callback
        self.capture.register_chunk_callback(self._on_audio_chunk)
        self._ptt_active = False
        self._collected_audio_chunks: list[np.ndarray] = []

    def _set_state(self, new_state: VoiceState) -> None:
        """Update state and notify HUD observer under thread lock."""
        with self._lock:
            self.state = new_state
        if self.state_callback:
            try:
                self.state_callback(new_state)
            except Exception:
                pass

    def _on_ptt_activate(self) -> None:
        """Push-to-talk pressed: interrupt TTS and start listening immediately (unless thinking/acting)."""
        with self._lock:
            curr_state = self.state
        if curr_state in (VoiceState.THINKING, VoiceState.ACTING):
            return
        self._ptt_active = True
        self.tts.interrupt()
        with self._lock:
            self._collected_audio_chunks.clear()
            self.vad.reset()
        self._set_state(VoiceState.LISTENING)

    def _on_ptt_deactivate(self) -> None:
        """Push-to-talk released: end listening and trigger processing."""
        self._ptt_active = False
        with self._lock:
            curr_state = self.state
        if curr_state == VoiceState.LISTENING:
            self._set_state(VoiceState.THINKING)

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        """Receive continuous audio stream from capture buffer."""
        if not self._running:
            return

        with self._lock:
            curr_state = self.state

        # 1. When IDLE: check for wake word / chat activation
        if curr_state == VoiceState.IDLE:
            # Check last 1.5 seconds of audio in ring buffer
            recent_audio = self.capture.buffer.read_last_n_seconds(1.5)
            wake_res = self.wakeword.process_audio(recent_audio, current_state="idle")
            if wake_res.detected and wake_res.speaker_verified:
                logger.info(f"Wake word '{wake_res.phrase}' detected! Score: {wake_res.score:.2f}")
                self.tts.interrupt()
                with self._lock:
                    self._collected_audio_chunks.clear()
                    self.vad.reset()
                self._set_state(VoiceState.LISTENING)

        # 2. When LISTENING: accumulate speech frames and monitor VAD
        elif curr_state == VoiceState.LISTENING:
            with self._lock:
                self._collected_audio_chunks.append(chunk)
            speech_state, prob = self.vad.process_chunk(chunk)
            if speech_state == SpeechState.ENDED and not self._ptt_active:
                logger.info("VAD detected end of speech.")
                self._set_state(VoiceState.THINKING)

    async def process_voice_turn(self, audio_override: np.ndarray | None = None) -> dict[str, Any]:
        """Execute a full voice turn from captured or supplied utterance audio."""
        if audio_override is not None:
            utterance_audio = audio_override
        else:
            with self._lock:
                if self._collected_audio_chunks:
                    utterance_audio = np.concatenate(self._collected_audio_chunks)
                    self._collected_audio_chunks.clear()
                    self.vad.reset()
                else:
                    utterance_audio = np.zeros(0, dtype=np.float32)

        if len(utterance_audio) == 0:
            self._set_state(VoiceState.IDLE)
            return {"status": "empty", "message": "No audio captured"}

        # Step 1: STT Transcription
        self._set_state(VoiceState.THINKING)
        transcription = self.stt.transcribe(utterance_audio)
        user_text = transcription.text.strip()

        if not user_text:
            self._set_state(VoiceState.IDLE)
            return {"status": "empty", "message": "No speech detected in audio"}

        # Step 2: Action Execution
        self._set_state(VoiceState.ACTING)
        try:
            exec_res = await self.core_controller.aexecute_user_request(user_text)
        except Exception as e:
            exec_res = {"status": "error", "message": str(e)}

        # Format spoken summary
        spoken_response = ""
        if exec_res.get("status") == "success":
            steps = exec_res.get("steps", [])
            if steps:
                last_step = steps[-1]
                tool = last_step.get("tool", "")
                result = last_step.get("result", {})
                if tool == "open_app":
                    spoken_response = f"Launched {result.get('name', 'application')}."
                elif tool == "open_url":
                    spoken_response = "Opened webpage in your browser."
                elif tool == "web_search":
                    count = result.get("count", 0)
                    spoken_response = f"Found {count} sources on {result.get('query', 'topic')}."
                elif tool == "ask_other_ai":
                    svc = result.get("service", "assistant")
                    resp = result.get("response", "")
                    # Spoken summary of first sentence
                    first_sent = resp.split(".")[0] if resp else "Done."
                    spoken_response = f"{svc.title()} responded: {first_sent}."
                else:
                    spoken_response = f"Completed action {tool}."
            else:
                spoken_response = "Action executed."
        else:
            spoken_response = f"Sorry, could not complete that: {exec_res.get('message', 'unknown error')}."

        # Step 3: TTS Output
        self._set_state(VoiceState.SPEAKING)
        spoken_chunks = self.tts.synthesize_and_speak(spoken_response)

        self._set_state(VoiceState.IDLE)
        return {
            "status": "success",
            "transcription": transcription.model_dump(),
            "execution": exec_res,
            "spoken_response": spoken_response,
            "tts_chunks_count": len(spoken_chunks),
        }

    def start(self) -> None:
        """Start background microphone capture and push-to-talk listener."""
        self._running = True
        self.capture.start()
        self.hotkey_listener.start()
        self._set_state(VoiceState.IDLE)

    def stop(self) -> None:
        """Stop voice pipeline."""
        self._running = False
        self.capture.stop()
        self.hotkey_listener.stop()
        self.tts.interrupt()
        self._set_state(VoiceState.IDLE)
