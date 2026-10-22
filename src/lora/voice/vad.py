"""Voice Activity Detection (VAD) and endpointing for L.O.R.A Voice Pipeline."""

import logging
from collections import deque
from enum import StrEnum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SpeechState(StrEnum):
    """Current state of voice activity."""

    SILENCE = "silence"
    SPEAKING = "speaking"
    ENDED = "ended"


class VoiceActivityDetector:
    """Detects speech boundaries and pauses to segment user voice commands cleanly."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.30,
        silence_duration_seconds: float = 0.6,
        min_speech_duration_seconds: float = 0.16,
        onnx_model_path: str | None = None,
        preroll_chunks: int = 16,  # ~512ms pre-roll at 512 samples / 16kHz (v1.5 Plan)
        max_utterance_seconds: float = 20.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_duration_seconds = silence_duration_seconds
        self.min_speech_duration_seconds = min_speech_duration_seconds
        self.state = SpeechState.SILENCE
        self.max_samples = int(max(1.0, max_utterance_seconds) * sample_rate)
        self._buffered_samples = 0
        self.noise_floor = 0.002
        self._warned_fallback = False
        self._silero_h = np.zeros((1, 1, 128), dtype=np.float32)
        self._silero_c = np.zeros((1, 1, 128), dtype=np.float32)
        self._silero_context = np.zeros((1, 64), dtype=np.float32)

        self._session: Any = None
        self._silero_vad: Any = None
        if onnx_model_path:
            if str(onnx_model_path).lower() in ("silero", "builtin", "faster-whisper", "auto"):
                try:
                    from faster_whisper.vad import get_vad_model
                    self._silero_vad = get_vad_model()
                except Exception:
                    self._silero_vad = None
            else:
                try:
                    import onnxruntime as ort
                    self._session = ort.InferenceSession(onnx_model_path)
                except Exception:
                    self._session = None

        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._total_speech_samples = 0
        self._recorded_utterance_chunks: list[np.ndarray] = []
        # 500ms pre-roll ring buffer before speech onset so the first syllable ("Hey") is never clipped.
        self._preroll: deque[np.ndarray] = deque(maxlen=max(0, preroll_chunks))

    def compute_energy_prob(self, chunk: np.ndarray) -> float:
        """Heuristic fallback RMS energy-based voice activity score (0.0 to 1.0).

        Calibrated for standard laptop microphones (e.g., Realtek Audio):
        - Ambient silence: ~0.001 - 0.008 RMS -> score 0.0
        - Conversational speech: ~0.015 - 0.035 RMS -> score ~0.30 - 0.95
        - Assertive / close speech: >= 0.040 RMS -> score 1.0
        """
        if len(chunk) == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(chunk**2)))
        scaled = (rms - 0.008) / 0.024
        return float(np.clip(scaled, 0.0, 1.0))

    def evaluate_chunk(self, chunk: np.ndarray) -> float:
        """Calculate speech probability for a single audio frame (typically 512 samples / 32ms)."""
        if self._silero_vad is not None and len(chunk) == 512:
            try:
                # Calling SileroVADModel.__call__ on each 32ms frame resets its
                # recurrent state. Keep per-stream state and acoustic context.
                audio = chunk.astype(np.float32).reshape(1, -1)
                inputs = np.concatenate((self._silero_context, audio), axis=1)
                probs, self._silero_h, self._silero_c = self._silero_vad.session.run(
                    None, {"input": inputs, "h": self._silero_h, "c": self._silero_c}
                )
                self._silero_context = audio[:, -64:].copy()
                prob = float(np.asarray(probs).reshape(-1)[0])
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if prob < 0.15:
                    self.noise_floor = 0.98 * self.noise_floor + 0.02 * min(rms, 0.03)
                # Fan RMS can exceed a soft voice's RMS. Do not veto confident
                # neural speech merely because the fan raised the noise floor.
                return prob if rms >= 0.0001 else 0.0
            except Exception as exc:
                if not self._warned_fallback:
                    logger.warning("Streaming VAD failed; energy fallback is less selective: %s", exc)
                    self._warned_fallback = True
        if self._session is not None:
            try:
                # Prepare tensor for Silero VAD ONNX: shape (1, len(chunk))
                inp = chunk.astype(np.float32).reshape(1, -1)
                sr_tensor = np.array([self.sample_rate], dtype=np.int64)
                outputs = self._session.run(None, {"input": inp, "sr": sr_tensor})
                return float(outputs[0][0][0])
            except Exception:
                pass
        if self._silero_vad is None and self._session is None and not self._warned_fallback:
            logger.info("VAD uses energy fallback; neural noise rejection is unavailable")
            self._warned_fallback = True
        return self.compute_energy_prob(chunk)

    def _record(self, chunk: np.ndarray) -> None:
        self._recorded_utterance_chunks.append(chunk)
        self._buffered_samples += len(chunk)

    def process_chunk(self, chunk: np.ndarray) -> tuple[SpeechState, float]:
        """Process incoming audio chunk, track speech lifecycle, and return (state, speech_prob)."""
        prob = self.evaluate_chunk(chunk)
        chunk_duration = len(chunk) / self.sample_rate

        if prob >= self.threshold:
            self._consecutive_speech_frames += 1
            self._consecutive_silence_frames = 0
            self._total_speech_samples += len(chunk)
            if self.state == SpeechState.SILENCE and not self._recorded_utterance_chunks:
                self._recorded_utterance_chunks.extend(self._preroll)
                self._buffered_samples += sum(len(c) for c in self._preroll)
                self._preroll.clear()
            self._record(chunk)

            # Check if speech just started
            if self.state == SpeechState.SILENCE:
                if (self._consecutive_speech_frames * chunk_duration) >= self.min_speech_duration_seconds:
                    self.state = SpeechState.SPEAKING
        else:
            self._consecutive_silence_frames += 1
            if self.state == SpeechState.SILENCE:
                # If we already had >= 2 consecutive speech frames (e.g. "Hey" before "LORA"),
                # tolerate a brief inter-syllable dip (up to 6 frames / ~192ms) without wiping the onset buffer.
                # Isolated 1-frame clicks (_consecutive_speech_frames == 1) reset immediately.
                if self._consecutive_speech_frames >= 2 and self._consecutive_silence_frames <= 6:
                    self._record(chunk)
                else:
                    if self._recorded_utterance_chunks:
                        for c in self._recorded_utterance_chunks:
                            self._preroll.append(c)
                        self._recorded_utterance_chunks.clear()
                        self._buffered_samples = 0
                    self._consecutive_speech_frames = 0
                    self._total_speech_samples = 0
                    self._preroll.append(chunk)
            elif self.state == SpeechState.SPEAKING:
                self._record(chunk)
                silence_elapsed = self._consecutive_silence_frames * chunk_duration
                if silence_elapsed >= self.silence_duration_seconds:
                    self.state = SpeechState.ENDED

        if self._buffered_samples >= self.max_samples:
            self.state = SpeechState.ENDED
        return self.state, prob

    def get_and_reset_utterance(self) -> np.ndarray:
        """Retrieve the recorded utterance audio and reset the detector to SILENCE."""
        if self._recorded_utterance_chunks:
            audio = np.concatenate(self._recorded_utterance_chunks)
        else:
            audio = np.zeros(0, dtype=np.float32)

        self.reset()
        return audio

    def reset(self) -> None:
        """Reset state tracking back to SILENCE."""
        self.state = SpeechState.SILENCE
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._total_speech_samples = 0
        self._recorded_utterance_chunks.clear()
        self._preroll.clear()
        self._buffered_samples = 0
        self._silero_h.fill(0)
        self._silero_c.fill(0)
        self._silero_context.fill(0)
