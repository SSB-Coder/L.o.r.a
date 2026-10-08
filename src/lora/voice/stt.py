"""Speech-to-Text (STT) Engine using faster-whisper on CPU."""

import time
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class TranscriptionResult(BaseModel):
    """Immutable result of speech transcription."""

    model_config = ConfigDict(frozen=True)

    text: str
    language: str = "en"
    duration_seconds: float = 0.0
    inference_time_seconds: float = 0.0
    confidence: float = 1.0


class FasterWhisperTranscriber:
    """Low-latency CPU transcription using faster-whisper (CTranslate2)."""

    DEFAULT_VOCABULARY_PROMPT = (
        "LORA, Awaken LORA, arXiv, RAG, Ollama, Python, VS Code, GitHub, Hugging Face, Gemma, Granite"
    )

    def __init__(
        self,
        model_size_or_path: str = "distil-small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        initial_prompt: str | None = None,
        lazy_load: bool = True,
    ) -> None:
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.initial_prompt = initial_prompt or self.DEFAULT_VOCABULARY_PROMPT
        self._model: Any = None

        if not lazy_load:
            self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> Any:
        """Instantiate WhisperModel if not already loaded."""
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size_or_path,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe a 1D float32 PCM numpy audio array (normalized to -1.0 .. 1.0)."""
        if len(audio) == 0:
            return TranscriptionResult(text="", duration_seconds=0.0)

        duration = len(audio) / sample_rate
        start_time = time.monotonic()

        model = self._ensure_model_loaded()

        # faster-whisper accepts float32 numpy arrays directly
        segments, info = model.transcribe(
            audio.astype(np.float32),
            beam_size=1,
            initial_prompt=self.initial_prompt,
            vad_filter=True,
            language="en",
        )

        texts = [seg.text.strip() for seg in segments if seg.text]
        full_text = " ".join(texts).strip()
        elapsed = time.monotonic() - start_time

        return TranscriptionResult(
            text=full_text,
            language=getattr(info, "language", "en"),
            duration_seconds=duration,
            inference_time_seconds=elapsed,
            confidence=1.0,
        )
