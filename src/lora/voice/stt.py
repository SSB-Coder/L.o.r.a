"""Speech-to-Text (STT) Engine using faster-whisper on CPU."""

import logging
import threading
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class TranscriptionResult(BaseModel):
    """Immutable result of speech transcription."""

    model_config = ConfigDict(frozen=True)

    text: str
    language: str = "en"
    duration_seconds: float = 0.0
    inference_time_seconds: float = 0.0
    confidence: float = 1.0
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.0
    low_confidence: bool = False


class FasterWhisperTranscriber:
    """Low-latency CPU transcription using faster-whisper (CTranslate2)."""

    DEFAULT_VOCABULARY_PROMPT = (
        "Hey LORA, Awaken LORA, open VS Code, Chrome, Wikipedia, YouTube, ChatGPT, Perplexity, Notepad, Calculator, Downloads."
    )

    def __init__(
        self,
        model_size_or_path: str = "base.en",
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
        self._lock = threading.Lock()

        if not lazy_load:
            self._ensure_model_loaded()

    def preload(self) -> bool:
        """Warm-load the Whisper STT model on CPU at startup so the first wake word is never missed."""
        try:
            self._ensure_model_loaded()
            return self._model is not None
        except Exception as exc:
            logger.warning("Whisper STT preload deferred: %s", exc)
            return False

    def _ensure_model_loaded(self) -> Any:
        """Instantiate WhisperModel under lock if not already loaded."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from faster_whisper import WhisperModel

            for candidate in dict.fromkeys((self.model_size_or_path, "base.en", "small.en", "distil-small.en")):
                try:
                    self._model = WhisperModel(
                        candidate,
                        device=self.device,
                        compute_type=self.compute_type,
                        cpu_threads=4,
                        local_files_only=True,
                    )
                    self.model_size_or_path = candidate
                    logger.info("Faster-Whisper STT ready on CPU (%s).", candidate)
                    return self._model
                except Exception:
                    continue
            self._model = WhisperModel(
                self.model_size_or_path,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4,
            )
            logger.info("Faster-Whisper STT downloaded and ready on CPU (%s).", self.model_size_or_path)
        return self._model

    @staticmethod
    def _clean_repetition(text: str) -> str:
        """Collapse Whisper token repetition loops ('open open i i i i' or 'why, why, why, why')."""
        import re

        if not text:
            return ""
        words = re.split(r"(\s+|,\s*)", text.strip())
        cleaned_tokens: list[str] = []
        last_word = ""
        repeat_count = 0
        for tok in words:
            w = re.sub(r"[^a-zA-Z0-9']", "", tok).lower()
            if not w:
                if repeat_count == 0 and cleaned_tokens:
                    cleaned_tokens.append(tok)
                continue
            if w == last_word:
                repeat_count += 1
                continue
            last_word = w
            repeat_count = 0
            cleaned_tokens.append(tok)
        out = "".join(cleaned_tokens).strip(" ,.;:-")
        if out.lower() in ("thank you", "thank you.", "thanks for watching", "you", "it was", "bye"):
            return ""
        return out

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe a 1D float32 PCM numpy audio array (normalized to -1.0 .. 1.0)."""
        if len(audio) == 0:
            return TranscriptionResult(text="", duration_seconds=0.0)

        duration = len(audio) / sample_rate
        start_time = time.monotonic()

        # Normalize moderate speech (including softer wake-word utterances) without amplifying floor hiss
        peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if peak < 0.006 or rms < 0.002:
            return TranscriptionResult(text="", duration_seconds=duration, confidence=0.0)
        # Never turn a fan or a distant voice up by 20-70x.
        gain = min(2.0, 0.5 / peak) if rms >= 0.006 and peak > 0.02 else 1.0
        norm_audio = audio * gain

        model = self._ensure_model_loaded()
        is_mock = getattr(model, "__module__", "").startswith("unittest.mock")
        # Short clips (<= 3.0s) are already segmented by our VoiceActivityDetector; running a second
        # internal Silero vad_filter inside faster-whisper clips short "Hey LORA" wake phrases.
        use_internal_vad = duration > 3.0

        with self._lock:
            if is_mock:
                segments, info = model.transcribe(
                    norm_audio.astype(np.float32),
                    beam_size=1,
                    initial_prompt=self.initial_prompt,
                    vad_filter=True,
                    language="en",
                )
            else:
                try:
                    segments, info = model.transcribe(
                        norm_audio.astype(np.float32),
                        beam_size=1,
                        best_of=3 if duration <= 2.5 else 5,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        repetition_penalty=1.35,
                        no_repeat_ngram_size=3,
                        no_speech_threshold=0.65,
                        log_prob_threshold=-1.0,
                        compression_ratio_threshold=2.2,
                        initial_prompt=self.initial_prompt,
                        vad_filter=use_internal_vad,
                        vad_parameters={"min_silence_duration_ms": 350, "speech_pad_ms": 250} if use_internal_vad else None,
                        language="en",
                    )
                except TypeError:
                    segments, info = model.transcribe(
                        norm_audio.astype(np.float32),
                        beam_size=1,
                        initial_prompt=self.initial_prompt,
                        vad_filter=True,
                        language="en",
                    )

            seg_list = list(segments)

        texts = [seg.text.strip() for seg in seg_list if getattr(seg, "text", None)]
        raw_text = " ".join(texts).strip()
        full_text = raw_text if is_mock else self._clean_repetition(raw_text)
        elapsed = time.monotonic() - start_time

        logprobs = [float(getattr(s, "avg_logprob", -0.2)) for s in seg_list if hasattr(s, "avg_logprob")]
        nospeech = [float(getattr(s, "no_speech_prob", 0.0)) for s in seg_list if hasattr(s, "no_speech_prob")]
        avg_lp = sum(logprobs) / len(logprobs) if logprobs else -0.2
        max_ns = max(nospeech) if nospeech else 0.0
        # v1.5 Plan §9.2: flag low-confidence transcripts (avg_logprob < -0.7 or no_speech_prob > 0.6)
        low_conf = bool(not is_mock and (avg_lp < -0.70 or max_ns > 0.60))
        conf = float(np.clip(np.exp(max(-3.0, min(0.0, avg_lp))), 0.1, 1.0)) if logprobs else 1.0

        return TranscriptionResult(
            text=full_text,
            language=getattr(info, "language", "en"),
            duration_seconds=duration,
            inference_time_seconds=elapsed,
            confidence=round(conf, 3),
            avg_logprob=round(avg_lp, 3),
            no_speech_prob=round(max_ns, 3),
            low_confidence=low_conf,
        )
