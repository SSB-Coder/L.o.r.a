"""Wake Word Detection ('Awaken LORA') and Personalized Speaker Verification."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("lora.voice.wakeword")


class WakeWordResult(BaseModel):
    """Immutable result of wake word evaluation."""

    model_config = ConfigDict(frozen=True)

    detected: bool
    phrase: str = "Awaken LORA"
    score: float = 0.0
    speaker_verified: bool = True
    speaker_similarity: float = 1.0
    extracted_query: str = ""


class SpeakerVerifier:
    """Verifies that the speaker's vocal acoustics match the enrolled owner."""

    def __init__(
        self,
        profile_path: Path | str | None = None,
        threshold: float = 0.55,
        enabled: bool | None = None,
    ) -> None:
        env_thresh = os.environ.get("LORA_SPEAKER_VERIFY_THRESHOLD")
        self.threshold = float(env_thresh) if env_thresh else threshold
        self.profile_path = Path(profile_path) if profile_path else Path("data/voice/user_verifier.json")
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = os.environ.get("LORA_SPEAKER_VERIFY_ENABLED", "1").lower() not in ("0", "false", "no")
        self.reference_embeddings: list[list[float]] = []
        self._load_profile()

    def _load_profile(self) -> None:
        """Load enrolled speaker embeddings if profile file exists."""
        if self.profile_path.exists():
            try:
                data = json.loads(self.profile_path.read_text(encoding="utf-8"))
                self.reference_embeddings = data.get("embeddings", [])
                if "threshold" in data and "LORA_SPEAKER_VERIFY_THRESHOLD" not in os.environ:
                    file_thresh = float(data["threshold"])
                    self.threshold = min(self.threshold, file_thresh) if file_thresh > 0 else self.threshold
            except Exception:
                self.reference_embeddings = []

    def save_profile(self, embeddings: list[list[float]], speaker_name: str = "owner") -> None:
        """Save newly enrolled reference embeddings."""
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "speaker": speaker_name,
            "embeddings": embeddings,
            "threshold": self.threshold,
        }
        self.profile_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.reference_embeddings = embeddings

    @staticmethod
    def extract_features(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract a normalized acoustic spectral signature from audio."""
        if len(audio) < 512:
            return np.zeros(32, dtype=np.float32)

        # Slice the highest-energy 1.0s window rather than trailing post-speech silence
        chunk_len = min(len(audio), sample_rate)
        if len(audio) > chunk_len:
            step = max(256, sample_rate // 10)  # 100ms stride
            best_idx = 0
            best_energy = -1.0
            for start in range(0, len(audio) - chunk_len + 1, step):
                w = audio[start : start + chunk_len]
                energy = float(np.mean(w**2))
                if energy > best_energy:
                    best_energy = energy
                    best_idx = start
            chunk = audio[best_idx : best_idx + chunk_len]
        else:
            chunk = audio

        fft_vals = np.abs(np.fft.rfft(chunk))
        num_bins = 32
        bin_size = max(1, len(fft_vals) // num_bins)

        features = np.zeros(num_bins, dtype=np.float32)
        for i in range(num_bins):
            start = i * bin_size
            end = min(len(fft_vals), (i + 1) * bin_size)
            if start < end:
                features[i] = float(np.mean(fft_vals[start:end]))

        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features

    def verify(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[bool, float]:
        """Verify if the audio matches enrolled owner voice (returns is_verified, similarity)."""
        if not self.enabled or not self.reference_embeddings:
            # If verifier is disabled or no enrolled voice profile exists yet, allow with default baseline
            return True, 1.0

        feat = self.extract_features(audio, sample_rate)
        similarities = []

        for ref in self.reference_embeddings:
            ref_arr = np.array(ref, dtype=np.float32)
            dot = float(np.dot(feat, ref_arr))
            ref_norm = np.linalg.norm(ref_arr)
            feat_norm = np.linalg.norm(feat)
            if ref_norm > 0 and feat_norm > 0:
                sim = dot / (ref_norm * feat_norm)
                similarities.append(float(sim))

        if not similarities:
            return True, 1.0

        max_sim = max(similarities)
        is_owner = max_sim >= self.threshold
        return is_owner, max_sim


class WakeWordDetector:
    """Detects 'Awaken LORA' (wake from sleep) and 'Hey LORA' (chat from idle).

    Features:
    - Dual activation states:
      1. From SLEEP: 'Awaken LORA' (or 'Wake up LORA') wakes the system into IDLE.
      2. From IDLE: 'Hey LORA' activates listening to chat.
    - Real-time keyword transcription verification via FasterWhisperTranscriber.
    - Phonetic tolerance for common Whisper near-homophones of 'LORA'.
    - Integrated speaker acoustic verification.
    """

    def __init__(
        self,
        wake_phrase: str = "Awaken LORA",
        chat_phrase: str = "Hey LORA",
        threshold: float = 0.65,
        model_path: str | None = None,
        speaker_verifier: SpeakerVerifier | None = None,
        transcriber: Any = None,
    ) -> None:
        self.wake_phrase = wake_phrase
        self.chat_phrase = chat_phrase
        self.threshold = threshold
        self.speaker_verifier = speaker_verifier or SpeakerVerifier()
        self.transcriber = transcriber
        self._oww_model: Any = None

        resolved_model = model_path
        if resolved_model is None and Path("models/voice/hey_lora.onnx").is_file():
            resolved_model = "models/voice/hey_lora.onnx"
        if resolved_model:
            try:
                from openwakeword.model import Model
                self._oww_model = Model(wakeword_models=[resolved_model])
            except Exception:
                self._oww_model = None

    SLEEP_KEYWORDS = (
        "awaken lora", "awaken laura", "awaken lola", "awaken lara", "awaken nora", "awaken flora",
        "awaken lauren", "awaken lyra", "awaken aura",
        "wake up lora", "wake up laura", "wake up lola", "wake up lara", "wake lora", "wake laura",
        "awaken", "wake up",
    )
    IDLE_KEYWORDS = (
        "hey lora", "hey laura", "hey lola", "hey lara", "hey nora", "hey dora", "hey flora", "hey clara",
        "hey lauren", "hey lyra", "hey aura", "hey lotta", "hey rory", "hey lore", "hey lord", "hey lorra",
        "hi lora", "hi laura", "hi lola", "hi lara", "hi nora", "hello lora", "hello laura", "hello lola",
        "awaken lora", "awaken laura", "ok lora", "okay lora", "ok laura", "okay laura",
        "yo lora", "yo laura", "halo lora", "halo laura", "a lora", "a laura",
        "lora", "laura", "lorra",
    )
    _SLEEP_PHONETIC_RE = re.compile(
        r"^\s*(?:awaken|wake(?:\s+up)?)[,.\s]+"
        r"(?:lora|laura|lorra|lola|lara|nora|dora|flora|clara|lauren|lyra|aura)\b"
        r"[,.:;!?\s]*(?P<rest>.*)$",
        re.IGNORECASE,
    )
    _IDLE_PHONETIC_RE = re.compile(
        r"^\s*(?:(?:hey|hi|hello|yo|ok|okay|awaken|wake(?:\s+up)?|halo|aye)[,.\s]+)?"
        r"(?:lora|laura|lorra|lola|lara|nora|dora|flora|clara|lauren|lyra|aura|lotta|lore)\b"
        r"[,.:;!?\s]*(?P<rest>.*)$",
        re.IGNORECASE,
    )

    def match_text(self, text: str, is_sleeping: bool) -> tuple[bool, str]:
        """Check an already-transcribed utterance for the activation phrase.

        Returns (matched, query_after_the_phrase). Lets callers transcribe once
        instead of once for the wake check and again for the query.
        """
        raw_clean = re.sub(r"\s+", " ", (text or "").strip())
        txt = raw_clean.lower()
        if not txt:
            return False, ""
        # Normalize punctuation between prefix ("Hey,") and name ("Lora")
        txt_norm = re.sub(r"[,.;:!?]+", " ", txt)
        txt_norm = re.sub(r"\s+", " ", txt_norm).strip()
        if is_sleeping:
            ms = self._SLEEP_PHONETIC_RE.match(raw_clean)
            if ms:
                return True, (ms.group("rest") or "").strip(" ,.:;!?")
            for kw in self.SLEEP_KEYWORDS:
                if kw in txt_norm:
                    idx = txt_norm.find(kw)
                    before_ok = idx == 0 or not txt_norm[idx - 1].isalnum()
                    after_idx = idx + len(kw)
                    after_ok = after_idx >= len(txt_norm) or not txt_norm[after_idx].isalnum()
                    if before_ok and after_ok:
                        after = txt_norm[after_idx:].strip(" ,.:;!?")
                        # Bare 'awaken' or 'wake up' without 'lora' only triggers if standalone or followed by <4 words
                        if kw in ("awaken", "wake up") and after and len(after.split()) > 4:
                            continue
                        return True, after
            return False, ""

        m = self._IDLE_PHONETIC_RE.match(raw_clean)
        if m:
            return True, (m.group("rest") or "").strip(" ,.:;!?")

        for kw in self.IDLE_KEYWORDS:
            if kw in txt_norm:
                idx = txt_norm.find(kw)
                # Ensure word-boundary match so "floral" doesn't match "flora"
                before_ok = idx == 0 or not txt_norm[idx - 1].isalnum()
                after_idx = idx + len(kw)
                after_ok = after_idx >= len(txt_norm) or not txt_norm[after_idx].isalnum()
                if before_ok and after_ok:
                    after = txt_norm[after_idx:].strip(" ,.:;!?")
                    return True, after
        return False, ""

    def evaluate_text(
        self,
        text: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
        current_state: str = "idle",
    ) -> WakeWordResult:
        """Like process_audio, but for text that was already transcribed."""
        is_sleeping = current_state.lower() in ("sleep", "sleeping")
        target = self.wake_phrase if is_sleeping else self.chat_phrase
        matched, query = self.match_text(text, is_sleeping)
        if not matched:
            return WakeWordResult(detected=False, phrase=target, score=0.0)
        is_verified, sim = self.speaker_verifier.verify(audio, sample_rate)
        return WakeWordResult(
            detected=is_verified,
            phrase=target,
            score=0.95,
            speaker_verified=is_verified,
            speaker_similarity=sim,
            extracted_query=query,
        )

    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        current_state: str = "idle",
    ) -> WakeWordResult:
        """Evaluate audio buffer for activation phrase matching current state."""
        if len(audio) < 1600:  # less than 100ms
            target = self.wake_phrase if current_state.lower() in ("sleep", "sleeping") else self.chat_phrase
            return WakeWordResult(detected=False, phrase=target, score=0.0)

        is_sleeping = current_state.lower() in ("sleep", "sleeping")
        target_phrase = self.wake_phrase if is_sleeping else self.chat_phrase
        detected = False
        wake_score = 0.0
        extracted_query = ""

        # 1. Live Faster-Whisper keyword transcription pass if transcriber present
        if self.transcriber is not None:
            try:
                res = self.transcriber.transcribe(audio, sample_rate=sample_rate)
                matched, query = self.match_text(res.text, is_sleeping)
                if matched:
                    detected = True
                    wake_score = 0.95
                    target_phrase = self.wake_phrase if is_sleeping else self.chat_phrase
                    extracted_query = query
            except Exception:
                pass

        # 2. openWakeWord model if configured
        if not detected and self._oww_model is not None:
            try:
                pcm_int16 = (audio * 32767).astype(np.int16)
                prediction = self._oww_model.predict(pcm_int16)
                for k, score in prediction.items():
                    if score > wake_score:
                        wake_score = float(score)
                detected = wake_score >= self.threshold
            except Exception:
                wake_score = 0.0

        # 3. Energy & Acoustic envelope check fallback (supports unit tests and energy triggers)
        if not detected and self.transcriber is None and self._oww_model is None:
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms > 0.03:
                wake_score = float(min(1.0, rms * 10))
            detected = wake_score >= self.threshold

        if detected:
            is_verified, sim = self.speaker_verifier.verify(audio, sample_rate)
            return WakeWordResult(
                detected=is_verified,
                phrase=target_phrase,
                score=wake_score,
                speaker_verified=is_verified,
                speaker_similarity=sim,
                extracted_query=extracted_query,
            )

        return WakeWordResult(
            detected=False,
            phrase=target_phrase,
            score=wake_score,
            speaker_verified=True,
            speaker_similarity=0.0,
            extracted_query="",
        )
