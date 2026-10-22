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
        self.profile_path = Path(profile_path).expanduser() if profile_path else Path.home() / ".lora" / "voice" / "user_verifier.json"
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = os.environ.get("LORA_SPEAKER_VERIFY_ENABLED", "1").lower() not in ("0", "false", "no")
        self.reference_embeddings: list[list[float]] = []
        self.loaded_profile_path: Path | None = None
        self._load_profile()

    def _load_profile(self) -> None:
        """Load enrolled speaker embeddings if profile file exists."""
        profile = self.profile_path
        canonical = Path.home() / ".lora" / "voice" / "user_verifier.json"
        if profile == canonical and not profile.exists():
            # Read the existing enrollment in place. Never overwrite a user's
            # profile or silently require them to enroll again after an upgrade.
            legacy = Path(__file__).resolve().parents[3] / "data" / "voice" / "user_verifier.json"
            if legacy.is_file():
                profile = legacy
        if profile.exists():
            try:
                data = json.loads(profile.read_text(encoding="utf-8"))
                embeddings = np.asarray(data.get("embeddings", []), dtype=np.float32)
                if embeddings.ndim != 2 or embeddings.shape[1] != 32 or not np.all(np.isfinite(embeddings)):
                    raise ValueError("Expected finite 32-bin speaker embeddings")
                if np.any(np.linalg.norm(embeddings, axis=1) == 0):
                    raise ValueError("Empty speaker embedding")
                self.reference_embeddings = embeddings.tolist()
                self.loaded_profile_path = profile
                if "threshold" in data and "LORA_SPEAKER_VERIFY_THRESHOLD" not in os.environ:
                    file_thresh = float(data["threshold"])
                    if 0.0 < file_thresh < 1.0:
                        self.threshold = file_thresh
            except (OSError, ValueError, TypeError):
                logger.warning("Speaker enrollment is invalid; run lora enroll")
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
        if not is_owner:
            logger.info("imposter_rejected similarity=%.3f", max_sim)
        return is_owner, max_sim


class WakeWordDetector:
    """Detect addressed Hey Lora/Laura phrases while awake, using local STT.

    Optional wake weights can gate transcription. Sleep listening is explicit.
    """

    def __init__(
        self,
        wake_phrase: str = "Awaken LORA",
        chat_phrase: str = "Hey LORA",
        threshold: float = 0.65,
        model_path: str | None = None,
        speaker_verifier: SpeakerVerifier | None = None,
        transcriber: Any = None,
        engine: str = "transcript",
        listen_while_sleeping: bool = False,
    ) -> None:
        self.wake_phrase = wake_phrase
        self.chat_phrase = chat_phrase
        self.threshold = threshold
        self.speaker_verifier = speaker_verifier or SpeakerVerifier()
        self.transcriber = transcriber
        self.engine = engine
        self.listen_while_sleeping = listen_while_sleeping
        self._oww_model: Any = None

        resolved_model = model_path
        if resolved_model is None:
            for candidate in ("models/voice/awaken_lora.onnx", "models/voice/hey_lora.onnx"):
                if Path(candidate).is_file():
                    resolved_model = candidate
                    break
        if resolved_model and engine == "openwakeword":
            try:
                from openwakeword.model import Model
                self._oww_model = Model(wakeword_models=[resolved_model])
            except Exception:
                self._oww_model = None

    def match_text(self, text: str, is_sleeping: bool) -> tuple[bool, str]:
        """Match an addressed phrase, retaining an immediate prompt verbatim."""
        if is_sleeping and not getattr(self, "listen_while_sleeping", False):
            return False, ""
        match = re.match(r"^\s*(?:hey|hi|hello|yo|ok|okay|awaken|wake\s+up)[,\s]+(?:lora|laura)\b[,.:;!?\s]*(.*)$", text or "", re.IGNORECASE | re.DOTALL)
        return (True, match.group(1).strip()) if match else (False, "")

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

        if is_sleeping and not getattr(self, "listen_while_sleeping", False):
            return WakeWordResult(detected=False, phrase=target_phrase)
        if self._oww_model is not None:
            try:
                prediction = self._oww_model.predict((audio * 32767).astype(np.int16))
                wake_score = max((float(score) for score in prediction.values()), default=0.0)
                detected = wake_score >= self.threshold
            except Exception as exc:
                logger.warning("Wake model inference failed: %s", exc)
        elif self.transcriber is not None:
            result = self.transcriber.transcribe(audio, sample_rate=sample_rate)
            detected, extracted_query = self.match_text(result.text, False)
            wake_score = 0.0  # A phrase rule has no calibrated acoustic probability.
        # Missing weights use local transcription; loudness alone never activates.

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
