"""Wake Word Detection ('Awaken LORA') and Personalized Speaker Verification."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class WakeWordResult(BaseModel):
    """Immutable result of wake word evaluation."""

    model_config = ConfigDict(frozen=True)

    detected: bool
    phrase: str = "Awaken LORA"
    score: float = 0.0
    speaker_verified: bool = True
    speaker_similarity: float = 1.0


class SpeakerVerifier:
    """Verifies that the speaker's vocal acoustics match the enrolled owner."""

    def __init__(self, profile_path: Path | str | None = None, threshold: float = 0.70) -> None:
        self.threshold = threshold
        self.profile_path = Path(profile_path) if profile_path else Path("data/voice/user_verifier.json")
        self.reference_embeddings: list[list[float]] = []
        self._load_profile()

    def _load_profile(self) -> None:
        """Load enrolled speaker embeddings if profile file exists."""
        if self.profile_path.exists():
            try:
                data = json.loads(self.profile_path.read_text(encoding="utf-8"))
                self.reference_embeddings = data.get("embeddings", [])
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

        # Compute FFT energy distribution across frequency bins
        chunk = audio[-16000:] if len(audio) > 16000 else audio
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
        if not self.reference_embeddings:
            # If no enrolled voice profile exists yet, allow with default baseline
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
    """Detects 'Awaken LORA' with optional openWakeWord ONNX model or acoustic pattern matcher."""

    def __init__(
        self,
        wake_phrase: str = "Awaken LORA",
        threshold: float = 0.65,
        model_path: str | None = None,
        speaker_verifier: SpeakerVerifier | None = None,
    ) -> None:
        self.wake_phrase = wake_phrase
        self.threshold = threshold
        self.speaker_verifier = speaker_verifier or SpeakerVerifier()
        self._oww_model: Any = None

        if model_path:
            try:
                from openwakeword.model import Model
                self._oww_model = Model(wakeword_models=[model_path])
            except Exception:
                self._oww_model = None

    def process_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> WakeWordResult:
        """Evaluate audio buffer for wake phrase and enrolled speaker verification."""
        if len(audio) < 1600:  # less than 100ms
            return WakeWordResult(detected=False, phrase=self.wake_phrase, score=0.0)

        wake_score = 0.0

        if self._oww_model is not None:
            try:
                # openWakeWord expects 16-bit PCM integer chunks
                pcm_int16 = (audio * 32767).astype(np.int16)
                prediction = self._oww_model.predict(pcm_int16)
                for k, score in prediction.items():
                    if score > wake_score:
                        wake_score = float(score)
            except Exception:
                wake_score = 0.0
        else:
            # Acoustic envelope & energy check for wake activity
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms > 0.03:
                # Scale RMS into a surrogate activation score
                wake_score = float(min(1.0, rms * 10))

        detected = wake_score >= self.threshold

        if detected:
            is_verified, sim = self.speaker_verifier.verify(audio, sample_rate)
            return WakeWordResult(
                detected=is_verified,
                phrase=self.wake_phrase,
                score=wake_score,
                speaker_verified=is_verified,
                speaker_similarity=sim,
            )

        return WakeWordResult(
            detected=False,
            phrase=self.wake_phrase,
            score=wake_score,
            speaker_verified=True,
            speaker_similarity=0.0,
        )
