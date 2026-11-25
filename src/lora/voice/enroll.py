"""Voice Enrollment CLI for 'Awaken LORA' Personalized Speaker Verification."""

import argparse
import time
from pathlib import Path

import numpy as np

from lora.voice.capture import AudioCapture
from lora.voice.wakeword import SpeakerVerifier


class EnrollmentSession:
    """Manages the interactive recording and generation of owner voice embeddings."""

    def __init__(
        self,
        verifier: SpeakerVerifier | None = None,
        samples_needed: int = 5,
        clip_duration_seconds: float = 2.5,
    ) -> None:
        self.verifier = verifier or SpeakerVerifier()
        self.samples_needed = samples_needed
        self.clip_duration_seconds = clip_duration_seconds

    def enroll_from_audio_clips(
        self,
        clips: list[np.ndarray],
        speaker_name: str = "owner",
    ) -> Path:
        """Enroll from a pre-recorded list of audio arrays (programmatic / test mode)."""
        if not clips:
            raise ValueError("At least one voice sample is required.")
        embeddings: list[list[float]] = []
        for clip in clips:
            feat = self.verifier.extract_features(clip)
            if not np.all(np.isfinite(feat)) or not np.any(feat):
                raise ValueError("No usable voice sample was recorded. Check the microphone and try again.")
            embeddings.append(feat.tolist())

        self.verifier.save_profile(embeddings, speaker_name=speaker_name)
        return self.verifier.profile_path

    def run_interactive(self, speaker_name: str = "owner") -> Path:
        """Run interactive CLI enrollment with microphone."""
        capture = AudioCapture()
        if self.samples_needed < 1:
            raise ValueError("At least one voice sample is required.")
        clips: list[np.ndarray] = []

        print("\n" + "=" * 60)
        print("🎙️  L.O.R.A Voice Calibration: Personalized Voice Enrollment")
        print("Phrase to speak: 'Awaken LORA'")
        print(f"Required samples: {self.samples_needed}")
        print("=" * 60 + "\n")

        for i in range(1, self.samples_needed + 1):
            input(f"[{i}/{self.samples_needed}] Press Enter and say 'Awaken LORA' clearly: ")
            print("  Recording 2.5s...", end="", flush=True)

            capture.buffer.clear()
            if not capture.start():
                raise ValueError(f"Could not start the microphone: {capture.status_text()}")
            try:
                time.sleep(self.clip_duration_seconds)
            finally:
                capture.stop()

            clip = capture.buffer.read_all()
            print(" Captured.")

            rms = float(np.sqrt(np.mean(clip**2))) if len(clip) > 0 else 0.0
            if len(clip) < 512 or not np.isfinite(rms) or rms < 0.01:
                raise ValueError("The voice sample was empty or too quiet. Check the microphone and try again.")
            clips.append(clip)

        self.enroll_from_audio_clips(clips, speaker_name=speaker_name)
        print("\n✅ Calibration complete!")
        print(f"Voice profile saved to: {self.verifier.profile_path}\n")
        return self.verifier.profile_path


def main() -> None:
    parser = argparse.ArgumentParser(description="LORA Voice Enrollment CLI")
    parser.add_argument("--speaker", default="owner", help="Name of the speaker")
    parser.add_argument("--samples", type=int, default=5, help="Number of audio calibration samples")
    args = parser.parse_args()

    session = EnrollmentSession(samples_needed=args.samples)
    session.run_interactive(speaker_name=args.speaker)


if __name__ == "__main__":
    main()
