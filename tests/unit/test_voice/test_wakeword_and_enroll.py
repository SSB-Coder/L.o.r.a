"""Unit tests for SpeakerVerifier, WakeWordDetector, and EnrollmentSession."""

from pathlib import Path

import numpy as np

from lora.voice.enroll import EnrollmentSession
from lora.voice.wakeword import SpeakerVerifier, WakeWordDetector


def test_speaker_verifier_profile_and_similarity(tmp_path: Path):
    profile_file = tmp_path / "test_user_verifier.json"
    verifier = SpeakerVerifier(profile_path=profile_file, threshold=0.70)

    # Create audio signals with distinct frequency components (300 Hz vs 5000 Hz)
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    audio_low = (np.sin(2 * np.pi * 300 * t) * 0.1).astype(np.float32)
    audio_high = (np.sin(2 * np.pi * 5000 * t) * 0.1).astype(np.float32)

    # Save low-frequency reference profile
    feat_low = verifier.extract_features(audio_low)
    verifier.save_profile([feat_low.tolist()], speaker_name="tester")

    assert profile_file.exists()

    # Re-verify with matching audio -> should pass with high similarity
    is_ver, sim_match = verifier.verify(audio_low)
    assert is_ver is True
    assert sim_match > 0.95

    # Verify with distinct audio -> should fail threshold and have lower similarity
    is_ver_high, sim_diff = verifier.verify(audio_high)
    assert is_ver_high is False
    assert sim_diff < 0.70


def test_wake_word_detector_with_verifier(tmp_path: Path):
    profile_file = tmp_path / "owner.json"
    verifier = SpeakerVerifier(profile_path=profile_file, threshold=0.60)

    # Train owner profile on audio1
    audio1 = np.ones(3200, dtype=np.float32) * 0.08
    feat1 = verifier.extract_features(audio1)
    verifier.save_profile([feat1.tolist()], speaker_name="owner")

    detector = WakeWordDetector(wake_phrase="Awaken LORA", threshold=0.5, speaker_verifier=verifier)

    # 1. Too short audio (<100ms)
    res_short = detector.process_audio(np.ones(500, dtype=np.float32))
    assert res_short.detected is False

    # 2. Quiet audio -> not detected
    res_quiet = detector.process_audio(np.ones(3200, dtype=np.float32) * 0.001)
    assert res_quiet.detected is False

    # 3. Energetic audio matching owner -> detected and speaker verified
    res_wake = detector.process_audio(audio1)
    assert res_wake.detected is True
    assert res_wake.phrase == "Awaken LORA"
    assert res_wake.speaker_verified is True


def test_enrollment_session_from_clips(tmp_path: Path):
    profile_file = tmp_path / "enrolled.json"
    verifier = SpeakerVerifier(profile_path=profile_file)
    session = EnrollmentSession(verifier=verifier, samples_needed=3)

    clips = [
        np.ones(4000, dtype=np.float32) * 0.05,
        np.ones(4000, dtype=np.float32) * 0.06,
        np.ones(4000, dtype=np.float32) * 0.055,
    ]

    saved_path = session.enroll_from_audio_clips(clips, speaker_name="owner")
    assert saved_path.exists()
    assert len(verifier.reference_embeddings) == 3
