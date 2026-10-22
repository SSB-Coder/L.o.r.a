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

    # Loudness alone cannot activate; sleeping capture is disabled.
    res_wake = detector.process_audio(audio1, current_state="sleep")
    assert res_wake.detected is False
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


def test_dual_activation_sleep_and_idle():
    from unittest.mock import MagicMock
    from lora.voice.stt import TranscriptionResult

    mock_stt = MagicMock()

    detector = WakeWordDetector(
        wake_phrase="Awaken LORA",
        chat_phrase="Hey LORA",
        transcriber=mock_stt,
    )

    audio_clip = np.ones(16000, dtype=np.float32) * 0.05

    # Sleep never transcribes; resume via click/hotkey.
    mock_stt.transcribe.return_value = TranscriptionResult(text="Awaken LORA please", duration_seconds=1.0)
    res_sleep = detector.process_audio(audio_clip, current_state="sleep")
    assert res_sleep.detected is False
    assert res_sleep.phrase == "Awaken LORA"
    mock_stt.transcribe.assert_not_called()

    # 2. State IDLE: user says "Hey LORA, what is quantum computing?"
    mock_stt.transcribe.return_value = TranscriptionResult(text="Hey LORA what is quantum computing?", duration_seconds=1.5)
    res_idle = detector.process_audio(audio_clip, current_state="idle")
    assert res_idle.detected is True
    assert res_idle.phrase == "Hey LORA"
    assert "quantum computing" in res_idle.extracted_query

    # 3. State SLEEP: user says "Hey LORA" -> should NOT wake from sleep!
    mock_stt.transcribe.return_value = TranscriptionResult(text="Hey LORA", duration_seconds=1.0)
    res_sleep_ignore = detector.process_audio(audio_clip, current_state="sleep")
    assert res_sleep_ignore.detected is False


def test_speaker_verifier_disabled_and_env_override(tmp_path: Path, monkeypatch):
    profile_file = tmp_path / "test_user.json"
    verifier = SpeakerVerifier(profile_path=profile_file, threshold=0.70)
    # Save a profile
    audio = np.ones(16000, dtype=np.float32) * 0.1
    verifier.save_profile([verifier.extract_features(audio).tolist()])

    # Distinct audio would normally fail
    bad_audio = np.zeros(16000, dtype=np.float32)
    verifier.threshold = 0.99

    # 1. When enabled=False directly
    verifier.enabled = False
    is_ver, sim = verifier.verify(bad_audio)
    assert is_ver is True
    assert sim == 1.0

    # 2. When env var LORA_SPEAKER_VERIFY_ENABLED=0
    monkeypatch.setenv("LORA_SPEAKER_VERIFY_ENABLED", "0")
    verifier2 = SpeakerVerifier(profile_path=profile_file)
    assert verifier2.enabled is False
    is_ver2, sim2 = verifier2.verify(bad_audio)
    assert is_ver2 is True


def test_speaker_verifier_slices_active_speech(tmp_path: Path):
    verifier = SpeakerVerifier(profile_path=tmp_path / "test.json")
    t = np.linspace(0, 0.8, int(16000 * 0.8), endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.2).astype(np.float32)
    # Tone padded with silence on both sides (total 32000 samples = 2s)
    padded = np.concatenate([np.zeros(4000, dtype=np.float32), tone, np.zeros(15200, dtype=np.float32)])
    feat = verifier.extract_features(padded)
    # Norm should be 1.0 and features should have non-zero components in speech bins
    assert np.linalg.norm(feat) > 0.99
    assert np.sum(feat) > 0.0


