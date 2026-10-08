"""Unit tests for AudioBuffer, AudioCapture, and VoiceActivityDetector."""

import numpy as np

from lora.voice.capture import AudioBuffer, AudioCapture
from lora.voice.vad import SpeechState, VoiceActivityDetector


def test_audio_buffer_operations():
    buf = AudioBuffer(max_seconds=2.0, sample_rate=1000)

    # 1. Write 500 samples
    samples1 = np.ones(500, dtype=np.float32) * 0.5
    buf.write(samples1)
    assert buf.duration_seconds == 0.5

    # 2. Read last 0.2 seconds (200 samples)
    recent = buf.read_last_n_seconds(0.2)
    assert len(recent) == 200
    assert np.allclose(recent, 0.5)

    # 3. Write another 1000 samples
    samples2 = np.ones(1000, dtype=np.float32) * 0.8
    buf.write(samples2)
    assert buf.duration_seconds == 1.5

    # 4. Read all and clear
    all_audio = buf.read_all()
    assert len(all_audio) == 1500
    assert buf.duration_seconds == 0.0


def test_audio_capture_callbacks():
    capture = AudioCapture(sample_rate=16000, chunk_size=512)
    captured_chunks = []

    capture.register_chunk_callback(lambda chunk: captured_chunks.append(chunk))

    # Feed synthetic chunk
    fake_chunk = np.ones(512, dtype=np.float32) * 0.1
    capture.feed_samples(fake_chunk)

    assert len(captured_chunks) == 1
    assert len(captured_chunks[0]) == 512
    assert capture.buffer.duration_seconds == 512 / 16000

    # Start and stop lifecycle
    capture.start()
    assert capture.is_recording is True
    capture.stop()
    assert capture.is_recording is False


def test_vad_lifecycle_and_endpointing():
    vad = VoiceActivityDetector(
        sample_rate=16000,
        threshold=0.5,
        silence_duration_seconds=0.1,  # short for fast testing
        min_speech_duration_seconds=0.05,
    )

    silence_chunk = np.zeros(512, dtype=np.float32)
    speech_chunk = np.ones(512, dtype=np.float32) * 0.1  # high RMS

    # 1. Initially SILENCE
    state, prob = vad.process_chunk(silence_chunk)
    assert state == SpeechState.SILENCE
    assert prob < 0.2

    # 2. Feed speech chunks -> transition to SPEAKING
    for _ in range(3):
        state, prob = vad.process_chunk(speech_chunk)

    assert state == SpeechState.SPEAKING
    assert prob > 0.8

    # 3. Feed silence chunks -> transition to ENDED
    for _ in range(5):
        state, prob = vad.process_chunk(silence_chunk)

    assert state == SpeechState.ENDED

    # 4. Extract recorded utterance
    utterance = vad.get_and_reset_utterance()
    assert len(utterance) > 0
    assert vad.state == SpeechState.SILENCE
