"""Unit tests for FasterWhisperTranscriber and TtsEngine."""

from unittest.mock import MagicMock

import numpy as np

from lora.voice.stt import FasterWhisperTranscriber
from lora.voice.tts.engine import TtsEngine, split_sentences
from lora.voice.tts.worker import Qwen3TtsWorkerProcess


def test_split_sentences():
    text = "Hello LORA! Can you open VS Code? Also search arXiv for quantum computing."
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello LORA!"
    assert sentences[1] == "Can you open VS Code?"
    assert sentences[2] == "Also search arXiv for quantum computing."

    assert split_sentences("") == []


def test_stt_transcription_empty_and_mocked():
    transcriber = FasterWhisperTranscriber(lazy_load=True)

    # Empty audio
    res_empty = transcriber.transcribe(np.zeros(0, dtype=np.float32))
    assert res_empty.text == ""
    assert res_empty.duration_seconds == 0.0

    # Mock WhisperModel
    mock_model = MagicMock()
    mock_segment1 = MagicMock()
    mock_segment1.text = "open VS Code"
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model.transcribe.return_value = ([mock_segment1], mock_info)

    transcriber._model = mock_model

    fake_audio = np.ones(16000, dtype=np.float32) * 0.1
    res = transcriber.transcribe(fake_audio)
    assert res.text == "open VS Code"
    assert res.language == "en"
    assert res.duration_seconds == 1.0


def test_tts_worker_and_engine_playback():
    worker = Qwen3TtsWorkerProcess(spawn_process=False)

    # Empty text
    chunk_empty = worker.synthesize("")
    assert len(chunk_empty.samples) == 0

    # Normal text
    chunk = worker.synthesize("Awaken LORA")
    assert len(chunk.samples) > 0
    assert chunk.sample_rate == 24000

    # Engine with custom playback sink
    played_audio_chunks = []

    def mock_sink(samples: np.ndarray, rate: int) -> None:
        played_audio_chunks.append(samples)

    engine = TtsEngine(worker=worker, audio_playback_sink=mock_sink)
    chunks = engine.synthesize_and_speak("First sentence. Second sentence.")
    assert len(chunks) == 2
    assert len(played_audio_chunks) == 2


def test_tts_engine_barge_in_interruption():
    worker = Qwen3TtsWorkerProcess(spawn_process=False)
    played_count = 0

    engine = TtsEngine(worker=worker)

    def mock_sink(samples: np.ndarray, rate: int) -> None:
        nonlocal played_count
        played_count += 1
        # Interruption happens while chunk 1 is being spoken
        engine.interrupt()

    engine.audio_playback_sink = mock_sink

    chunks = engine.synthesize_and_speak("One sentence here. Two sentences here. Three sentences here. Four sentences here.")
    # Interrupted after first chunk
    assert len(chunks) == 1
    assert played_count == 1
    assert engine.is_speaking is False
