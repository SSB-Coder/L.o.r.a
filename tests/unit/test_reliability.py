"""Repeated-use, cancellation, audio-state and compatibility regressions."""

import asyncio
import concurrent.futures
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lora.agent.turns import TurnCoordinator, TurnHandle, run_coroutine
from lora.core.paths import PROJECT_ROOT, project_path
from lora.voice.capture import AudioBuffer
from lora.voice.hotkey import PushToTalkListener
from lora.voice.speech_stream import SpeechStreamer
from lora.voice.stt import FasterWhisperTranscriber
from lora.voice.vad import SpeechState, VoiceActivityDetector
from lora.voice.wakeword import SpeakerVerifier


@pytest.fixture
def background_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(3)
        assert not thread.is_alive()
        loop.close()


def test_sixty_turns_survive_errors_and_release_model_slot(background_loop):
    from lora.models.scheduler import model_slot
    coordinator = TurnCoordinator()
    for i in range(60):
        handle = coordinator.begin()
        assert handle is not None

        async def turn():
            async with model_slot():
                await asyncio.sleep(0)
                if i % 7 == 0:
                    raise RuntimeError("simulated retrieval/model failure")
                return f"answer-{i}"

        if i % 7 == 0:
            with pytest.raises(RuntimeError):
                run_coroutine(background_loop, turn(), 2, handle)
        else:
            assert run_coroutine(background_loop, turn(), 2, handle) == f"answer-{i}"
        assert handle.finished.wait(1)
        assert not coordinator.busy
    assert run_coroutine(background_loop, asyncio.sleep(0, result="still alive"), 2) == "still alive"


def test_shutdown_drains_task_finalizers_before_loop_close(background_loop):
    from lora.agent.turns import drain_pending_tasks

    started = threading.Event()
    cleaned = threading.Event()

    async def pending():
        started.set()
        try:
            await asyncio.sleep(100)
        finally:
            await asyncio.sleep(.02)
            cleaned.set()

    handle = TurnHandle()
    errors = []

    def worker():
        try:
            run_coroutine(background_loop, pending(), 3, handle)
        except concurrent.futures.CancelledError:
            errors.append("cancelled")

    thread = threading.Thread(target=worker)
    thread.start()
    assert started.wait(1)
    run_coroutine(background_loop, drain_pending_tasks(), 2)
    thread.join(1)
    assert cleaned.is_set() and handle.finished.is_set()
    assert not thread.is_alive() and errors == ["cancelled"]


def test_timeout_waits_for_cleanup_and_then_accepts_another_turn(background_loop):
    coordinator = TurnCoordinator()
    cleaned = threading.Event()
    handle = coordinator.begin()

    async def hanging():
        try:
            await asyncio.sleep(100)
        finally:
            await asyncio.sleep(0.03)
            cleaned.set()

    with pytest.raises(TimeoutError):
        run_coroutine(background_loop, hanging(), 0.05, handle)
    assert cleaned.is_set() and handle.finished.is_set()
    next_turn = coordinator.begin()
    assert next_turn is not None and not coordinator.is_current(handle)
    assert run_coroutine(background_loop, asyncio.sleep(0, result="next"), 2, next_turn) == "next"


def test_cancel_does_not_release_ownership_before_native_work_unwinds(background_loop):
    coordinator = TurnCoordinator()
    handle = coordinator.begin()
    entered, release = threading.Event(), threading.Event()
    errors = []

    async def native_work():
        entered.set()
        # Simulate synchronous native inference: the loop cannot cancel it yet.
        assert release.wait(2)
        await asyncio.sleep(0)

    def worker():
        try:
            run_coroutine(background_loop, native_work(), 3, handle)
        except concurrent.futures.CancelledError:
            errors.append("cancelled")

    thread = threading.Thread(target=worker)
    thread.start()
    assert entered.wait(1)
    coordinator.cancel()
    assert coordinator.busy and coordinator.begin() is None
    assert not coordinator.is_current(handle)
    release.set()
    thread.join(3)
    assert not thread.is_alive() and errors == ["cancelled"]
    assert handle.finished.wait(1) and not coordinator.busy


def test_cancel_before_submission_runs_no_user_code(background_loop):
    handle = TurnHandle()
    handle.cancel()
    executed = []

    async def work():
        executed.append(True)

    with pytest.raises(concurrent.futures.CancelledError):
        run_coroutine(background_loop, work(), 2, handle)
    assert handle.finished.wait(1) and not executed


def test_startup_failure_can_release_unsubmitted_turn():
    handle = TurnHandle()
    handle.finish_if_unsubmitted()
    assert handle.finished.is_set()


def test_ring_buffer_wraps_without_exposing_mutable_storage():
    buffer = AudioBuffer(max_seconds=1, sample_rate=10)
    expected = []
    for size in (3, 8, 1, 19, 2):
        data = np.arange(size, dtype=np.float32)
        expected = (expected + data.tolist())[-10:]
        buffer.write(data)
        np.testing.assert_array_equal(buffer.read_last_n_seconds(1), expected)
    output = buffer.read_last_n_seconds(1)
    output.fill(-99)
    np.testing.assert_array_equal(buffer.read_all(), expected)
    assert buffer.duration_seconds == 0
    assert buffer._buffer.nbytes == 40


def test_vad_continuous_audio_is_bounded():
    vad = VoiceActivityDetector(max_utterance_seconds=1)
    for _ in range(100):
        state, _ = vad.process_chunk(np.full(512, 0.06, dtype=np.float32))
        if state == SpeechState.ENDED:
            break
    else:
        pytest.fail("Continuous noise/speech grew without an endpoint")
    assert len(vad.get_and_reset_utterance()) <= 16000 + 512


def test_streaming_silero_carries_state_and_context():
    calls = []

    class Session:
        def run(self, _, inputs):
            calls.append({k: v.copy() for k, v in inputs.items()})
            return np.array([0.95]), inputs['h'] + 1, inputs['c'] + 1

    vad = VoiceActivityDetector()
    vad._silero_vad = SimpleNamespace(session=Session())
    chunk = np.full(512, 0.03, dtype=np.float32)
    vad.evaluate_chunk(chunk)
    vad.evaluate_chunk(chunk)
    assert np.all(calls[0]['h'] == 0) and np.all(calls[1]['h'] == 1)
    np.testing.assert_array_equal(calls[1]['input'][0, :64], chunk[-64:])
    vad.reset()
    vad.evaluate_chunk(chunk)
    assert np.all(calls[2]['h'] == 0)


def test_existing_owner_enrollment_is_found_without_rewriting_it(tmp_path, monkeypatch):
    from lora.voice import wakeword
    monkeypatch.setattr(Path, 'home', lambda: tmp_path / 'home')
    monkeypatch.setattr(wakeword, '__file__', str(tmp_path / 'src/lora/voice/wakeword.py'))
    profile = tmp_path / 'data/voice/user_verifier.json'
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({'embeddings': [np.ones(32).tolist()], 'threshold': 0.7}))
    original = profile.read_bytes()
    verifier = SpeakerVerifier(enabled=True)
    assert verifier.loaded_profile_path == profile
    assert verifier.reference_embeddings
    assert not verifier.profile_path.exists() and profile.read_bytes() == original


def test_malformed_owner_profile_cannot_crash_transcription(tmp_path):
    profile = tmp_path / 'bad.json'
    profile.write_text(json.dumps({'embeddings': [[1, 2, 3]]}))
    verifier = SpeakerVerifier(profile_path=profile)
    assert not verifier.reference_embeddings


def test_silent_audio_never_loads_whisper():
    stt = FasterWhisperTranscriber()
    stt._ensure_model_loaded = lambda: pytest.fail("Silence loaded a speech model")
    assert not stt.transcribe(np.full(16000, 0.0005, dtype=np.float32)).text


def test_ctrl_escape_and_long_alias_both_interrupt(monkeypatch):
    keys = SimpleNamespace(**{k: k for k in ('ctrl', 'ctrl_l', 'ctrl_r', 'alt', 'alt_l', 'alt_r', 'shift', 'shift_l', 'shift_r', 'esc', 'space')})
    monkeypatch.setitem(sys.modules, 'pynput', SimpleNamespace(keyboard=SimpleNamespace(Key=keys)))
    fired = []
    listener = PushToTalkListener(on_killswitch=lambda: fired.append(True))
    for combo in ((keys.ctrl, keys.esc), (keys.ctrl, keys.alt, keys.shift, keys.esc)):
        for key in combo:
            listener._on_key_press(key)
        for key in reversed(combo):
            listener._on_key_release(key)
    assert len(fired) == 2


def test_full_audio_queue_cannot_block_interrupt_or_leak_producer():
    playback_started = threading.Event()
    done = threading.Event()

    def play(chunk, stop):
        playback_started.set()
        stop.wait(2)

    stream = SpeechStreamer(lambda text: text, play, on_done=lambda _: done.set())
    text = 'One sentence. Two sentences. Three sentences. Four sentences. Five sentences. '
    stream.feed(text)
    stream.finish(text)
    assert playback_started.wait(1)
    time.sleep(0.05)  # Let the producer fill its bounded audio queue.
    started = time.monotonic()
    stream.stop()
    assert time.monotonic() - started < 0.3
    assert done.wait(1)
    for thread in stream._threads:
        thread.join(1)
        assert not thread.is_alive()


def test_model_assets_resolve_from_another_launch_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert project_path('models/kokoro') == PROJECT_ROOT / 'models/kokoro'


def test_thirty_mixed_runtime_turns_keep_history_bounded(background_loop):
    from lora.agent.runtime import AgentRuntime, TurnResult
    from lora.dialogue import ConversationHistory
    from lora.models.laya_router import LayaDecision

    router = SimpleNamespace(predict=lambda q: LayaDecision(
        intent="research" if q.startswith("research") else "chat", intent_confidence=0,
    ))
    runtime = AgentRuntime("http://unused", None, ConversationHistory(max_turns=3),
                           registry_factory=lambda: None, laya_router=router)
    runtime.skills = None
    research_count = []

    async def research(q, on_status, on_text, *args, **kwargs):
        research_count.append(q)
        await asyncio.sleep(0)
        if "failure" in q:
            raise RuntimeError("search provider unavailable")
        on_text("Research answer", False)
        return TurnResult(text="Research answer", kind="research")

    async def chat(q, on_text, **kwargs):
        await asyncio.sleep(0)
        return TurnResult(text="I'm here.", kind="chat")

    runtime._run_research, runtime._run_chat = research, chat
    coordinator = TurnCoordinator()
    kinds = set()
    for i in range(30):
        q = ("what time is it", "Let's chat", "research stars", "research failure", "Let's chat again")[i % 5]
        result = run_coroutine(background_loop, runtime.run_turn(q, lambda _: None, lambda *args: None),
                               2, coordinator.begin())
        kinds.add(result.kind)
        assert len(runtime.history.messages()) <= 6
        assert not coordinator.busy
    assert kinds == {"smalltalk", "chat", "research", "error"}
    assert len(research_count) == 12


def test_broadcast_uses_socket_owner_loop(background_loop):
    from lora.gateway.ws import ConnectionManager
    manager = ConnectionManager()
    owners = []

    class Socket:
        async def send_json(self, event):
            owners.append(asyncio.get_running_loop())

    connection = Socket()
    run_coroutine(background_loop, manager.connect(connection), 2)

    async def broadcast():
        await manager.broadcast({"type": "test"})

    asyncio.run(broadcast())
    assert owners == [background_loop]


def test_job_worker_survives_runtime_creation_failure():
    from lora.agent.jobs import JobWorker
    from lora.gateway.queue import JobItem, JobOrigin, JobQueue, JobStatus
    queue = JobQueue()
    completed = threading.Event()

    class Runtime:
        async def run_job(self, *args, **kwargs):
            completed.set()
            return {"answer": "recovered"}

    attempts = []
    def factory(origin):
        attempts.append(origin)
        if len(attempts) == 1:
            raise RuntimeError("temporary initialization failure")
        return Runtime()

    async def enqueue():
        await queue.enqueue(JobItem("first", JobOrigin.MOBILE, "first"))
        await queue.enqueue(JobItem("second", JobOrigin.MOBILE, "second"))

    asyncio.run(enqueue())
    worker = JobWorker(queue, factory, poll_s=0.01)
    worker.start()
    try:
        assert completed.wait(2)
    finally:
        worker.stop()
        worker.join(3)
    assert not worker.is_alive() and worker._loop.is_closed()
    assert queue.get_job("first").status == JobStatus.FAILED
    assert queue.get_job("second").status == JobStatus.COMPLETED


def test_one_huge_sentence_cannot_bypass_speech_budget():
    spoken = []
    done = threading.Event()
    stream = SpeechStreamer(lambda text: spoken.append(text) or text,
                            lambda *args: None, max_chars=100, on_done=lambda _: done.set())
    stream.finish("word " * 10000)
    assert done.wait(1)
    assert sum(map(len, spoken)) <= 100


def test_lazy_router_constructor_does_not_load_native_model(monkeypatch):
    from lora.models import laya_router
    def unexpected(*args, **kwargs):
        pytest.fail("Lazy router eagerly loaded native weights")
    monkeypatch.setattr(laya_router, 'laya', SimpleNamespace(Agent=unexpected, Router=unexpected))
    router = laya_router.LayaDecisionRouter(lazy_load=True)
    assert router.agent is None and router.router is None and len(router.questions) == 12


def test_numpy_index_matches_exact_dense_search_and_round_trips(tmp_path):
    from lora.grounding.vector_index import NumpyFlatIndex, load_index, save_index
    rng = np.random.default_rng(7)
    values = rng.standard_normal((256, 32)).astype(np.float32)
    query = rng.standard_normal((2, 32)).astype(np.float32)
    index = NumpyFlatIndex(32)
    index.add(values[:100])
    index.add(values[100:])
    scores, indices = index.search(query, 5)
    expected = query @ values.T
    order = np.argsort(-expected, axis=1)[:, :5]
    np.testing.assert_array_equal(indices, order)
    np.testing.assert_allclose(scores, np.take_along_axis(expected, order, axis=1))
    path = tmp_path / 'vectors.index'
    save_index(index, path)
    restored = load_index(path)
    assert restored.ntotal == 256
    np.testing.assert_array_equal(restored.search(query, 5)[1], indices)


def test_default_retrieval_imports_no_native_faiss(monkeypatch):
    from lora.grounding.vector_index import NumpyFlatIndex, make_index
    monkeypatch.delenv('LORA_VECTOR_BACKEND', raising=False)
    monkeypatch.setitem(sys.modules, 'faiss', None)
    assert isinstance(make_index(32), NumpyFlatIndex)


def test_deep_dropdown_handles_missing_verifier_without_inflating_score():
    from lora.agent.deep import build_verification_dropdown_meta
    from lora.agent.verification import Verification
    report = Verification(verdict="abstain", score=0, supported=0, total=2, stages=[
        {"id": "V3", "score": None, "status": "skipped"},
        {"id": "V7", "score": 1, "status": "pass"},
        {"id": "V8", "score": 0, "status": "fail"},
    ])
    meta = build_verification_dropdown_meta(report, roundtable_info={"score": 1, "decision": "approve"})
    assert meta['score'] == 0 and meta['stages'][0]['score'] is None


def test_legacy_media_index_rebuild_preserves_saved_files(tmp_path, monkeypatch):
    from lora.grounding.media_vector_db import MediaVectorDB
    monkeypatch.delenv('LORA_VECTOR_BACKEND', raising=False)
    embedder = SimpleNamespace(encode=lambda texts: np.ones((len(texts), 4), dtype=np.float32))
    db = MediaVectorDB(embedding_engine=embedder, storage_dir=tmp_path)
    source = tmp_path / 'note.txt'
    source.write_text('A saved source about stars and space.')
    db.add_file(source)
    db.save_db()
    index = tmp_path / 'faiss.index'
    index.write_bytes(b'old-native-faiss-binary')
    metadata = (tmp_path / 'metadata.json').read_bytes()
    restored = MediaVectorDB(embedding_engine=embedder, storage_dir=tmp_path)
    assert restored.load_db()
    assert restored.chunk_count == db.chunk_count
    assert index.read_bytes() == b'old-native-faiss-binary'
    assert (tmp_path / 'metadata.json').read_bytes() == metadata


def test_cancelled_refinement_drains_its_worker(background_loop):
    from lora.agent.turns import run_owned_thread
    started, stop, cleaned = threading.Event(), threading.Event(), threading.Event()
    coordinator = TurnCoordinator()
    handle = coordinator.begin()
    errors = []

    def native():
        started.set()
        assert stop.wait(2)
        cleaned.set()

    def worker():
        try:
            run_coroutine(background_loop, run_owned_thread(native, on_cancel=stop.set), 3, handle)
        except concurrent.futures.CancelledError:
            errors.append("cancelled")

    thread = threading.Thread(target=worker)
    thread.start()
    assert started.wait(1)
    coordinator.cancel()
    thread.join(3)
    assert not thread.is_alive() and cleaned.is_set()
    assert handle.finished.is_set() and errors == ['cancelled']


@pytest.mark.parametrize('query', ["How's your day going?", "I'm tired", "Can we talk?", "I've had a long day"])
def test_personal_conversation_does_not_fall_back_to_web_research(query):
    from lora.models.laya_router import LayaDecisionRouter
    result = LayaDecisionRouter._heuristic_predict(query)
    assert result['answers']['intent']['choice'] == 'chat'


def test_explicit_research_still_overrides_casual_keywords():
    from lora.models.laya_router import LayaDecisionRouter
    result = LayaDecisionRouter._heuristic_predict('Research how jokes affect memory')
    assert result['answers']['intent']['choice'] == 'research'


def test_personal_indexed_question_stays_local_with_explicit_unchecked_status(background_loop, monkeypatch):
    from lora.agent import runtime as module
    from lora.agent.runtime import AgentRuntime
    from lora.agent.verification import Verification
    from lora.dialogue import ConversationHistory
    from lora.models.laya_router import LayaDecision

    router = SimpleNamespace(predict=lambda _: LayaDecision(intent="research", intent_confidence=0))
    media = SimpleNamespace(file_count=1, search=lambda *args: [
        {"filename": "my-notes.txt", "text": "The experiment ran for forty epochs.", "score": .9}
    ])
    runtime = AgentRuntime('http://unused', None, ConversationHistory(), lambda: None,
                           media_db=media, laya_router=router)
    runtime.skills = None
    events = []

    async def stream(messages, on_text, **kwargs):
        assert 'forty epochs' in messages[-1]['content']
        assert kwargs['speak'] is False
        on_text('Unverified draft.', False)
        return 'Unverified draft.'

    async def verify(*args, **kwargs):
        return Verification('abstain', 0, 0, 1, repaired_answer='I could not verify that yet.')

    runtime._stream = stream
    monkeypatch.setattr(module, 'verify_answer', verify)
    result = run_coroutine(background_loop, runtime.run_turn('How many epochs did my model need?',
                           lambda _: None, lambda text, speak=True: events.append((text, speak))), 2)
    assert result.kind == 'chat'
    assert result.text == 'Unverified draft.'
    assert result.meta['verdict'] == 'unchecked'
    assert result.meta['source_check']['status'] == 'missing_citations'
    assert 'score' not in result.meta['source_check']
    assert [text for text, speak in events if speak] == ['Unverified draft.']
