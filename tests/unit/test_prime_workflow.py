"""Regressions from the user's native fault, busy listening and slow startup."""

import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lora.voice.capture import AudioCapture
from lora.voice.isolation import SpeechProcess, SpeechWorkerClosedError, SpeechWorkerError


class _Backend:
    def __init__(self, settings=None):
        self.settings = settings or {}

    def echo(self, value):
        return value

    def crash(self):
        os._exit(73)

    def block(self):
        Path(self.settings["started"]).write_text("ready", encoding="utf-8")
        time.sleep(30)


def _factory(kind, settings):
    return _Backend(settings)


def test_native_worker_crash_does_not_exit_parent_and_next_request_recovers():
    worker = SpeechProcess('stt', {}, timeout=10, factory=_factory)
    try:
        assert worker.call('echo', 'first') == 'first'
        with pytest.raises(SpeechWorkerError, match='next request will restart'):
            worker.call('crash')
        assert worker.call('echo', 'still alive') == 'still alive'
        assert worker.restarts == 2
    finally:
        worker.close()
    with pytest.raises(SpeechWorkerError, match='closed'):
        worker.call('echo', 'no')


def test_intentional_close_unblocks_inflight_speech_without_restart(tmp_path):
    started = tmp_path / "worker-entered"
    # No cross-process condition objects: terminating a child while it holds a
    # condition can itself corrupt that test fixture's synchronization state.
    worker = SpeechProcess("tts", {"started": str(started)}, timeout=30, factory=_factory)
    errors = []

    def call():
        try:
            worker.call("block")
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=call)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not started.is_file() and time.monotonic() < deadline:
            time.sleep(.01)
        assert started.is_file(), "Speech worker did not enter the pending call"
        worker.close()
        thread.join(3)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], SpeechWorkerClosedError)
        assert "restart" not in str(errors[0])
        with pytest.raises(SpeechWorkerClosedError):
            worker.call("echo", "must not restart")
        assert worker.restarts == 1 and worker._process is None
        worker.close()  # Repeated shutdown remains safe.
    finally:
        worker.close()
        thread.join(3)


def test_close_during_worker_startup_cannot_miss_the_new_child(monkeypatch):
    entered, allow_start, terminated = threading.Event(), threading.Event(), threading.Event()
    errors = []

    class Process:
        exitcode = None
        running = False

        def start(self):
            entered.set()
            assert allow_start.wait(3)
            self.running = True

        def is_alive(self):
            return self.running and not terminated.is_set()

        def terminate(self):
            self.exitcode = 0
            terminated.set()

        def join(self, timeout=None): pass
        def close(self): pass

    class Connection:
        def send(self, value): pass
        def poll(self, timeout):
            assert terminated.wait(3), "close() missed the child being started"
            return True
        def recv(self): raise EOFError
        def close(self): pass

    process = Process()
    context = SimpleNamespace(Pipe=lambda: (Connection(), Connection()), Process=lambda **kwargs: process)
    monkeypatch.setattr("lora.voice.isolation.mp.get_context", lambda _: context)
    worker = SpeechProcess("tts", {})

    def call():
        try:
            worker.call("synthesize", "queued sentence")
        except Exception as exc:
            errors.append(exc)

    caller = threading.Thread(target=call)
    closer = threading.Thread(target=worker.close)
    caller.start()
    try:
        assert entered.wait(2)
        closer.start()
        allow_start.set()
        caller.join(4)
        closer.join(4)
        assert not caller.is_alive() and not closer.is_alive()
        assert terminated.is_set() and worker._process is None
        assert len(errors) == 1 and isinstance(errors[0], SpeechWorkerClosedError)
    finally:
        allow_start.set()
        terminated.set()
        caller.join(4)
        if closer.ident is not None:
            closer.join(4)
        worker.close()


def test_microphone_callback_returns_before_slow_inference_and_owns_audio():
    capture = AudioCapture()
    capture.is_recording = True
    called = threading.Event()
    release = threading.Event()
    received = []

    def slow(samples):
        called.set()
        release.wait(2)
        received.append(samples.copy())

    capture.register_chunk_callback(slow)
    capture._consumer = threading.Thread(target=capture._consume_audio)
    capture._consumer.start()
    block = np.ones((512, 1), dtype=np.float32)
    try:
        before = time.monotonic()
        capture._audio_callback(block, 512, None, None)
        assert time.monotonic() - before < .1
        assert called.wait(1)
        block.fill(0)  # PortAudio is allowed to reuse its original memory.
        release.set()
    finally:
        capture.close()
    assert received and np.all(received[0] == 1)


def test_two_microphone_starts_create_only_one_native_stream(monkeypatch):
    streams = []

    class Stream:
        def __init__(self, **kwargs):
            time.sleep(.02)
            streams.append(self)
        def start(self): pass
        def stop(self): pass
        def close(self): pass

    monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(
        query_devices=lambda *args: {'name': 'fake', 'default_samplerate': 16000}, InputStream=Stream))
    capture = AudioCapture()
    threads = [threading.Thread(target=capture.start) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    capture.close()
    assert len(streams) == 1


def test_fast_routing_never_loads_neural_weights():
    from lora.models.laya_router import LayaDecisionRouter
    router = LayaDecisionRouter(lazy_load=True)
    router._ensure_backend = MagicMock(side_effect=AssertionError('native router must not load'))
    assert router.predict_fast('tell me a joke').intent == 'chat'
    assert router.predict_fast('open notepad').intent == 'control'
    assert router.predict_fast('research mamba models').intent == 'research'


def test_confident_neural_speech_not_masked_by_fan_rms():
    from lora.voice.vad import VoiceActivityDetector
    vad = VoiceActivityDetector()
    vad.noise_floor = .03
    vad._silero_vad = SimpleNamespace(session=SimpleNamespace(run=lambda *args: (
        np.array([.9]), vad._silero_h, vad._silero_c)))
    assert vad.evaluate_chunk(np.full(512, .004, dtype=np.float32)) == pytest.approx(.9)


def test_writer_selected_for_complex_chat_without_automatic_swap(monkeypatch):
    from lora.agent.runtime import AgentRuntime
    from lora.dialogue import ConversationHistory
    rt = AgentRuntime('http://unused', None, ConversationHistory(), lambda: None)
    calls = []

    async def sources(*args): return []
    async def stream(messages, on_text, **kwargs):
        calls.append(kwargs.get('role'))
        return 'Hello.'

    rt._user_sources = sources
    rt._stream = stream
    import asyncio
    asyncio.run(rt._run_chat('Compare the architectural trade-offs in detail', lambda *args: None))
    assert calls == ['writer']


def test_enrolled_owner_threshold_is_honored(tmp_path):
    import json

    from lora.voice.wakeword import SpeakerVerifier
    profile = tmp_path / 'owner.json'
    profile.write_text(json.dumps({'embeddings': [np.ones(32).tolist()], 'threshold': .55}))
    verifier = SpeakerVerifier(profile_path=profile, threshold=.7)
    assert verifier.threshold == .55


def test_continuous_sleep_listening_is_explicit_and_still_requires_addressing():
    from lora.voice.wakeword import WakeWordDetector
    detector = WakeWordDetector(listen_while_sleeping=True)
    assert detector.match_text('Hey Laura, open notepad', is_sleeping=True) == (True, 'open notepad')
    assert detector.match_text('open notepad', is_sleeping=True) == (False, '')
    assert not WakeWordDetector().match_text('Hey Laura', is_sleeping=True)[0]


def test_configured_documents_root_resolves_redirect_without_granting_other_folders(tmp_path, monkeypatch):
    from lora.safety.policy import PolicyEngine
    from lora.tools.files import known
    documents = tmp_path / 'OneDrive' / 'Documents'
    documents.mkdir(parents=True)
    monkeypatch.setattr(known, '_windows_known_folder', lambda name: documents)
    config = tmp_path / 'policy.yaml'
    config.write_text('allowed_roots: ["~/Documents"]\n')
    policy = PolicyEngine(config_path=config)
    assert policy.check_path(str(documents / 'notes.txt'), origin='desktop', mode='read').allowed
    assert not policy.check_path(str(tmp_path / 'Downloads' / 'notes.txt'), origin='desktop', mode='read').allowed


def test_completed_speech_no_longer_blocks_deferred_listening():
    from lora.voice.speech_stream import SpeechStreamer
    done = threading.Event()
    streamer = SpeechStreamer(lambda _: None, lambda *args: None,
                               on_done=lambda _: done.set())
    streamer.feed('Hello there. ')
    streamer.finish()
    assert done.wait(2)
    assert not streamer.has_pending


def test_mcp_opt_in_restores_registered_hub_without_starting_by_default(monkeypatch, tmp_path):
    from lora.agent.runtime import AgentRuntime
    from lora.dialogue import ConversationHistory
    from lora.tools import mcp, mcp_stdio
    manager, hub = MagicMock(), MagicMock()
    monkeypatch.setattr(mcp, 'McpClientManager', lambda **kwargs: manager)
    monkeypatch.setattr(mcp_stdio, 'McpHub', lambda instance: hub)
    rt = AgentRuntime('http://unused', None, ConversationHistory(), lambda: SimpleNamespace(action_gate=MagicMock()),
                     mcp_config_path=tmp_path / 'mcp.json')
    monkeypatch.delenv('LORA_ENABLE_MCP', raising=False)
    rt.start_mcp()
    hub.connect_all.assert_not_called()
    monkeypatch.setenv('LORA_ENABLE_MCP', '1')
    rt.start_mcp()
    rt.start_mcp()
    hub.connect_all.assert_called_once()
    assert rt.mcp_hub is hub


@pytest.mark.skipif(os.name != 'nt', reason='Windows filename playback contract')
def test_windows_playback_uses_supported_flags_and_never_stops_capture(monkeypatch):
    from pathlib import Path

    from lora.voice.playback import play_audio
    paths = []
    stop = threading.Event()

    def play(sound, flags):
        if sound is not None:
            assert isinstance(sound, str) and flags == 3
            assert Path(sound).is_file()
            paths.append(sound)

    sounddevice = MagicMock()
    monkeypatch.setitem(sys.modules, 'sounddevice', sounddevice)
    monkeypatch.setitem(sys.modules, 'winsound', SimpleNamespace(
        PlaySound=play, SND_FILENAME=1, SND_ASYNC=2))
    play_audio(np.ones(1000, dtype=np.float32) * .1, 24000, stop,
               lambda _: stop.set(), threading.Event())
    assert paths and not Path(paths[0]).exists()
    sounddevice.play.assert_not_called()
    sounddevice.stop.assert_not_called()
