import argparse
import logging
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import uvicorn

from lora.config.manager import ConfigManager, LoraUserConfig
from lora.gateway.server import create_gateway_app
from lora.storage.notebook import NotebookStore
from lora.ui.audio_cues import AudioCueManager, SoundCue

try:
    from lora.ui.hud import LoraDesktopHUD
    from lora.ui.tray import LoraSystemTray
    from lora.ui.widget import AmbientDropletWidget, WidgetState
except Exception:  # pragma: no cover - headless environment without PySide6
    LoraDesktopHUD = None  # type: ignore[assignment,misc]
    LoraSystemTray = None  # type: ignore[assignment,misc]
    AmbientDropletWidget = None  # type: ignore[assignment,misc]
    WidgetState = None  # type: ignore[assignment,misc]

logger = logging.getLogger("lora")


def _add_run_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Path to custom LORA config directory (default: ~/.lora)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Gateway HTTP/WebSocket port (default: 8765)",
    )
    p.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Gateway bind host (default: 127.0.0.1)",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Run without desktop UI (Gateway daemon only)",
    )
    p.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable microphone capture and TTS voice pipeline",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Force air-gapped offline research mode",
    )
    p.add_argument(
        "--mobile-bind",
        default=None,
        metavar="HOST:PORT",
        help="Also listen for phone requests on this LAN address (token-protected). Off by default.",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="L.O.R.A - Layer Oriented Research Assistant (Native Desktop Platform)"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="LORA subcommands")

    start_parser = subparsers.add_parser("start", help="Start L.O.R.A desktop HUD or daemon")
    _add_run_arguments(start_parser)

    stop_parser = subparsers.add_parser("stop", aliases=["kill"], help="Cleanly shut down LORA and its workers")
    stop_parser.add_argument("--config-dir", type=Path, default=None)

    net_parser = subparsers.add_parser("net-check", help="Verify LAN IP, Ollama status, and ports")
    net_parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL")

    enroll_parser = subparsers.add_parser("enroll", help="Calibrate voice profile for 'Awaken LORA'")
    enroll_parser.add_argument("--speaker", default="owner", help="Speaker name")
    enroll_parser.add_argument("--samples", type=int, default=5, help="Number of audio samples")

    _add_run_arguments(parser)
    return parser


def init_platform(base_dir: Path | None = None) -> tuple[ConfigManager, LoraUserConfig]:
    """Initializes the ~/.lora directory hierarchy, locks NTFS ACLs, and loads configuration."""
    cfg_mgr = ConfigManager(base_dir=base_dir)
    cfg_mgr.ensure_ecosystem(apply_acls=True)
    from lora.storage.migration import migrate_notebooks
    migrate_notebooks(cfg_mgr.paths.base_dir, Path(__file__).resolve().parents[2])
    user_cfg = cfg_mgr.load_config()
    return cfg_mgr, user_cfg


def _detect_lan_ip() -> str:
    """Detect the active private LAN IPv4 address for mobile QR pairing (never 0.0.0.0)."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass
    return "127.0.0.1"


def start_gateway_thread(
    host: str,
    port: int,
    queue=None,
    origin: str = "desktop",
    token_validator=None,
    media_db=None,
    connection_manager=None,
    notebook_store=None,
) -> uvicorn.Server | None:
    """Launches the FastAPI Gateway inside a dedicated background thread."""
    import socket

    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((probe_host, port)) == 0:
            logger.info("Gateway daemon already active at http://%s:%d", host, port)
            return None

    from lora.gateway.auth import HostOriginGuard

    host_guard = HostOriginGuard(listener_origin=origin, mobile_ip=host, port=port)
    app = create_gateway_app(
        origin=origin,
        queue=queue,
        token_validator=token_validator,
        host_guard=host_guard,
        media_db=media_db,
        connection_manager=connection_manager,
        notebook_store=notebook_store,
    )
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    logger.info("Gateway daemon running at http://%s:%d", host, port)
    return server


def run_desktop_app(
    cfg_mgr: ConfigManager,
    user_cfg: LoraUserConfig,
    args: argparse.Namespace,
) -> int:
    """Launches the PySide6 holographic HUD, Ambient Droplet Widget, System Tray, and Voice Pipeline.

    Threading rule: only the GUI thread may touch ``hud``/``tray``/``widget``. Worker threads (Ollama,
    Whisper, TTS, the sounddevice callback, the pynput hotkey) hand results over with
    ``ui.post(fn)``. ``QTimer.singleShot(0, fn)`` from those threads silently never fires.
    """
    import asyncio

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from lora.agent.runtime import AgentRuntime, TurnResult
    from lora.dialogue import ConversationHistory
    from lora.ui.threadsafe import UiDispatcher

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    # CRITICAL: System tray + popup dialogs (QMessageBox / QFileDialog) must NEVER quit the app when closed!
    app.setQuitOnLastWindowClosed(False)
    ui = UiDispatcher()  # lives on the GUI thread

    from lora.grounding.media_vector_db import MediaVectorDB

    media_db = MediaVectorDB()
    audio = AudioCueManager(enabled=user_cfg.voice.sound_effects_enabled and not args.no_audio)
    hud = LoraDesktopHUD(media_db=media_db)
    tray = LoraSystemTray()
    widget = AmbientDropletWidget()

    capture = None
    vad = None
    hotkey = None
    from lora.agent.turns import TurnCoordinator, run_coroutine

    turns = TurnCoordinator()
    is_exiting = threading.Event()
    current_streamer: list[object] = [None]
    speech_cooldown_until = [time.monotonic() + 1.0]  # ignore startup chime during initial 1.0s
    pending_utterances = deque(maxlen=2)
    deferred_utterances = deque(maxlen=2)
    transcription_lock = threading.Lock()
    armed_until = [0.0]
    voice_generation = [0]

    def on_sleep_toggled(is_sleeping: bool, play_sound: bool = True) -> None:
        with transcription_lock:
            voice_generation[0] += 1
            armed_until[0] = 0.0
            pending_utterances.clear()
            deferred_utterances.clear()
        speech_cooldown_until[0] = time.monotonic() + 0.6
        if is_sleeping:
            if play_sound and hud._current_voice_state not in ("sleep", "sleeping"):
                audio.play_cue(SoundCue.SLEEP)
            tray.update_state(WidgetState.SLEEPING)
            widget.set_state(WidgetState.SLEEPING)
            hud.set_voice_state("sleeping")
            stop_current_speech(update_ui=False)
            if capture is not None:
                if master_config.voice.close_mic_on_sleep:
                    capture.stop()
                capture.buffer.clear()
            if vad is not None:
                vad.reset()
        else:
            if play_sound and hud._current_voice_state in ("sleep", "sleeping"):
                audio.play_cue(SoundCue.WAKE)
            tray.update_state(WidgetState.IDLE)
            widget.set_state(WidgetState.IDLE)
            hud.set_voice_state("idle")
            if capture is not None and not capture.is_streaming and not args.no_audio:
                try:
                    capture.start()
                except Exception:
                    pass

    hud.sleep_toggled.connect(on_sleep_toggled)
    tray.sleep_toggled.connect(lambda: hud.sleep_toggled.emit(hud._current_voice_state != "sleeping"))
    tray.open_hud_requested.connect(hud.toggle_window)
    widget.hud_requested.connect(hud.toggle_window)
    widget.sleep_toggled.connect(on_sleep_toggled)

    def on_exit() -> None:
        if is_exiting.is_set():
            return
        is_exiting.set()
        turns.cancel()
        # Speech can outlive the model turn. Stop its producer before closing
        # the native worker, otherwise queued sentences hit a closed pipe.
        streamer = current_streamer[0]
        if streamer is not None:
            streamer.stop()
        logger.info("Exiting L.O.R.A, releasing audio capture stream...")
        audio.play_cue(SoundCue.SLEEP)
        if capture is not None:
            try:
                capture.close() if hasattr(capture, "close") else capture.stop()
            except Exception:
                pass
        if hotkey is not None:
            try:
                hotkey.stop()
            except Exception:
                pass
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        widget.close()
        hud.close()
        tray.hide()
        app.quit()

    tray.exit_requested.connect(on_exit)
    hud.exit_requested.connect(on_exit)
    app.aboutToQuit.connect(on_exit)

    # Voice capture, STT, and Kokoro TTS (default voice: af_sky) setup
    transcriber = None
    speech_active = threading.Event()
    active_voice = ["af_sky"]
    voice_enabled = [True]
    voice_init_error = [""]
    from lora.core.config import load_config as load_master_config
    master_config = load_master_config(Path(__file__).resolve().parents[2] / "config" / "lora.yaml")
    from lora.core.paths import project_path
    model_root = project_path(master_config.models_dir)
    headset_mode = master_config.voice.headset_mode

    def on_voice_changed(new_voice: str) -> None:
        active_voice[0] = new_voice or "af_sky"
        logger.info("Active TTS voice selected: %s", active_voice[0])

    hud.voice_changed.connect(on_voice_changed)
    hud.voice_toggled.connect(lambda enabled: voice_enabled.__setitem__(0, bool(enabled)))

    tts_engine = None
    if not args.no_audio:
        try:
            from lora.voice.tts.engine import TtsEngine
            from lora.voice.tts.worker import KokoroTtsWorkerProcess
            tts_worker_class = KokoroTtsWorkerProcess
            if os.environ.get("LORA_IN_PROCESS_SPEECH") != "1":
                from lora.voice.isolation import ProcessKokoro
                tts_worker_class = ProcessKokoro

            kokoro_dir = str(model_root / "kokoro")
            tts_worker = tts_worker_class(
                models_dir=kokoro_dir,
                default_voice=active_voice[0],
            )
            tts_engine = TtsEngine(worker=tts_worker)
            logger.info("TTS engine initialized (default voice: %s).", active_voice[0])
        except Exception as e:
            logger.warning("Could not initialize TTS engine: %s", e)

    # ---- speech: sentences are spoken while the reply is still being written ----
    tts_error_reported = threading.Event()

    def synth_sentence(sentence: str):
        if tts_engine is None or is_exiting.is_set():
            return None
        try:
            chunk = tts_engine.worker.synthesize(sentence, voice=active_voice[0])
        except Exception:
            if is_exiting.is_set():
                return None  # Shutdown intentionally interrupts the worker pipe.
            logger.exception("Speech output worker failed; keeping desktop alive")
            if not tts_error_reported.is_set():
                tts_error_reported.set()
                ui.post(lambda: hud.append_message("System", "My voice stopped, but text chat is still available. I'll restart the voice worker for the next reply."))
            return None
        if is_exiting.is_set():
            return None
        if chunk.error and not tts_error_reported.is_set():
            tts_error_reported.set()
            ui.post(lambda: hud.append_message("L.O.R.A", "My Kokoro voice isn't available. Run python scripts/download_models.py --kokoro-only; you can still chat with me here."))
        return chunk

    def play_chunk(chunk, stop_event: threading.Event) -> None:
        from lora.voice.playback import play_audio

        speech_active.set()
        try:
            play_audio(chunk.samples, chunk.sample_rate, stop_event,
                       lambda rms: ui.post(lambda lvl=min(1.0, rms * 8.0): (
                           hud.set_audio_level(lvl), widget.set_audio_level(lvl))),
                       is_exiting)
        finally:
            speech_cooldown_until[0] = time.monotonic() + 0.45

    def make_streamer(turn_done: threading.Event, is_sleep_turn: list[bool] | None = None, is_current=lambda: True):
        """A SpeechStreamer for one turn, or None when audio/voice is off."""
        if args.no_audio or not voice_enabled[0] or is_exiting.is_set():
            return None
        from lora.voice.speech_stream import SpeechStreamer

        def post_voice(fn) -> None:
            ui.post(lambda: fn() if is_current() and not is_exiting.is_set() else None)

        def _started() -> None:
            if not is_current() or is_exiting.is_set():
                return
            speech_active.set()
            if not (is_sleep_turn and is_sleep_turn[0]):
                post_voice(lambda: (hud.set_voice_state("speaking"), widget.set_state(WidgetState.SPEAKING))
                           if hud._current_voice_state not in ("sleep", "sleeping") else None)

        def _done(spoke: bool) -> None:
            if not is_current() or is_exiting.is_set():
                return
            speech_active.clear()
            speech_cooldown_until[0] = time.monotonic() + 0.45
            if not spoke and turn_done.is_set():
                logger.warning("Nothing was spoken (TTS produced no audio).")
            if turn_done.is_set():
                is_sleeping = (
                    (is_sleep_turn and is_sleep_turn[0])
                    or hud._current_voice_state in ("sleep", "sleeping")
                    or widget.state == WidgetState.SLEEPING
                )
                if is_sleeping:
                    post_voice(lambda: (
                        on_sleep_toggled(True, play_sound=False),
                        hud.set_audio_level(0.0),
                        widget.set_audio_level(0.0),
                    ))
                else:
                    post_voice(lambda: (
                        hud.set_voice_state("idle"),
                        widget.set_state(WidgetState.IDLE),
                        hud.set_audio_level(0.0),
                        widget.set_audio_level(0.0),
                        audio.play_cue(SoundCue.COMPLETE) if spoke else None,
                    ))

        return SpeechStreamer(synth_sentence, play_chunk, on_start=_started, on_done=_done)

    def stop_current_speech(update_ui: bool = True) -> None:
        streamer = current_streamer[0]
        if streamer is not None:
            streamer.stop()
        speech_active.clear()
        speech_cooldown_until[0] = time.monotonic() + 0.35
        if tts_engine is not None:
            tts_engine.interrupt()
        if sys.platform == "win32":
            try:
                import winsound

                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        if update_ui:
            ui.post(lambda: (
                hud.set_voice_state("idle"),
                widget.set_state(WidgetState.IDLE),
                hud.set_audio_level(0.0),
                widget.set_audio_level(0.0),
            ))

    hud.stop_audio_requested.connect(stop_current_speech)
    widget.exit_requested.connect(on_exit)

    history = ConversationHistory(max_turns=6)

    def _profile_text() -> str:
        try:
            from lora.storage.memory import MemoryStore

            return MemoryStore(profile_path=cfg_mgr.paths.user_profile_file).get_profile_markdown()
        except Exception:
            return ""

    def _make_registry():
        from lora.tools.registry import ToolRegistry

        return ToolRegistry(media_db=media_db)

    runtime = AgentRuntime(
        ollama_url=user_cfg.models.ollama_url,
        user_default_model=user_cfg.models.default_model,
        history=history,
        registry_factory=_make_registry,
        profile_fn=_profile_text,
        media_db=media_db,
        mcp_config_path=cfg_mgr.paths.mcp_file,
        skills_dir=cfg_mgr.paths.skills_dir,
        agents_dir=cfg_mgr.paths.agents_dir,
        notebook_db_path=cfg_mgr.paths.notebooks_dir / "notebooks.db",
    )
    logger.info("Laya decision router initialized (%s).", runtime.laya_router.backend)

    # Wire manual/auto Model, Skill, Voice, Verification, and Power callbacks between HUD and AgentRuntime
    def on_model_selected(model_name: str) -> None:
        msg = runtime.set_model(model_name)
        logger.info(msg)

    def on_skill_selected(skill_name: str) -> None:
        msg = runtime.set_skill(skill_name)
        logger.info(msg)

    hud.model_changed.connect(on_model_selected)
    hud.skill_changed.connect(on_skill_selected)
    runtime.on_model_changed = lambda active, models: ui.post(lambda a=active, m=models: hud.update_model_list(m, a))
    runtime.on_skill_changed = lambda sk: ui.post(lambda s=sk: hud.set_active_skill(s))
    runtime.on_voice_changed_cb = lambda vc: (
        active_voice.__setitem__(0, vc),
        ui.post(lambda v=vc: hud.set_active_voice(v)),
    )
    runtime.on_verification = lambda meta: ui.post(lambda m=dict(meta): hud.set_verification(m))

    def _on_emergency_stop(reason: str = "User stop command") -> None:
        from lora.safety.killswitch import KillSwitch

        logger.info("Task stop triggered: %s", reason)
        KillSwitch.get_global().trigger(reason)
        stop_current_speech(update_ui=False)
        if current_streamer[0] is not None:
            try:
                current_streamer[0].stop()
            except Exception:
                pass
        turns.cancel()
        ui.post(lambda: hud.append_message("System", "Stopped. Waiting for active work to release its resources."))
        def show_stopped() -> None:
            sleeping = hud._current_voice_state in ("sleep", "sleeping")
            hud.set_voice_state("sleeping" if sleeping else "idle")
            widget.set_state(WidgetState.SLEEPING if sleeping else WidgetState.IDLE)
        ui.post(show_stopped)

    def _handle_power_cmd(action: str) -> None:
        if action == "sleep":
            ui.post(lambda: on_sleep_toggled(True))
        elif action == "turn_off":
            ui.post(on_exit)
        elif action == "stop":
            _on_emergency_stop("Voice stop command")

    runtime.on_power_command = _handle_power_cmd
    widget.emergency_stop_requested.connect(lambda: _on_emergency_stop("Droplet widget emergency stop"))

    def _confirm_blocking(text: str) -> bool:
        """Ask the user (GUI thread dialog) and wait for the answer from the worker thread."""
        answer = {"ok": False}
        done = threading.Event()

        def _ask() -> None:
            from PySide6.QtWidgets import QMessageBox

            box = QMessageBox(hud)
            box.setWindowTitle("LORA")
            box.setText(text)
            box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            box.setDefaultButton(QMessageBox.StandardButton.No)
            answer["ok"] = box.exec() == QMessageBox.StandardButton.Yes
            done.set()

        ui.post(_ask)
        done.wait(90)
        return answer["ok"]

    async def _confirm(text: str) -> bool:
        return await asyncio.to_thread(_confirm_blocking, text)

    bg_loop = asyncio.new_event_loop()
    bg_thread = threading.Thread(target=bg_loop.run_forever, daemon=True, name="lora-async-loop")
    bg_thread.start()

    def _run_coro(coro, timeout: float = 1800.0, handle=None):
        return run_coroutine(bg_loop, coro, timeout, handle=handle)

    runtime.confirm = _confirm
    runtime.on_sources = lambda srcs: ui.post(lambda s=srcs: hud.set_sources(s))
    hud.media_updated.connect(runtime.on_sources_changed)

    def on_query_submitted(query: str, depth: str = "") -> None:
        """Runs on the GUI thread. Starts one assistant turn in the background."""
        query = (query or "").strip()
        depth_clean = (depth or "").strip().lower()
        if depth_clean in ("quick", "standard", "deep", "deep_with_refinement"):
            mode_override: str | None = depth_clean
        elif depth_clean in ("eli5", "simple"):
            mode_override = "standard"
        elif depth_clean in ("detailed", "expert"):
            mode_override = "deep"
        else:
            mode_override = None
        turn_attachments = list(getattr(hud, "pending_attachments", []) or [])
        hud.pending_attachments = []
        if not query and not turn_attachments:
            return
        from lora.agent.commands import parse_command
        from lora.safety.killswitch import KillSwitch

        incoming_cmd = parse_command(query) if not turn_attachments else None
        if incoming_cmd is not None and incoming_cmd.kind == "stop":
            _on_emergency_stop("User stop command")
            return

        if turns.busy:
            fast_cmd = incoming_cmd
            if fast_cmd is not None and fast_cmd.kind in ("open", "google", "site_search", "list_dir", "switch_model", "switch_voice", "switch_skill", "sleep"):
                def _fast_worker() -> None:
                    try:
                        if fast_cmd.kind == "sleep":
                            stop_current_speech(update_ui=False)
                        res = _run_coro(runtime._run_command(fast_cmd, query, lambda _s: None, lambda _t, _sp=True: None), timeout=25.0)
                        if res and res.text:
                            ui.post(lambda t=res.text: hud.append_message("L.O.R.A", t))
                    except Exception:
                        pass

                threading.Thread(target=_fast_worker, daemon=True).start()
                return
            hud.set_status("Still working. Press Ctrl+Esc to cancel before starting another request.")
            return


        handle = turns.begin()
        if handle is None:
            return
        # Only reset after the old asyncio task has completed cancellation.
        KillSwitch.get_global().reset()
        turn_is_sleep = [False]

        if incoming_cmd is not None and incoming_cmd.kind == "sleep":
            turn_is_sleep[0] = True
        elif hud._current_voice_state in ("sleep", "sleeping"):
            on_sleep_toggled(False)

        stop_current_speech(update_ui=False)
        hud.set_voice_state("thinking")
        widget.set_state(WidgetState.THINKING)
        hud.stream_begin(query, "")
        turn_done = threading.Event()

        def _worker() -> None:
            started = time.monotonic()
            def post_turn(fn) -> None:
                ui.post(lambda: fn() if turns.is_current(handle) and not is_exiting.is_set() else None)

            streamer = None
            last_push = [0.0]

            def on_status(text: str) -> None:
                post_turn(lambda t=text: hud.set_status(t))

            def on_text(full: str, speak: bool = True) -> None:
                if handle.cancelled.is_set() or not turns.is_current(handle):
                    return
                now = time.monotonic()
                if now - last_push[0] >= 0.06:
                    last_push[0] = now
                    post_turn(lambda t=full: hud.stream_update(t))
                if speak and streamer is not None:
                    streamer.feed(full)

            turn_timeout = 1200.0
            try:
                streamer = make_streamer(turn_done, is_sleep_turn=turn_is_sleep,
                                         is_current=lambda: turns.is_current(handle))
                current_streamer[0] = streamer
                default_timeout = "1800.0" if mode_override in ("deep", "deep_with_refinement") else "1200.0"
                turn_timeout = float(os.environ.get("LORA_TURN_TIMEOUT", default_timeout))
                result = _run_coro(
                    runtime.run_turn(
                        query,
                        on_status,
                        on_text,
                        mode=mode_override,
                        attachments=turn_attachments or None,
                        should_abort=lambda: "User stopped Task" if handle.cancelled.is_set() else None,
                    ),
                    timeout=turn_timeout,
                    handle=handle,
                )
            except Exception as exc:
                import concurrent.futures
                if isinstance(exc, concurrent.futures.CancelledError) or (handle.cancelled.is_set() and not isinstance(exc, TimeoutError)):
                    logger.info("Turn cancelled by user stop: %s", exc)
                    result = TurnResult(text="User stopped Task.", kind="command")
                elif isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
                    logger.warning("Turn exceeded timeout limit (%.1fs)", turn_timeout)
                    result = TurnResult(
                        text="The research task exceeded the allowed time limit. The background analysis took longer than expected—try breaking down the request into specific questions.",
                        kind="error",
                        failure="TimeoutError",
                    )
                else:
                    logger.exception("Turn crashed")
                    result = TurnResult(text="Something went wrong on my side. Try again in a moment.", kind="error", failure=str(exc))
            finally:
                # A cancelled Future can finish before native work unwinds.
                # The coordinator remains busy until the actual task is done.
                turn_done.set()
                handle.finish_if_unsubmitted()
                if handle.cancelled.is_set() and streamer is not None:
                    streamer.stop()

            logger.info(
                "Turn finished: kind=%s in %.1fs, sources=%d%s",
                result.kind, time.monotonic() - started, len(result.sources),
                f", failure={result.failure}" if result.failure else "",
            )

            if (getattr(result, "meta", None) and result.meta.get("power_action") == "sleep") or turn_is_sleep[0]:
                turn_is_sleep[0] = True

            def _post() -> None:
                if not turns.owns(handle) or is_exiting.is_set():
                    return
                # Sleep may have been requested while this task was finishing.
                if hud._current_voice_state in ("sleep", "sleeping"):
                    if streamer is not None:
                        streamer.stop()
                    return
                hud.present_answer(
                    query, result.text, sources=result.sources, confidence=0.0, label="", streamed=True,
                    meta=result.meta,
                )
                turn_done.set()
                if turn_is_sleep[0]:
                    if streamer is None:
                        on_sleep_toggled(True)
                    else:
                        spoken = result.spoken if result.spoken is not None else result.text
                        if result.spoken is not None:
                            streamer.feed(spoken)
                        streamer.finish(spoken)
                        if not streamer.has_pending:
                            on_sleep_toggled(True)
                    return
                if streamer is None:
                    hud.set_voice_state("idle")
                    widget.set_state(WidgetState.IDLE)
                    return
                spoken = result.spoken if result.spoken is not None else result.text
                if result.spoken is not None:
                    streamer.feed(spoken)
                streamer.finish(spoken)
                is_sp = streamer.has_pending
                hud.set_voice_state("speaking" if is_sp else "idle")
                widget.set_state(WidgetState.SPEAKING if is_sp else WidgetState.IDLE)

            ui.post(_post)

        threading.Thread(target=_worker, daemon=True).start()

    hud.query_submitted.connect(on_query_submitted)
    hud.rewrite_requested.connect(lambda dir, depth: on_query_submitted(f"Explain that at {depth or 'detailed'} depth", depth))

    def _preload_voice() -> None:
        """Load the Kokoro CPU voice and Faster-Whisper STT model at startup so turn 1 is instant."""
        if args.no_audio or is_exiting.is_set():
            return
        if transcriber is not None:
            try:
                transcriber.preload()
            except Exception as exc:
                if not is_exiting.is_set():
                    logger.warning("STT preload failed: %s", exc)
        if tts_engine is None or is_exiting.is_set():
            return
        try:
            ok = tts_engine.worker.preload()
            diag = tts_engine.worker.diagnosis()
            (logger.info if ok else logger.warning)("Voice: %s %s", diag["summary"], diag.get("fix", "") if not ok else "")
        except Exception as exc:
            if not is_exiting.is_set():
                logger.warning("Voice preload failed: %s", exc)

    def _warm_up_model() -> None:
        """Load the model in the background at startup and sync installed models to the UI dropdown."""
        if is_exiting.is_set():
            return
        if os.environ.get("LORA_ENABLE_MCP", "0") == "1":
            threading.Thread(target=runtime.start_mcp, daemon=True, name="lora-mcp").start()
        async def _go() -> None:
            from lora.models.ollama_client import OllamaClient
            from lora.models.selection import is_excluded_model, plan_chat

            client = OllamaClient(base_url=user_cfg.models.ollama_url)
            plan = None
            for _ in range(30):
                if is_exiting.is_set():
                    return
                try:
                    installed = await client.installed_models()
                    if installed:
                        visible = [m for m in installed if not is_excluded_model(m)] or installed
                        ui.post(lambda m=visible: hud.update_model_list(m, runtime.manual_model or "auto"))
                    plan = await plan_chat(client, user_default=user_cfg.models.default_model)
                    break
                except Exception:
                    await asyncio.sleep(2.0)
            if plan is None:
                logger.warning("Warm-up skipped: no model available from Ollama.")
                return
            ready = await client.warm_up(plan.model, options=plan.options, keep_alive=plan.keep_alive)
            logger.info("Model %s %s", plan.model, "is loaded and ready" if ready else "warm-up failed")

        try:
            _run_coro(_go(), timeout=240.0)
        except Exception as exc:
            logger.warning("Model warm-up failed: %s", exc)
        if not is_exiting.is_set():
            _preload_voice()

    if not args.no_audio:
        try:
            import numpy as np

            from lora.voice.capture import AudioCapture
            from lora.voice.stt import FasterWhisperTranscriber
            from lora.voice.vad import SpeechState, VoiceActivityDetector
            from lora.voice.wakeword import WakeWordDetector
            transcriber_class = FasterWhisperTranscriber
            vad_class = VoiceActivityDetector
            if os.environ.get("LORA_IN_PROCESS_SPEECH") != "1":
                from lora.voice.isolation import ProcessTranscriber, ProcessVad
                transcriber_class = ProcessTranscriber
                vad_class = ProcessVad

            capture = AudioCapture(sample_rate=16000)
            stt_config = master_config.voice.stt
            transcriber = transcriber_class(
                model_size_or_path=stt_config.model, device="cpu", compute_type=stt_config.compute,
                initial_prompt=stt_config.initial_prompt or None, lazy_load=True,
            )
            speaker_config = master_config.voice.wake.speaker_verifier
            enabled_override = os.environ.get("LORA_SPEAKER_VERIFY_ENABLED")
            speaker_verifier_enabled = speaker_config.enabled if enabled_override is None else enabled_override.lower() in ("1", "true", "yes")
            from lora.voice.wakeword import SpeakerVerifier
            verifier = SpeakerVerifier(enabled=speaker_verifier_enabled,
                                       profile_path=speaker_config.model, threshold=speaker_config.threshold)
            if verifier.enabled and not verifier.reference_embeddings:
                hud.append_message("System", "No valid owner voice profile was found. Run lora enroll for owner filtering; wake-phrase listening is still available.")
            wakeword = WakeWordDetector(
                wake_phrase="Awaken LORA",
                chat_phrase="Hey LORA",
                model_path=str(project_path(master_config.voice.wake.model)) if project_path(master_config.voice.wake.model).is_file() else None,
                speaker_verifier=verifier,
                transcriber=transcriber,
                engine=master_config.voice.wake.engine,
                listen_while_sleeping=not master_config.voice.close_mic_on_sleep,
            )
            vad_threshold = float(os.environ.get("LORA_VAD_THRESHOLD", "0.35"))
            vad = vad_class(
                sample_rate=16000,
                threshold=vad_threshold,
                silence_duration_seconds=0.60,
                min_speech_duration_seconds=0.10,
                onnx_model_path="silero",
                preroll_chunks=16,
            )

            is_transcribing = threading.Event()
            is_ptt_active = threading.Event()
            ptt_start_time = [0.0]
            ptt_chunks: list[np.ndarray] = []
            arm_seconds = 12.0
            stt_error_shown = [False]
            vad_error_shown = [False]

            def _stt_failed(ex: Exception) -> None:
                logger.warning("Speech recognition error: %s", ex, exc_info=True)
                if not stt_error_shown[0]:
                    stt_error_shown[0] = True
                    msg = (
                        f"Speech recognition failed: {type(ex).__name__}: {ex}. "
                        "The first run downloads the Whisper model, so it needs internet once."
                    )
                    ui.post(lambda: hud.append_message("L.O.R.A", msg))

            def _is_low_stt_confidence(stt_res: object) -> bool:
                if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
                    return False
                low_flag = getattr(stt_res, "low_confidence", False)
                if low_flag is True:
                    return True
                avg_lp = getattr(stt_res, "avg_logprob", 0.0)
                if isinstance(avg_lp, (int, float)) and not isinstance(avg_lp, bool):
                    return float(avg_lp) < -0.70
                return False

            def _submit_voice_query(q: str, low_confidence: bool = False, generation: int | None = None) -> None:
                """GUI thread."""
                if hud._current_voice_state in ("sleep", "sleeping"):
                    return  # Drop transcripts queued before sleep closed the microphone.
                if generation is not None and generation != voice_generation[0]:
                    return
                hud.append_message("You", q)
                hud.set_input_text(q)
                if low_confidence:
                    # Never load/run Laya on the GUI thread while checking an
                    # uncertain transcript. Ask directly and leave the UI live.
                    from PySide6.QtWidgets import QMessageBox

                    ans = QMessageBox.question(
                        hud,
                        "Confirm Voice Command",
                        f"Low audio confidence - did you mean:\n\n\"{q}\"?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if ans != QMessageBox.StandardButton.Yes:
                        hud.append_message("L.O.R.A", "Cancelled low-confidence voice command.")
                        hud.set_voice_state("idle")
                        return
                on_query_submitted(q, getattr(hud, "_depth_mode", ""))

            def _trigger_transcription(audio_samples: np.ndarray) -> None:
                with transcription_lock:
                    if is_transcribing.is_set():
                        logger.info("Push-to-talk ignored: still transcribing the previous clip.")
                        return
                    is_transcribing.set()
                    generation = voice_generation[0]
                logger.info("Transcribing %.1fs of push-to-talk audio.", len(audio_samples) / 16000)
                ui.post(lambda: hud.set_voice_state("thinking"))

                def _worker() -> None:
                    submitted = False
                    try:
                        peak = float(np.max(np.abs(audio_samples))) if len(audio_samples) else 0.0
                        if len(audio_samples) < 2400 or peak < 0.004:
                            logger.info("Push-to-talk audio was silent (peak=%.4f).", peak)
                            ui.post(lambda: hud.append_message(
                                "L.O.R.A",
                                "I couldn't hear anything. Check that the right microphone is selected and not muted.",
                            ))
                            return
                        res = transcriber.transcribe(audio_samples)
                        if generation != voice_generation[0]:
                            return
                        query_text = res.text.strip()
                        if query_text:
                            logger.info("Transcribed voice input: '%s' (conf=%.2f)", query_text, res.confidence)
                            submitted = True
                            is_low = _is_low_stt_confidence(res)
                            ui.post(lambda q=query_text, lc=is_low: _submit_voice_query(q, low_confidence=lc, generation=generation))
                        else:
                            logger.info("No speech detected in audio segment.")
                            ui.post(lambda: hud.append_message("L.O.R.A", "I didn't catch that. Try again?"))
                    except Exception as ex:
                        _stt_failed(ex)
                    finally:
                        is_transcribing.clear()
                        if not submitted and not speech_active.is_set():
                            ui.post(lambda: hud.set_voice_state("idle") if generation == voice_generation[0] else None)

                threading.Thread(target=_worker, daemon=True).start()

            def _process_utterance_async(audio_clip: np.ndarray) -> None:
                """Hands-free path: transcribe ONCE, then decide wake word vs query vs ignore."""
                with transcription_lock:
                    if is_ptt_active.is_set():
                        return
                    if turns.busy:
                        deferred_utterances.append((time.monotonic(), audio_clip))
                        return
                    if is_transcribing.is_set():
                        pending_utterances.append(audio_clip)
                        return
                    is_transcribing.set()
                    generation = voice_generation[0]

                def _bg() -> None:
                    submitted = False
                    try:
                        current_st = hud._current_voice_state
                        sleeping = current_st in ("sleep", "sleeping")
                        if sleeping and master_config.voice.close_mic_on_sleep:
                            return
                        if not verifier.verify(audio_clip)[0]:
                            return  # Reject non-owner wake AND following prompt before STT.
                        if time.monotonic() >= armed_until[0] and wakeword.engine == "openwakeword":
                            if wakeword._oww_model is not None:
                                if not wakeword.process_audio(audio_clip, current_state=current_st).detected:
                                    return
                        res = transcriber.transcribe(audio_clip)
                        if generation != voice_generation[0]:
                            return
                        text = res.text.strip()
                        if not text:
                            return
                        logger.debug("Voice candidate (state=%s, avg_logprob=%.2f)", current_st, res.avg_logprob)

                        if not sleeping and time.monotonic() < armed_until[0]:
                            armed_until[0] = 0.0
                            submitted = True
                            is_low = _is_low_stt_confidence(res)
                            ui.post(lambda q=text, lc=is_low: _submit_voice_query(q, low_confidence=lc, generation=generation))
                            return

                        wake = wakeword.evaluate_text(text, audio_clip, current_state=current_st)
                        if not wake.detected:
                            if wake.score > 0 and not wake.speaker_verified:
                                logger.warning(
                                    "Wake word matched '%s' but rejected by speaker verification (similarity=%.2f < threshold=%.2f). Set LORA_SPEAKER_VERIFY_ENABLED=0 to bypass.",
                                    wake.phrase,
                                    wake.speaker_similarity,
                                    wakeword.speaker_verifier.threshold,
                                )
                            return
                        if sleeping:
                            def resume_voice(q=wake.extracted_query, lc=_is_low_stt_confidence(res)):
                                on_sleep_toggled(False)
                                if q:
                                    _submit_voice_query(q, low_confidence=lc, generation=voice_generation[0])
                                else:
                                    armed_until[0] = time.monotonic() + arm_seconds
                                    hud.set_voice_state("listening")
                            submitted = True
                            ui.post(resume_voice)
                            return
                        speech_cooldown_until[0] = time.monotonic() + 0.35
                        is_low = _is_low_stt_confidence(res)
                        if wake.extracted_query:
                            from lora.agent.commands import parse_command

                            maybe_cmd = parse_command(wake.extracted_query)
                            if maybe_cmd is None or maybe_cmd.kind != "sleep":
                                audio.play_cue(SoundCue.WAKE)
                            logger.info("Wake word with trailing query: '%s'", wake.extracted_query)
                            submitted = True
                            ui.post(lambda q=wake.extracted_query, lc=is_low: _submit_voice_query(q, low_confidence=lc, generation=generation))
                        else:
                            audio.play_cue(SoundCue.WAKE)
                            logger.info("Wake word detected. Listening for the query...")
                            with transcription_lock:
                                if generation != voice_generation[0]:
                                    return
                                armed_until[0] = time.monotonic() + arm_seconds
                            ui.post(lambda: hud.set_voice_state("listening") if generation == voice_generation[0] else None)
                    except Exception as ex:
                        _stt_failed(ex)
                    finally:
                        with transcription_lock:
                            is_transcribing.clear()
                            if submitted or hud._current_voice_state in ("sleep", "sleeping"):
                                pending_utterances.clear()
                            following = pending_utterances.popleft() if pending_utterances else None
                        if following is not None:
                            _process_utterance_async(following)

                threading.Thread(target=_bg, daemon=True).start()

            _last_ui_update_time = [0.0]

            def _on_mic_chunk(chunk: np.ndarray) -> None:
                """Runs on the sounddevice thread: no Qt calls except via ui.post."""
                if len(chunk) == 0:
                    return

                rms = float(np.sqrt(np.mean(chunk**2)))
                level = min(1.0, rms * 8.0)
                now = time.monotonic()
                if now - _last_ui_update_time[0] >= 0.04:
                    _last_ui_update_time[0] = now
                    ui.post(lambda lvl=level: (hud.set_audio_level(lvl), widget.set_audio_level(lvl)))

                if armed_until[0] and now > armed_until[0]:
                    armed_until[0] = 0.0
                    if hud._current_voice_state == "listening" and not is_ptt_active.is_set():
                        ui.post(lambda: hud.set_voice_state("idle"))

                if is_ptt_active.is_set():
                    ptt_chunks.append(chunk.copy())
                    return

                # v1.5 Plan §9.2: When headset_mode is False (laptop speakers), ignore mic input during TTS
                # and during the post-TTS room-echo decay window so L.O.R.A never interrupts or transcribes itself.
                if speech_active.is_set() or now < speech_cooldown_until[0]:
                    if headset_mode and speech_active.is_set() and rms > 0.18:
                        stop_current_speech()
                    vad.reset()
                    return

                if hud._current_voice_state in ("sleep", "sleeping") and master_config.voice.close_mic_on_sleep:
                    vad.reset()
                    return

                try:
                    vad_state, _prob = vad.process_chunk(chunk)
                except Exception:
                    if is_exiting.is_set():
                        return
                    logger.exception("Voice detector failed; capture remains open")
                    if not vad_error_shown[0]:
                        vad_error_shown[0] = True
                        ui.post(lambda: hud.append_message("System", "Voice detection stopped. I'll restart it on the next audio block; you can still type here."))
                    return
                if vad_state == SpeechState.ENDED:
                    utterance = vad.get_and_reset_utterance()
                    if len(utterance) >= 2400:  # >= 150ms (includes 500ms pre-roll + "Hey LORA")
                        _process_utterance_async(utterance)

            capture.register_chunk_callback(_on_mic_chunk)
            capture.start()

            def _drain_deferred_voice() -> None:
                streamer = current_streamer[0]
                if turns.busy or speech_active.is_set() or (streamer is not None and streamer.has_pending):
                    return
                if time.monotonic() < speech_cooldown_until[0] or is_exiting.is_set():
                    return
                with transcription_lock:
                    if is_transcribing.is_set() or not deferred_utterances:
                        return
                    recorded_at, clip = deferred_utterances.popleft()
                if time.monotonic() - recorded_at < 30:
                    _process_utterance_async(clip)

            deferred_voice_timer = QTimer(hud)
            deferred_voice_timer.setInterval(100)
            deferred_voice_timer.timeout.connect(_drain_deferred_voice)
            deferred_voice_timer.start()
            if not capture.is_streaming:
                logger.warning("Microphone is not running: %s", capture.status_text())

            def _on_ptt_down() -> None:
                if hud._current_voice_state in ("thinking", "think", "verify"):
                    return
                if hud._current_voice_state in ("sleep", "sleeping"):
                    on_sleep_toggled(False)
                if is_ptt_active.is_set():
                    return
                armed_until[0] = 0.0
                ptt_chunks.clear()
                is_ptt_active.set()
                ptt_start_time[0] = time.monotonic()
                stop_current_speech(update_ui=False)
                vad.reset()
                ui.post(lambda: hud.set_voice_state("listening"))
                if capture:
                    capture.buffer.clear()
                    if not capture.is_streaming:
                        capture.start()

            def _on_ptt_up() -> None:
                if not is_ptt_active.is_set():
                    if hud._current_voice_state in ("listening", "listen"):
                        ui.post(lambda: hud.set_voice_state("idle"))
                    return
                is_ptt_active.clear()
                dur = time.monotonic() - ptt_start_time[0]
                logger.info("Push-to-talk released after %.1fs (mic streaming: %s)", dur, capture.is_streaming)
                if not capture.is_streaming:
                    reason = capture.status_text()
                    ui.post(lambda: (
                        hud.append_message("L.O.R.A", f"My microphone isn't running: {reason}"),
                        hud.set_voice_state("idle"),
                    ))
                elif dur >= 0.20:
                    if ptt_chunks:
                        audio_samples = np.concatenate(ptt_chunks)
                        ptt_chunks.clear()
                    else:
                        audio_samples = capture.buffer.read_last_n_seconds(min(15.0, dur + 0.3))
                    _trigger_transcription(audio_samples)
                else:
                    ptt_chunks.clear()
                    ui.post(lambda: hud.set_voice_state("idle"))

            hud.ptt_pressed.connect(_on_ptt_down)
            hud.ptt_released.connect(_on_ptt_up)

            try:
                from lora.voice.hotkey import PushToTalkListener

                hotkey = PushToTalkListener(
                    on_activate=lambda: ui.post(_on_ptt_down),
                    on_deactivate=lambda: ui.post(_on_ptt_up),
                    on_killswitch=lambda: ui.post(lambda: _on_emergency_stop("Hotkey pressed (Ctrl+Esc)")),
                )
                hotkey.on_sleep_toggle = lambda: ui.post(lambda: on_sleep_toggled(hud._current_voice_state not in ("sleep", "sleeping")))
                hotkey.start()
            except Exception as e:
                logger.warning("Global push-to-talk hotkey unavailable: %s", e)
        except Exception as e:
            voice_init_error[0] = f"{type(e).__name__}: {e}"
            logger.warning("Voice capture could not be initialized: %s", e, exc_info=True)

    def _refresh_mic_status() -> None:
        if args.no_audio:
            hud.set_mic_status(False, "disabled (--no-audio)")
        elif capture is None:
            hud.set_mic_status(False, f"voice pipeline failed to start ({voice_init_error[0] or 'see log'})")
        elif hud._current_voice_state in ("sleep", "sleeping") and master_config.voice.close_mic_on_sleep:
            hud.set_mic_status(False, "Microphone closed (sleep mode)")
        elif speech_active.is_set() and not headset_mode:
            hud.set_mic_status(True, "Microphone on; recognition waits for speaker playback")
        else:
            age = capture.seconds_since_audio()
            if not capture.is_streaming or (age is not None and age > 4.0):
                # Auto-heal stalled Windows audio stream after device switch or driver hiccup
                try:
                    logger.info("Microphone stream stalled (age=%s); restarting capture stream...", age)
                    capture.stop()
                    capture.start()
                except Exception as exc:
                    logger.debug("Mic stream auto-restart failed: %s", exc)
            if not capture.is_streaming:
                hud.set_mic_status(False, capture.status_text())
            elif age is None or age > 4.0:
                hud.set_mic_status(False, f"{capture.device_name}: no audio arriving")
            else:
                hud.set_mic_status(True, capture.device_name)

    def _diag_extras() -> dict:
        extras: dict = {}
        if capture is not None:
            extras["mic"] = (capture.is_streaming and (capture.seconds_since_audio() or 99) < 3.0, capture.status_text())
        if tts_engine is not None:
            try:
                extras["tts"] = tts_engine.worker.diagnosis()
            except Exception as exc:
                extras["tts"] = {"ok": False, "summary": f"voice check failed ({exc})"}
        return extras

    runtime.diag_extras = _diag_extras

    mic_timer = QTimer(hud)
    mic_timer.setInterval(2000)
    mic_timer.timeout.connect(_refresh_mic_status)
    mic_timer.start()

    from lora.agent.jobs import JobWorker
    from lora.gateway.queue import JobQueue
    from lora.gateway.ws import ConnectionManager

    job_queue = JobQueue()
    ws_manager = ConnectionManager()
    mobile_ws_manager = ConnectionManager()
    gateway_server = start_gateway_thread(
        host=args.host,
        port=args.port,
        queue=job_queue,
        media_db=media_db,
        connection_manager=ws_manager,
        notebook_store=NotebookStore(cfg_mgr.paths.notebooks_dir / "notebooks.db"),
    )

    def _make_job_runtime(origin: str) -> AgentRuntime:
        return AgentRuntime(
            ollama_url=user_cfg.models.ollama_url,
            user_default_model=user_cfg.models.default_model,
            history=ConversationHistory(max_turns=4),
            registry_factory=_make_registry,
            profile_fn=_profile_text if origin == "desktop" else (lambda: ""),
            media_db=media_db if origin == "desktop" else None,
            embedding_engine=media_db.embedding_engine,
            allow_actions=(origin == "desktop"),
            notebook_db_path=cfg_mgr.paths.notebooks_dir / "notebooks.db",
            laya_router=runtime.laya_router,
        )

    async def _broadcast_by_origin(event: dict) -> None:
        origin = str(event.get("origin", "mobile")).lower()
        if origin == "desktop":
            await ws_manager.broadcast(event)
        else:
            await mobile_ws_manager.broadcast(event)

    job_worker = JobWorker(
        job_queue,
        _make_job_runtime,
        desktop_busy=lambda: turns.busy,
        broadcaster=_broadcast_by_origin,
    )
    job_worker.start()

    mobile_server_ref: list[uvicorn.Server | None] = [None]
    mobile_url_ref: list[str] = [""]

    def _ensure_mobile_gateway(bind_str: str | None = None) -> str:
        if mobile_url_ref[0]:
            return mobile_url_ref[0]
        import secrets as _secrets

        from lora.gateway.auth import TokenValidator
        from lora.gateway.qr import generate_pairing_qr_terminal, get_pairing_url

        lan_ip = _detect_lan_ip()
        target = bind_str or getattr(args, "mobile_bind", None) or f"{lan_ip}:8766"
        m_host, _, m_port = target.rpartition(":")
        m_port_int = int(m_port or 8766)
        bind_host = m_host or lan_ip
        display_ip = lan_ip if bind_host in ("0.0.0.0", "") else bind_host
        token_file = Path(cfg_mgr.paths.user_profile_file).parent / "mobile_token.txt"
        token = os.environ.get("LORA_MOBILE_TOKEN") or (
            token_file.read_text(encoding="utf-8").strip() if token_file.exists() else _secrets.token_urlsafe(32)
        )
        token_file.write_text(token, encoding="utf-8")
        srv = start_gateway_thread(
            bind_host,
            m_port_int,
            queue=job_queue,
            origin="mobile",
            token_validator=TokenValidator(mobile_token=token),
            media_db=media_db,
            connection_manager=mobile_ws_manager,
            notebook_store=NotebookStore(cfg_mgr.paths.notebooks_dir / "notebooks.db"),
        )
        if srv is not None:
            mobile_server_ref[0] = srv
        url = get_pairing_url(display_ip, m_port_int, token)
        mobile_url_ref[0] = url
        try:
            from lora.core.logging import scrub_sensitive_data

            qr_ascii = generate_pairing_qr_terminal(display_ip, m_port_int, token)
            logger.info("Mobile gateway on %s\n%s", scrub_sensitive_data(url), qr_ascii)
        except Exception:
            from lora.core.logging import scrub_sensitive_data

            logger.info("Mobile gateway on %s (token saved in %s)", scrub_sensitive_data(url), token_file)
        return url

    try:
        _ensure_mobile_gateway(getattr(args, "mobile_bind", None))
    except Exception as exc:
        logger.warning("Could not start the mobile gateway: %s", exc)

    def _on_show_mobile_qr() -> None:
        try:
            from lora.gateway.qr import generate_pairing_qr_terminal

            url = _ensure_mobile_gateway()
            host_part = url.split("//", 1)[-1].split("/")[0]
            ip_str, _, port_str = host_part.rpartition(":")
            token_str = url.split("token=", 1)[-1]
            qr_text = generate_pairing_qr_terminal(ip_str, int(port_str or 8766), token_str)
            body = (
                f"### Mobile LAN Companion Pairing\n"
                f"Open this URL on your phone (connected to the same Wi-Fi/LAN) or scan the QR code below:\n\n"
                f"**[{url}]({url})**\n\n"
                f"```\n{qr_text}\n```"
            )
            hud.present_answer("Pair Mobile (QR)", body, streamed=False)
        except Exception as exc:
            hud.append_message("System", f"Could not generate mobile pairing QR: {exc}")

    hud.mobile_qr_requested.connect(_on_show_mobile_qr)

    # Show holographic HUD centered and Ambient Droplet Widget snapped to bottom-right corner
    screen_geo = app.primaryScreen().geometry()
    hud.move(
        max(0, (screen_geo.width() - hud.width()) // 2),
        max(0, (screen_geo.height() - hud.height()) // 2),
    )
    snap_x, snap_y = widget.calculate_snap_position(
        screen_geo.width() - widget.width() - 24,
        screen_geo.height() - widget.height() - 24,
        screen_geo.width(),
        screen_geo.height(),
        screen_geo.left(),
        screen_geo.top(),
    )
    widget.move(snap_x, snap_y)
    hud.show()
    widget.show()
    tray.show()
    speech_cooldown_until[0] = time.monotonic() + 0.8
    audio.play_cue(SoundCue.WAKE)
    # Defer background model & voice warm-up slightly so the HUD WebGL window opens smoothly without GIL stutter
    QTimer.singleShot(250, lambda: threading.Thread(target=_warm_up_model, daemon=True).start())
    logger.info("L.O.R.A Desktop Platform initialized successfully.")

    from lora.safety.lifecycle import consume_shutdown_request
    started_at = time.time()
    shutdown_timer = QTimer()
    shutdown_timer.timeout.connect(lambda: on_exit() if consume_shutdown_request(cfg_mgr.paths.base_dir, started_at) else None)
    shutdown_timer.start(500)
    exit_code = app.exec()

    # Clean shutdown
    on_exit()
    job_worker.stop()
    job_worker.join(timeout=10.0)
    if job_worker.is_alive():
        logger.warning("Job worker is still releasing resources during shutdown")
    from lora.agent.turns import drain_pending_tasks

    loop_drained = False
    try:
        _run_coro(drain_pending_tasks(), timeout=10.0)
        loop_drained = True
    except Exception:
        logger.warning("Background work is still releasing resources during shutdown", exc_info=True)
    if runtime._registry is not None:
        try:
            _run_coro(runtime._registry.close(), timeout=10.0)
        except Exception:
            pass
    if loop_drained:
        bg_loop.call_soon_threadsafe(bg_loop.stop)
        bg_thread.join(timeout=2.0)
        if not bg_thread.is_alive():
            bg_loop.close()
    if tts_engine is not None:
        try:
            tts_engine.worker.shutdown()
        except Exception:
            pass
    streamer = current_streamer[0]
    if streamer is not None:
        streamer.wait(timeout=2.0)
    for speech_worker in (transcriber, vad):
        if speech_worker is not None and hasattr(speech_worker, "shutdown"):
            speech_worker.shutdown()
    if runtime.mcp_hub is not None:
        runtime.mcp_hub.close()
    for server in (gateway_server, mobile_server_ref[0]):
        if server is not None:
            server.should_exit = True
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if getattr(args, "subcommand", None) in {"stop", "kill"}:
        from lora.safety.lifecycle import request_shutdown

        request_shutdown(getattr(args, "config_dir", None))
        print("LORA shutdown requested.")
        return 0

    if getattr(args, "subcommand", None) == "net-check":
        import json
        import socket
        import sys
        import urllib.request

        if sys.platform == "win32":
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

        print("\n[NET-CHECK] L.O.R.A Network & Service Diagnostic Check")
        print("=" * 55)
        lan_ip = _detect_lan_ip()
        print(f"  * Private LAN IP:        {lan_ip}")
        ollama_url = getattr(args, "ollama_url", "http://127.0.0.1:11434")
        try:
            req = urllib.request.Request(f"{ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    models = [m.get("name") for m in data.get("models", [])]
                    print(f"  * Ollama Service:        ONLINE ({len(models)} models available)")
                    for m in models[:5]:
                        print(f"      - {m}")
                    if len(models) > 5:
                        print(f"      ... and {len(models) - 5} more")
                else:
                    print(f"  * Ollama Service:        HTTP {resp.status}")
        except Exception as exc:
            print(f"  * Ollama Service:        OFFLINE ({exc})")

        for port in (8765, 8766):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                res = s.connect_ex(("127.0.0.1", port))
                if res == 0:
                    print(f"  * Port {port}:              IN USE / BOUND")
                else:
                    print(f"  * Port {port}:              AVAILABLE")

        print("\n  [Isolation & Firewall Audit]")
        try:
            from lora.tools.net_check import run_net_check

            user_mobile_bind = (
                getattr(args, "mobile_bind", None)
                or (f"{lan_ip}:8765" if lan_ip != "127.0.0.1" else "192.168.50.10:8765")
            )
            audit = run_net_check(mobile_bind=user_mobile_bind)
            for c in audit.checks:
                status_tag = "[PASS]" if c.passed else "[WARN]"
                print(f"  {status_tag:7} {c.name}: {c.details}")
        except Exception as exc:
            print(f"  [WARN]  Isolation audit skipped: {exc}")

        print("=" * 55 + "\n")
        return 0

    if getattr(args, "subcommand", None) == "enroll":
        from lora.voice.enroll import EnrollmentSession

        session = EnrollmentSession(samples_needed=getattr(args, "samples", 5))
        session.run_interactive(speaker_name=getattr(args, "speaker", "owner"))
        return 0

    from lora.core.logging import setup_logging
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    setup_logging(log_dir=log_dir)
    from lora.core.diagnostics import enable_fault_reporting
    enable_fault_reporting(log_dir)

    cfg_mgr, user_cfg = init_platform(base_dir=getattr(args, "config_dir", None))

    if getattr(args, "offline", False) or getattr(user_cfg.security, "offline_only", False):
        os.environ["LORA_OFFLINE"] = "1"
        os.environ["LORA_VERIFY_OFFLINE"] = "1"
        logger.info("Offline mode enabled (--offline / security.offline_only).")

    if args.headless:
        from lora.agent.jobs import JobWorker
        from lora.agent.runtime import AgentRuntime
        from lora.dialogue import ConversationHistory
        from lora.gateway.queue import JobQueue
        from lora.gateway.ws import ConnectionManager
        from lora.grounding.media_vector_db import MediaVectorDB
        from lora.safety.lifecycle import consume_shutdown_request
        from lora.tools.registry import ToolRegistry

        logger.info("Starting L.O.R.A in headless daemon mode...")
        h_queue = JobQueue()
        h_ws = ConnectionManager()
        h_media = MediaVectorDB()
        from lora.models.laya_router import LayaDecisionRouter
        h_router = LayaDecisionRouter(lazy_load=True)
        h_worker = JobWorker(
            h_queue,
            lambda origin: AgentRuntime(
                ollama_url=user_cfg.models.ollama_url,
                user_default_model=user_cfg.models.default_model,
                history=ConversationHistory(max_turns=4),
                registry_factory=lambda: ToolRegistry(media_db=h_media),
                media_db=h_media if origin == "desktop" else None,
                embedding_engine=h_media.embedding_engine,
                allow_actions=(origin == "desktop"),
                notebook_db_path=cfg_mgr.paths.notebooks_dir / "notebooks.db",
                laya_router=h_router,
            ),
            broadcaster=h_ws.broadcast,
        )
        h_worker.start()
        server = start_gateway_thread(
            host=args.host,
            port=args.port,
            queue=h_queue,
            media_db=h_media,
            connection_manager=h_ws,
            notebook_store=NotebookStore(cfg_mgr.paths.notebooks_dir / "notebooks.db"),
        )
        started_at = time.time()
        try:
            while server and not server.should_exit:
                if consume_shutdown_request(cfg_mgr.paths.base_dir, started_at):
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Shutting down L.O.R.A headless daemon...")
        finally:
            h_worker.stop()
            if server is not None:
                server.should_exit = True
        return 0

    return run_desktop_app(cfg_mgr, user_cfg, args)


cli = main


if __name__ == "__main__":
    sys.exit(main())
