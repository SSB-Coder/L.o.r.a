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

    from lora.voice.desktop_output import create_speech_output

    synth_sentence, play_chunk, make_streamer, stop_current_speech = create_speech_output(
        sound_cue=SoundCue,
        widget_state=WidgetState,
        active_voice=active_voice,
        args=args,
        audio=audio,
        current_streamer=current_streamer,
        hud=hud,
        is_exiting=is_exiting,
        on_sleep_toggled=on_sleep_toggled,
        speech_active=speech_active,
        speech_cooldown_until=speech_cooldown_until,
        tts_engine=tts_engine,
        tts_error_reported=tts_error_reported,
        ui=ui,
        voice_enabled=voice_enabled,
        widget=widget,
    )

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

    from lora.voice.desktop_input import start_voice_input

    capture, transcriber, vad, hotkey = start_voice_input(
        timer_class=QTimer,
        sound_cue=SoundCue,
        _on_emergency_stop=_on_emergency_stop,
        args=args,
        armed_until=armed_until,
        audio=audio,
        current_streamer=current_streamer,
        deferred_utterances=deferred_utterances,
        headset_mode=headset_mode,
        hud=hud,
        is_exiting=is_exiting,
        master_config=master_config,
        on_query_submitted=on_query_submitted,
        on_sleep_toggled=on_sleep_toggled,
        pending_utterances=pending_utterances,
        project_path=project_path,
        speech_active=speech_active,
        speech_cooldown_until=speech_cooldown_until,
        stop_current_speech=stop_current_speech,
        transcription_lock=transcription_lock,
        turns=turns,
        ui=ui,
        voice_generation=voice_generation,
        voice_init_error=voice_init_error,
        widget=widget,
    )

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
