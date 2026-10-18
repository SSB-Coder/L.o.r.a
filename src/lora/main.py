import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

import uvicorn

from lora.config.manager import ConfigManager, LoraUserConfig
from lora.gateway.server import create_gateway_app
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="L.O.R.A - Layer Oriented Research Assistant (Native Desktop Platform)"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Path to custom LORA config directory (default: ~/.lora)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Gateway HTTP/WebSocket port (default: 8765)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Gateway bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without desktop UI (Gateway daemon only)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable microphone capture and TTS voice pipeline",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force air-gapped offline research mode",
    )
    return parser


def init_platform(base_dir: Path | None = None) -> tuple[ConfigManager, LoraUserConfig]:
    """Initializes the ~/.lora directory hierarchy, locks NTFS ACLs, and loads configuration."""
    cfg_mgr = ConfigManager(base_dir=base_dir)
    cfg_mgr.ensure_ecosystem(apply_acls=True)
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
    hotkey = None
    is_exiting = threading.Event()
    speech_cooldown_until = [time.monotonic() + 1.0]  # ignore startup chime during initial 1.0s

    def on_sleep_toggled(is_sleeping: bool, play_sound: bool = True) -> None:
        speech_cooldown_until[0] = time.monotonic() + 0.6
        if is_sleeping:
            if play_sound and hud._current_voice_state not in ("sleep", "sleeping"):
                audio.play_cue(SoundCue.SLEEP)
            tray.update_state(WidgetState.SLEEPING)
            widget.set_state(WidgetState.SLEEPING)
            hud.set_voice_state("sleeping")
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
        logger.info("Exiting L.O.R.A, releasing audio capture stream...")
        audio.play_cue(SoundCue.SLEEP)
        if capture is not None:
            try:
                capture.stop()
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
    headset_mode = os.environ.get("LORA_HEADSET_MODE", "0").strip().lower() in ("1", "true", "yes")

    def on_voice_changed(new_voice: str) -> None:
        active_voice[0] = new_voice or "af_sky"
        logger.info("Active TTS voice selected: %s", active_voice[0])

    hud.voice_changed.connect(on_voice_changed)
    hud.voice_toggled.connect(lambda enabled: voice_enabled.__setitem__(0, bool(enabled)))

    tts_engine = None
    if not args.no_audio:
        try:
            from lora.voice.tts.engine import TtsEngine
            from lora.voice.tts.worker import Qwen3TtsWorkerProcess

            kokoro_dir = (
                str(Path("models/kokoro").resolve())
                if Path("models/kokoro").exists()
                else r"C:\Models\kokoro"
            )
            tts_worker = Qwen3TtsWorkerProcess(
                models_dir=kokoro_dir,
                tokenizer_dir=r"C:\Models\Qwen3-TTS-Tokenizer-12Hz",
                default_voice=active_voice[0],
                spawn_process=False,
            )
            tts_engine = TtsEngine(worker=tts_worker)
            logger.info("TTS engine initialized (default voice: %s).", active_voice[0])
        except Exception as e:
            logger.warning("Could not initialize TTS engine: %s", e)

    # ---- speech: sentences are spoken while the reply is still being written ----
    current_streamer: list[object] = [None]

    def synth_sentence(sentence: str):
        if tts_engine is None:
            return None
        return tts_engine.worker.synthesize(sentence, voice=active_voice[0])

    def play_chunk(chunk, stop_event: threading.Event) -> None:
        import io
        import wave
        import numpy as np

        samples, sample_rate = chunk.samples, chunk.sample_rate
        if len(samples) == 0 or stop_event.is_set() or is_exiting.is_set():
            return
        speech_active.set()
        duration = len(samples) / sample_rate
        used_winsound = False
        # On Windows, prefer winsound.PlaySound (OS waveOut mixer) so PortAudio's 16kHz microphone InputStream
        # is never disrupted or aborted by 24kHz sd.play() / sd.stop() stream switching!
        if sys.platform == "win32":
            try:
                import winsound

                pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(pcm16)
                winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY | winsound.SND_ASYNC)
                used_winsound = True
            except Exception:
                used_winsound = False

        if not used_winsound:
            try:
                import sounddevice as sd

                sd.play(samples, samplerate=sample_rate)
            except Exception:
                return

        num_steps = max(1, int(duration * 25))
        slice_size = max(1, len(samples) // num_steps)
        for step_idx in range(num_steps):
            if stop_event.is_set() or is_exiting.is_set():
                if used_winsound and sys.platform == "win32":
                    try:
                        import winsound

                        winsound.PlaySound(None, winsound.SND_PURGE)
                    except Exception:
                        pass
                return
            step = samples[step_idx * slice_size : (step_idx + 1) * slice_size]
            rms = float(np.sqrt(np.mean(step**2))) if len(step) else 0.0
            ui.post(lambda lvl=min(1.0, rms * 8.0): (hud.set_audio_level(lvl), widget.set_audio_level(lvl)))
            time.sleep(duration / num_steps)
        speech_cooldown_until[0] = time.monotonic() + 0.45

    def make_streamer(turn_done: threading.Event, is_sleep_turn: list[bool] | None = None):
        """A SpeechStreamer for one turn, or None when audio/voice is off."""
        if args.no_audio or not voice_enabled[0]:
            return None
        from lora.voice.speech_stream import SpeechStreamer

        def _started() -> None:
            speech_active.set()
            if not (is_sleep_turn and is_sleep_turn[0]):
                ui.post(lambda: (hud.set_voice_state("speaking"), widget.set_state(WidgetState.SPEAKING)))

        def _done(spoke: bool) -> None:
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
                    ui.post(lambda: (
                        on_sleep_toggled(True, play_sound=False),
                        hud.set_audio_level(0.0),
                        widget.set_audio_level(0.0),
                    ))
                else:
                    ui.post(lambda: (
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

    last_query = [""]
    history = ConversationHistory(max_turns=6)
    inflight = threading.Event()
    inflight_since = [0.0]

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
        notebook_db_path=Path(cfg_mgr.paths.user_profile_file).parent / "notebooks.db",
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

    def _handle_power_cmd(action: str) -> None:
        if action == "sleep":
            ui.post(lambda: on_sleep_toggled(True))
        elif action == "turn_off":
            ui.post(on_exit)

    runtime.on_power_command = _handle_power_cmd

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
    threading.Thread(target=bg_loop.run_forever, daemon=True, name="lora-async-loop").start()

    def _run_coro(coro, timeout: float = 180.0):
        fut = asyncio.run_coroutine_threadsafe(coro, bg_loop)
        return fut.result(timeout=timeout)

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
        if inflight.is_set():
            # Auto-recover if a previous turn has been stuck > 90s
            if inflight_since[0] > 0 and (time.monotonic() - inflight_since[0]) > 90.0:
                logger.warning("Previous turn exceeded 90s lock window; clearing inflight flag.")
                inflight.clear()
            else:
                from lora.agent.commands import parse_command

                fast_cmd = parse_command(query) if not turn_attachments else None
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
                logger.info("Ignoring query while another is still running: %r", query)
                return
        inflight.set()
        inflight_since[0] = time.monotonic()
        last_query[0] = query
        turn_is_sleep = [False]
        from lora.agent.commands import parse_command

        incoming_cmd = parse_command(query) if not turn_attachments else None
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
            streamer = make_streamer(turn_done, is_sleep_turn=turn_is_sleep)
            current_streamer[0] = streamer
            last_push = [0.0]

            def on_status(text: str) -> None:
                ui.post(lambda t=text: hud.set_status(t))

            def on_text(full: str, speak: bool = True) -> None:
                now = time.monotonic()
                if now - last_push[0] >= 0.06:
                    last_push[0] = now
                    ui.post(lambda t=full: hud.stream_update(t))
                if speak and streamer is not None:
                    streamer.feed(full)

            try:
                turn_timeout = float(os.environ.get("LORA_TURN_TIMEOUT", "600.0"))
                result = _run_coro(
                    runtime.run_turn(
                        query,
                        on_status,
                        on_text,
                        mode=mode_override,
                        attachments=turn_attachments or None,
                    ),
                    timeout=turn_timeout,
                )
            except Exception as exc:
                logger.exception("Turn crashed")
                result = TurnResult(text="Something went wrong on my side. Try again in a moment.", kind="error", failure=str(exc))
            finally:
                # Always release inflight even if the UI callback is delayed
                inflight.clear()
                inflight_since[0] = 0.0

            logger.info(
                "Turn finished: kind=%s in %.1fs, sources=%d%s",
                result.kind, time.monotonic() - started, len(result.sources),
                f", failure={result.failure}" if result.failure else "",
            )

            if (getattr(result, "meta", None) and result.meta.get("power_action") == "sleep") or turn_is_sleep[0]:
                turn_is_sleep[0] = True

            def _post() -> None:
                try:
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
                finally:
                    inflight.clear()
                    inflight_since[0] = 0.0

            ui.post(_post)

        threading.Thread(target=_worker, daemon=True).start()

    hud.query_submitted.connect(on_query_submitted)
    hud.rewrite_requested.connect(lambda dir, depth: on_query_submitted(f"Explain that at {depth or 'detailed'} depth", depth))

    def _preload_voice() -> None:
        """Load the Kokoro CPU voice and Faster-Whisper STT model at startup so turn 1 is instant."""
        if args.no_audio:
            return
        if transcriber is not None:
            try:
                transcriber.preload()
            except Exception as exc:
                logger.warning("STT preload failed: %s", exc)
        if tts_engine is None:
            return
        try:
            ok = tts_engine.worker.preload()
            diag = tts_engine.worker.diagnosis()
            (logger.info if ok else logger.warning)("Voice: %s %s", diag["summary"], diag.get("fix", "") if not ok else "")
        except Exception as exc:
            logger.warning("Voice preload failed: %s", exc)

    def _warm_up_model() -> None:
        """Load the model in the background at startup and sync installed models to the UI dropdown."""
        _preload_voice()
        if os.environ.get("LORA_ENABLE_MCP", "0") == "1":
            threading.Thread(target=runtime.start_mcp, daemon=True, name="lora-mcp").start()
        async def _go() -> None:
            from lora.models.ollama_client import OllamaClient
            from lora.models.selection import is_excluded_model, plan_chat

            client = OllamaClient(base_url=user_cfg.models.ollama_url)
            plan = None
            for _ in range(30):
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

    if not args.no_audio:
        try:
            import numpy as np

            from lora.voice.capture import AudioCapture
            from lora.voice.stt import FasterWhisperTranscriber
            from lora.voice.vad import SpeechState, VoiceActivityDetector
            from lora.voice.wakeword import WakeWordDetector

            capture = AudioCapture(sample_rate=16000)
            transcriber = FasterWhisperTranscriber(lazy_load=True)
            speaker_verifier_enabled = os.environ.get("LORA_SPEAKER_VERIFY_ENABLED", "0").lower() in ("1", "true", "yes")
            from lora.voice.wakeword import SpeakerVerifier
            verifier = SpeakerVerifier(enabled=speaker_verifier_enabled)
            wakeword = WakeWordDetector(
                wake_phrase="Awaken LORA",
                chat_phrase="Hey LORA",
                model_path="models/voice/hey_lora.onnx" if Path("models/voice/hey_lora.onnx").is_file() else None,
                speaker_verifier=verifier,
                transcriber=transcriber,
            )
            vad_threshold = float(os.environ.get("LORA_VAD_THRESHOLD", "0.50"))
            vad = VoiceActivityDetector(
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
            armed_until = [0.0]
            ARM_SECONDS = 12.0
            stt_error_shown = [False]

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
                    return float(avg_lp) < -0.85
                return False

            def _submit_voice_query(q: str, low_confidence: bool = False) -> None:
                """GUI thread."""
                hud.append_message("You", q)
                hud.set_input_text(q)
                if low_confidence:
                    from lora.agent.commands import parse_command

                    maybe_cmd = parse_command(q)
                    if maybe_cmd is not None or not runtime.laya_router.check_stt_plausible(q):
                        from PySide6.QtWidgets import QMessageBox

                        ans = QMessageBox.question(
                            hud,
                            "Confirm Voice Command",
                            f"Low audio confidence — did you mean:\n\n\"{q}\"?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        )
                        if ans != QMessageBox.StandardButton.Yes:
                            hud.append_message("L.O.R.A", "Cancelled low-confidence voice command.")
                            hud.set_voice_state("idle")
                            return
                on_query_submitted(q, getattr(hud, "_depth_mode", ""))

            def _trigger_transcription(audio_samples: np.ndarray) -> None:
                if is_transcribing.is_set():
                    logger.info("Push-to-talk ignored: still transcribing the previous clip.")
                    return
                is_transcribing.set()
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
                        query_text = res.text.strip()
                        if query_text:
                            logger.info("Transcribed voice input: '%s' (conf=%.2f)", query_text, res.confidence)
                            submitted = True
                            is_low = _is_low_stt_confidence(res)
                            ui.post(lambda q=query_text, lc=is_low: _submit_voice_query(q, low_confidence=lc))
                        else:
                            logger.info("No speech detected in audio segment.")
                            ui.post(lambda: hud.append_message("L.O.R.A", "I didn't catch that. Try again?"))
                    except Exception as ex:
                        _stt_failed(ex)
                    finally:
                        is_transcribing.clear()
                        if not submitted and not speech_active.is_set():
                            ui.post(lambda: hud.set_voice_state("idle"))

                threading.Thread(target=_worker, daemon=True).start()

            def _process_utterance_async(audio_clip: np.ndarray) -> None:
                """Hands-free path: transcribe ONCE, then decide wake word vs query vs ignore."""
                if is_transcribing.is_set() or is_ptt_active.is_set():
                    return
                is_transcribing.set()

                def _bg() -> None:
                    try:
                        current_st = hud._current_voice_state
                        sleeping = current_st in ("sleep", "sleeping")
                        res = transcriber.transcribe(audio_clip)
                        text = res.text.strip()
                        if not text:
                            return
                        logger.info("Heard: '%s' (state=%s, avg_logprob=%.2f)", text, current_st, res.avg_logprob)

                        if not sleeping and time.monotonic() < armed_until[0]:
                            armed_until[0] = 0.0
                            is_low = _is_low_stt_confidence(res)
                            ui.post(lambda q=text, lc=is_low: _submit_voice_query(q, low_confidence=lc))
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
                        speech_cooldown_until[0] = time.monotonic() + 0.35
                        is_low = _is_low_stt_confidence(res)
                        if sleeping:
                            logger.info("Wake word detected while sleeping. Waking up.")
                            if wake.extracted_query:
                                logger.info("Wake from sleep with trailing query: '%s'", wake.extracted_query)
                                ui.post(
                                    lambda q=wake.extracted_query, lc=is_low: (
                                        on_sleep_toggled(False),
                                        _submit_voice_query(q, low_confidence=lc),
                                    )
                                )
                            else:
                                armed_until[0] = time.monotonic() + ARM_SECONDS
                                ui.post(lambda: (on_sleep_toggled(False), hud.set_voice_state("listening")))
                        elif wake.extracted_query:
                            from lora.agent.commands import parse_command

                            maybe_cmd = parse_command(wake.extracted_query)
                            if maybe_cmd is None or maybe_cmd.kind != "sleep":
                                audio.play_cue(SoundCue.WAKE)
                            logger.info("Wake word with trailing query: '%s'", wake.extracted_query)
                            ui.post(lambda q=wake.extracted_query, lc=is_low: _submit_voice_query(q, low_confidence=lc))
                        else:
                            audio.play_cue(SoundCue.WAKE)
                            logger.info("Wake word detected. Listening for the query...")
                            armed_until[0] = time.monotonic() + ARM_SECONDS
                            ui.post(lambda: hud.set_voice_state("listening"))
                    except Exception as ex:
                        _stt_failed(ex)
                    finally:
                        is_transcribing.clear()

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

                if is_transcribing.is_set() or hud._current_voice_state in ("thinking", "think", "verify"):
                    return

                vad_state, _prob = vad.process_chunk(chunk)
                if vad_state == SpeechState.ENDED:
                    utterance = vad.get_and_reset_utterance()
                    if len(utterance) >= 2400:  # >= 150ms (includes 500ms pre-roll + "Hey LORA")
                        _process_utterance_async(utterance)

            capture.register_chunk_callback(_on_mic_chunk)
            capture.start()
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

                hotkey = PushToTalkListener(on_activate=_on_ptt_down, on_deactivate=_on_ptt_up)
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
        elif hud._current_voice_state in ("sleep", "sleeping"):
            hud.set_mic_status(True, f"{capture.device_name or 'microphone'} (sleep mode)")
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
    )

    def _make_job_runtime(origin: str) -> AgentRuntime:
        return AgentRuntime(
            ollama_url=user_cfg.models.ollama_url,
            user_default_model=user_cfg.models.default_model,
            history=ConversationHistory(max_turns=4),
            registry_factory=_make_registry,
            profile_fn=_profile_text,
            media_db=media_db,
            allow_actions=(origin == "desktop"),
            notebook_db_path=Path(cfg_mgr.paths.user_profile_file).parent / "notebooks.db",
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
        desktop_busy=inflight.is_set,
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
        target = bind_str or getattr(args, "mobile_bind", None) or "0.0.0.0:8766"
        m_host, _, m_port = target.rpartition(":")
        m_port_int = int(m_port or 8766)
        display_ip = lan_ip if m_host in ("0.0.0.0", "") else m_host
        token_file = Path(cfg_mgr.paths.user_profile_file).parent / "mobile_token.txt"
        token = os.environ.get("LORA_MOBILE_TOKEN") or (
            token_file.read_text(encoding="utf-8").strip() if token_file.exists() else _secrets.token_urlsafe(32)
        )
        token_file.write_text(token, encoding="utf-8")
        srv = start_gateway_thread(
            m_host or "0.0.0.0",
            m_port_int,
            queue=job_queue,
            origin="mobile",
            token_validator=TokenValidator(mobile_token=token),
            media_db=media_db,
            connection_manager=mobile_ws_manager,
        )
        if srv is not None:
            mobile_server_ref[0] = srv
        url = get_pairing_url(display_ip, m_port_int, token)
        mobile_url_ref[0] = url
        try:
            qr_ascii = generate_pairing_qr_terminal(display_ip, m_port_int, token)
            logger.info("Mobile gateway on %s\n%s", url, qr_ascii)
        except Exception:
            logger.info("Mobile gateway on %s (token saved in %s)", url, token_file)
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

    exit_code = app.exec()

    # Clean shutdown
    on_exit()
    job_worker.stop()
    if runtime._registry is not None:
        try:
            _run_coro(runtime._registry.browser_driver.close(), timeout=10.0)
        except Exception:
            pass
    try:
        bg_loop.call_soon_threadsafe(bg_loop.stop)
    except Exception:
        pass
    if tts_engine is not None:
        try:
            tts_engine.worker.shutdown()
        except Exception:
            pass
    if runtime.mcp_hub is not None:
        runtime.mcp_hub.close()
    for server in (gateway_server, mobile_server_ref[0]):
        if server is not None:
            server.should_exit = True
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    parser.add_argument(
        "--mobile-bind",
        default=None,
        metavar="HOST:PORT",
        help="Also listen for phone requests on this LAN address (token-protected). Off by default.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_mgr, user_cfg = init_platform(base_dir=args.config_dir)

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

        logger.info("Starting L.O.R.A in headless daemon mode...")
        h_queue = JobQueue()
        h_ws = ConnectionManager()
        h_media = MediaVectorDB()
        h_worker = JobWorker(
            h_queue,
            lambda origin: AgentRuntime(
                ollama_url=user_cfg.models.ollama_url,
                user_default_model=user_cfg.models.default_model,
                history=ConversationHistory(max_turns=4),
                media_db=h_media,
                allow_actions=(origin == "desktop"),
                notebook_db_path=Path(cfg_mgr.paths.user_profile_file).parent / "notebooks.db",
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
        )
        try:
            while server and not server.should_exit:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Shutting down L.O.R.A headless daemon...")
            h_worker.stop()
            if server is not None:
                server.should_exit = True
        return 0

    return run_desktop_app(cfg_mgr, user_cfg, args)


cli = main


if __name__ == "__main__":
    sys.exit(main())
