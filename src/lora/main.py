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
from lora.ui.hud import LoraDesktopHUD
from lora.ui.tray import LoraSystemTray
from lora.ui.widget import WidgetState

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


def start_gateway_thread(host: str, port: int, queue=None, origin: str = "desktop", token_validator=None) -> uvicorn.Server | None:
    """Launches the FastAPI Gateway inside a dedicated background thread."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((host, port)) == 0:
            logger.info("Gateway daemon already active at http://%s:%d", host, port)
            return None

    app = create_gateway_app(origin=origin, queue=queue, token_validator=token_validator)
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
    """Launches the PySide6 holographic HUD, System Tray, and Audio/Voice Pipeline.

    Threading rule: only the GUI thread may touch ``hud``/``tray``. Worker threads (Ollama,
    Whisper, TTS, the sounddevice callback, the pynput hotkey) hand results over with
    ``ui.post(fn)``. ``QTimer.singleShot(0, fn)`` from those threads silently never fires.
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    import asyncio

    from lora.agent.runtime import AgentRuntime, TurnResult
    from lora.dialogue import ConversationHistory
    from lora.ui.threadsafe import UiDispatcher

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    ui = UiDispatcher()  # lives on the GUI thread

    # Initialize components
    from lora.grounding.media_vector_db import MediaVectorDB

    media_db = MediaVectorDB()
    audio = AudioCueManager(enabled=user_cfg.voice.sound_effects_enabled and not args.no_audio)
    hud = LoraDesktopHUD(media_db=media_db)
    tray = LoraSystemTray()

    def on_sleep_toggled(is_sleeping: bool) -> None:
        if is_sleeping:
            audio.play_cue(SoundCue.SLEEP)
            tray.update_state(WidgetState.SLEEPING)
            hud.set_voice_state("sleeping")
        else:
            audio.play_cue(SoundCue.WAKE)
            tray.update_state(WidgetState.IDLE)
            hud.set_voice_state("idle")

    hud.sleep_toggled.connect(on_sleep_toggled)
    tray.sleep_toggled.connect(lambda: hud.sleep_toggled.emit(hud._current_voice_state != "sleeping"))
    tray.open_hud_requested.connect(hud.toggle_window)

    # Clean shutdown & microphone release
    capture = None
    hotkey = None
    is_exiting = threading.Event()

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
        hud.close()
        tray.hide()
        app.quit()

    tray.exit_requested.connect(on_exit)
    hud.exit_requested.connect(on_exit)
    app.aboutToQuit.connect(on_exit)

    # Voice capture, STT, and TTS setup
    transcriber = None
    speech_active = threading.Event()
    active_voice = ["vivian"]
    voice_enabled = [True]
    voice_init_error = [""]

    def on_voice_changed(new_voice: str) -> None:
        active_voice[0] = new_voice
        logger.info("Active TTS voice selected: %s", new_voice)

    hud.voice_changed.connect(on_voice_changed)
    hud.voice_toggled.connect(lambda enabled: voice_enabled.__setitem__(0, bool(enabled)))

    tts_engine = None
    if not args.no_audio:
        try:
            from lora.voice.tts.engine import TtsEngine
            from lora.voice.tts.worker import Qwen3TtsWorkerProcess

            tts_worker = Qwen3TtsWorkerProcess(
                models_dir=r"C:\Models\Qwen3-TTS-12Hz-1.7B-CustomVoice",
                tokenizer_dir=r"C:\Models\Qwen3-TTS-Tokenizer-12Hz",
                default_voice=active_voice[0],
                spawn_process=False,
            )
            tts_engine = TtsEngine(worker=tts_worker)
            logger.info("TTS engine initialized.")
        except Exception as e:
            logger.warning("Could not initialize TTS engine: %s", e)

    # ---- speech: sentences are spoken while the reply is still being written ----
    current_streamer: list[object] = [None]

    def synth_sentence(sentence: str):
        if tts_engine is None:
            return None
        return tts_engine.worker.synthesize(sentence, voice=active_voice[0])

    def play_chunk(chunk, stop_event: threading.Event) -> None:
        import numpy as np
        import sounddevice as sd

        samples, sample_rate = chunk.samples, chunk.sample_rate
        if len(samples) == 0 or stop_event.is_set():
            return
        duration = len(samples) / sample_rate
        sd.play(samples, samplerate=sample_rate)
        num_steps = max(1, int(duration * 25))
        slice_size = max(1, len(samples) // num_steps)
        for step_idx in range(num_steps):
            if stop_event.is_set():
                sd.stop()
                return
            step = samples[step_idx * slice_size : (step_idx + 1) * slice_size]
            rms = float(np.sqrt(np.mean(step**2))) if len(step) else 0.0
            ui.post(lambda lvl=min(1.0, rms * 8.0): hud.set_audio_level(lvl))  # level only: no per-step pulses
            time.sleep(duration / num_steps)
        if not stop_event.is_set():
            sd.wait()

    def make_streamer(turn_done: threading.Event):
        """A SpeechStreamer for one turn, or None when audio/voice is off."""
        if args.no_audio or not voice_enabled[0]:
            return None
        from lora.voice.speech_stream import SpeechStreamer

        def _started() -> None:
            speech_active.set()
            ui.post(lambda: hud.set_voice_state("speaking"))

        def _done(spoke: bool) -> None:
            speech_active.clear()
            if not spoke and turn_done.is_set():
                logger.warning("Nothing was spoken (TTS produced no audio).")
            if turn_done.is_set():
                ui.post(lambda: (
                    hud.set_voice_state("idle"),
                    hud.set_audio_level(0.0),
                    audio.play_cue(SoundCue.COMPLETE) if spoke else None,
                ))

        return SpeechStreamer(synth_sentence, play_chunk, on_start=_started, on_done=_done)

    def stop_current_speech(update_ui: bool = True) -> None:
        streamer = current_streamer[0]
        if streamer is not None:
            streamer.stop()
        speech_active.clear()
        if tts_engine is not None:
            tts_engine.interrupt()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        if update_ui:
            ui.post(lambda: (hud.set_voice_state("idle"), hud.set_audio_level(0.0)))

    hud.stop_audio_requested.connect(stop_current_speech)

    last_query = [""]
    history = ConversationHistory(max_turns=6)
    inflight = threading.Event()

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
    )

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

    runtime.confirm = _confirm

    def on_query_submitted(query: str, depth: str = "") -> None:
        """Runs on the GUI thread. Starts one assistant turn in the background."""
        query = (query or "").strip()
        if not query:
            return
        if inflight.is_set():
            logger.info("Ignoring query while another is still running: %r", query)
            return
        inflight.set()
        last_query[0] = query
        if hud._current_voice_state in ("sleep", "sleeping"):
            on_sleep_toggled(False)

        stop_current_speech(update_ui=False)
        hud.set_voice_state("thinking")
        hud.stream_begin(query, "")
        turn_done = threading.Event()

        def _worker() -> None:
            started = time.monotonic()
            streamer = make_streamer(turn_done)
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
                result = asyncio.run(runtime.run_turn(query, on_status, on_text))
            except Exception as exc:  # never leave the UI waiting
                logger.exception("Turn crashed")
                result = TurnResult(text="Something went wrong on my side. Try again in a moment.", kind="error", failure=str(exc))
            logger.info(
                "Turn finished: kind=%s in %.1fs, sources=%d%s",
                result.kind, time.monotonic() - started, len(result.sources),
                f", failure={result.failure}" if result.failure else "",
            )

            def _post() -> None:
                try:
                    hud.present_answer(
                        query, result.text, sources=result.sources, confidence=0.0, label="", streamed=True,
                        meta=result.meta,
                    )
                    turn_done.set()
                    if streamer is None:
                        hud.set_voice_state("idle")
                        return
                    spoken = result.spoken if result.spoken is not None else result.text
                    if result.spoken is not None:
                        streamer.feed(spoken)
                    streamer.finish(spoken)
                    hud.set_voice_state("speaking" if streamer.has_pending else "idle")
                finally:
                    inflight.clear()

            ui.post(_post)

        threading.Thread(target=_worker, daemon=True).start()

    hud.query_submitted.connect(on_query_submitted)
    hud.rewrite_requested.connect(lambda dir, depth: on_query_submitted(last_query[0], depth))

    def _preload_voice() -> None:
        """Load the natural voice first: Ollama sizes its GPU share from what is free when IT loads."""
        if tts_engine is None or args.no_audio:
            return
        try:
            ok = tts_engine.worker.preload()
            diag = tts_engine.worker.diagnosis()
            (logger.info if ok else logger.warning)("Voice: %s %s", diag["summary"], diag.get("fix", "") if not ok else "")
        except Exception as exc:
            logger.warning("Voice preload failed: %s", exc)

    def _warm_up_model() -> None:
        """Load the model in the background at startup so the first greeting is instant."""
        _preload_voice()
        threading.Thread(target=runtime.start_mcp, daemon=True, name="lora-mcp").start()  # ~/.lora/mcp.json servers
        async def _go() -> None:
            from lora.models.ollama_client import OllamaClient
            from lora.models.selection import plan_chat

            client = OllamaClient(base_url=user_cfg.models.ollama_url)
            plan = None
            for _ in range(30):  # Ollama may still be starting: wait up to ~1 minute
                try:
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
            asyncio.run(_go())
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
            wakeword = WakeWordDetector(
                wake_phrase="Awaken LORA",
                chat_phrase="Hey LORA",
                transcriber=transcriber,
            )
            vad_threshold = float(os.environ.get("LORA_VAD_THRESHOLD", "0.35"))
            vad = VoiceActivityDetector(
                sample_rate=16000,
                threshold=vad_threshold,
                silence_duration_seconds=0.7,
                min_speech_duration_seconds=0.25,
            )

            is_transcribing = threading.Event()
            is_ptt_active = threading.Event()
            ptt_start_time = [0.0]
            armed_until = [0.0]  # after "Hey LORA" alone, the next utterance is the query
            ARM_SECONDS = 10.0
            stt_error_shown = [False]

            def _stt_failed(ex: Exception) -> None:
                logger.warning("Speech recognition error: %s", ex, exc_info=True)
                if not stt_error_shown[0]:
                    stt_error_shown[0] = True
                    msg = (
                        f"Speech recognition failed: {type(ex).__name__}: {ex}. "
                        "The first run downloads the 'distil-small.en' model, so it needs internet once."
                    )
                    ui.post(lambda: hud.append_message("L.O.R.A", msg))

            def _submit_voice_query(q: str) -> None:
                """GUI thread."""
                hud.append_message("You", q)
                hud.set_input_text(q)
                on_query_submitted(q, active_voice[0])

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
                        if len(audio_samples) < 3200 or peak < 0.005:
                            logger.info("Push-to-talk audio was silent (peak=%.4f).", peak)
                            ui.post(lambda: hud.append_message(
                                "L.O.R.A",
                                "I couldn't hear anything. Check that the right microphone is selected and not muted.",
                            ))
                            return
                        res = transcriber.transcribe(audio_samples)
                        query_text = res.text.strip()
                        if query_text:
                            logger.info("Transcribed voice input: '%s'", query_text)
                            submitted = True
                            ui.post(lambda q=query_text: _submit_voice_query(q))
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
                        text = transcriber.transcribe(audio_clip).text.strip()
                        if not text:
                            return
                        logger.info("Heard: '%s' (state=%s)", text, current_st)

                        if not sleeping and time.monotonic() < armed_until[0]:
                            armed_until[0] = 0.0
                            ui.post(lambda q=text: _submit_voice_query(q))
                            return

                        wake = wakeword.evaluate_text(text, audio_clip, current_state=current_st)
                        if not wake.detected:
                            return  # ordinary conversation in the room: ignore it
                        audio.play_cue(SoundCue.WAKE)
                        if sleeping:
                            logger.info("Wake word detected while sleeping. Waking up.")
                            ui.post(lambda: on_sleep_toggled(False))
                        elif wake.extracted_query:
                            logger.info("Wake word with trailing query: '%s'", wake.extracted_query)
                            ui.post(lambda q=wake.extracted_query: _submit_voice_query(q))
                        else:
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
                    ui.post(lambda lvl=level: hud.set_audio_level(lvl))

                if armed_until[0] and now > armed_until[0]:
                    armed_until[0] = 0.0
                    if hud._current_voice_state == "listening" and not is_ptt_active.is_set():
                        ui.post(lambda: hud.set_voice_state("idle"))

                if speech_active.is_set():
                    if rms > 0.15:
                        stop_current_speech()
                    return

                if is_transcribing.is_set() or hud._current_voice_state in ("thinking", "think", "verify"):
                    return

                if is_ptt_active.is_set():
                    return

                vad_state, _prob = vad.process_chunk(chunk)
                if vad_state == SpeechState.ENDED:
                    utterance = vad.get_and_reset_utterance()
                    if len(utterance) >= 4800:
                        _process_utterance_async(utterance)

            capture.register_chunk_callback(_on_mic_chunk)
            capture.start()
            if not capture.is_streaming:
                logger.warning("Microphone is not running: %s", capture.status_text())

            def _on_ptt_down() -> None:
                if hud._current_voice_state in ("thinking", "think", "verify", "sleep", "sleeping") or is_transcribing.is_set():
                    return
                armed_until[0] = 0.0
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
                elif dur >= 0.25:
                    audio_samples = capture.buffer.read_last_n_seconds(min(15.0, dur + 0.3))
                    _trigger_transcription(audio_samples)
                else:
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

    # Tell the user the truth about the microphone (name, or why it is off) and watch it.
    def _refresh_mic_status() -> None:
        if args.no_audio:
            hud.set_mic_status(False, "disabled (--no-audio)")
        elif capture is None:
            hud.set_mic_status(False, f"voice pipeline failed to start ({voice_init_error[0] or 'see log'})")
        elif not capture.is_streaming:
            hud.set_mic_status(False, capture.status_text())
        else:
            age = capture.seconds_since_audio()
            if age is None or age > 3.0:
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

    mic_timer = QTimer(hud)  # parented to the HUD so it stays alive; runs on the GUI thread
    mic_timer.setInterval(2000)
    mic_timer.timeout.connect(_refresh_mic_status)
    mic_timer.start()

    # Start Gateway background server
    from lora.agent.jobs import JobWorker
    from lora.gateway.queue import JobQueue

    job_queue = JobQueue()
    gateway_server = start_gateway_thread(host=args.host, port=args.port, queue=job_queue)

    def _make_job_runtime(origin: str) -> AgentRuntime:
        # Phone jobs may chat and research, never open apps/files or type; desktop-origin jobs keep full tools.
        return AgentRuntime(
            ollama_url=user_cfg.models.ollama_url,
            user_default_model=user_cfg.models.default_model,
            history=ConversationHistory(max_turns=4),
            registry_factory=_make_registry,
            profile_fn=_profile_text,
            media_db=media_db,
            allow_actions=(origin == "desktop"),
        )

    job_worker = JobWorker(job_queue, _make_job_runtime, desktop_busy=inflight.is_set)
    job_worker.start()

    mobile_server = None
    if getattr(args, "mobile_bind", None):
        try:
            import secrets as _secrets

            from lora.gateway.auth import TokenValidator

            m_host, _, m_port = args.mobile_bind.rpartition(":")
            token_file = Path(cfg_mgr.paths.user_profile_file).parent / "mobile_token.txt"
            token = os.environ.get("LORA_MOBILE_TOKEN") or (
                token_file.read_text(encoding="utf-8").strip() if token_file.exists() else _secrets.token_urlsafe(32)
            )
            token_file.write_text(token, encoding="utf-8")
            mobile_server = start_gateway_thread(
                m_host, int(m_port), queue=job_queue, origin="mobile", token_validator=TokenValidator(mobile_token=token)
            )
            logger.info("Mobile gateway on http://%s:%s. Pairing token saved in %s", m_host, m_port, token_file)
        except Exception as exc:
            logger.warning("Could not start the mobile gateway: %s", exc)

    # Show holographic HUD centered
    screen_geo = app.primaryScreen().geometry()
    hud.move(
        max(0, (screen_geo.width() - hud.width()) // 2),
        max(0, (screen_geo.height() - hud.height()) // 2),
    )
    hud.show()
    tray.show()
    audio.play_cue(SoundCue.WAKE)
    threading.Thread(target=_warm_up_model, daemon=True).start()
    logger.info("L.O.R.A Desktop Platform initialized successfully.")

    exit_code = app.exec()

    # Clean shutdown
    on_exit()
    job_worker.stop()
    if tts_engine is not None:
        try:
            tts_engine.worker.shutdown()
        except Exception:
            pass
    if runtime.mcp_hub is not None:
        runtime.mcp_hub.close()
    for server in (gateway_server, mobile_server):
        if server is not None:
            server.should_exit = True
    return exit_code


def main() -> int:
    parser = build_argument_parser()
    parser.add_argument(
        "--mobile-bind",
        default=None,
        metavar="HOST:PORT",
        help="Also listen for phone requests on this LAN address (token-protected). Off by default.",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_mgr, user_cfg = init_platform(base_dir=args.config_dir)

    if args.headless:
        logger.info("Starting L.O.R.A in headless daemon mode...")
        server = start_gateway_thread(host=args.host, port=args.port)
        try:
            import time

            while not server.should_exit:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Shutting down L.O.R.A headless daemon...")
            server.should_exit = True
        return 0

    return run_desktop_app(cfg_mgr, user_cfg, args)


if __name__ == "__main__":
    sys.exit(main())
