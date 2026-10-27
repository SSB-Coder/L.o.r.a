"""Desktop input voice wiring with explicit shared-state/callback dependencies.

No Qt objects or native backends are constructed merely by importing this module.
The desktop retains ownership of the returned handles and their shutdown.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger("lora")


def start_voice_input(
    *,
    timer_class,
    sound_cue,
    _on_emergency_stop,
    args,
    armed_until,
    audio,
    current_streamer,
    deferred_utterances,
    headset_mode,
    hud,
    is_exiting,
    master_config,
    on_query_submitted,
    on_sleep_toggled,
    pending_utterances,
    project_path,
    speech_active,
    speech_cooldown_until,
    stop_current_speech,
    transcription_lock,
    turns,
    ui,
    voice_generation,
    voice_init_error,
    widget,
):
    """Wire desktop voice input without changing turn or GUI-thread ownership."""
    QTimer = timer_class  # noqa: N806 - class-valued dependency alias
    SoundCue = sound_cue  # noqa: N806 - class-valued dependency alias
    capture = transcriber = vad = hotkey = None
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
                model_size_or_path=stt_config.model,
                device="cpu",
                compute_type=stt_config.compute,
                initial_prompt=stt_config.initial_prompt or None,
                lazy_load=True,
            )
            speaker_config = master_config.voice.wake.speaker_verifier
            enabled_override = os.environ.get("LORA_SPEAKER_VERIFY_ENABLED")
            speaker_verifier_enabled = (
                speaker_config.enabled
                if enabled_override is None
                else enabled_override.lower() in ("1", "true", "yes")
            )
            from lora.voice.wakeword import SpeakerVerifier

            verifier = SpeakerVerifier(
                enabled=speaker_verifier_enabled,
                profile_path=speaker_config.model,
                threshold=speaker_config.threshold,
            )
            if verifier.enabled and not verifier.reference_embeddings:
                hud.append_message(
                    "System",
                    "No valid owner voice profile was found. Run lora enroll for owner filtering; wake-phrase listening is still available.",
                )
            wakeword = WakeWordDetector(
                wake_phrase="Awaken LORA",
                chat_phrase="Hey LORA",
                model_path=str(project_path(master_config.voice.wake.model))
                if project_path(master_config.voice.wake.model).is_file()
                else None,
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

            def _submit_voice_query(
                q: str, low_confidence: bool = False, generation: int | None = None
            ) -> None:
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
                        f'Low audio confidence - did you mean:\n\n"{q}"?',
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
                            ui.post(
                                lambda: hud.append_message(
                                    "L.O.R.A",
                                    "I couldn't hear anything. Check that the right microphone is selected and not muted.",
                                )
                            )
                            return
                        res = transcriber.transcribe(audio_samples)
                        if generation != voice_generation[0]:
                            return
                        query_text = res.text.strip()
                        if query_text:
                            logger.info(
                                "Transcribed voice input: '%s' (conf=%.2f)",
                                query_text,
                                res.confidence,
                            )
                            submitted = True
                            is_low = _is_low_stt_confidence(res)
                            ui.post(
                                lambda q=query_text, lc=is_low: _submit_voice_query(
                                    q, low_confidence=lc, generation=generation
                                )
                            )
                        else:
                            logger.info("No speech detected in audio segment.")
                            ui.post(
                                lambda: hud.append_message(
                                    "L.O.R.A", "I didn't catch that. Try again?"
                                )
                            )
                    except Exception as ex:
                        _stt_failed(ex)
                    finally:
                        is_transcribing.clear()
                        if not submitted and not speech_active.is_set():
                            ui.post(
                                lambda: (
                                    hud.set_voice_state("idle")
                                    if generation == voice_generation[0]
                                    else None
                                )
                            )

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
                                if not wakeword.process_audio(
                                    audio_clip, current_state=current_st
                                ).detected:
                                    return
                        res = transcriber.transcribe(audio_clip)
                        if generation != voice_generation[0]:
                            return
                        text = res.text.strip()
                        if not text:
                            return
                        logger.debug(
                            "Voice candidate (state=%s, avg_logprob=%.2f)",
                            current_st,
                            res.avg_logprob,
                        )

                        if not sleeping and time.monotonic() < armed_until[0]:
                            armed_until[0] = 0.0
                            submitted = True
                            is_low = _is_low_stt_confidence(res)
                            ui.post(
                                lambda q=text, lc=is_low: _submit_voice_query(
                                    q, low_confidence=lc, generation=generation
                                )
                            )
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

                            def resume_voice(
                                q=wake.extracted_query, lc=_is_low_stt_confidence(res)
                            ):
                                on_sleep_toggled(False)
                                if q:
                                    _submit_voice_query(
                                        q, low_confidence=lc, generation=voice_generation[0]
                                    )
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
                            ui.post(
                                lambda q=wake.extracted_query, lc=is_low: _submit_voice_query(
                                    q, low_confidence=lc, generation=generation
                                )
                            )
                        else:
                            audio.play_cue(SoundCue.WAKE)
                            logger.info("Wake word detected. Listening for the query...")
                            with transcription_lock:
                                if generation != voice_generation[0]:
                                    return
                                armed_until[0] = time.monotonic() + arm_seconds
                            ui.post(
                                lambda: (
                                    hud.set_voice_state("listening")
                                    if generation == voice_generation[0]
                                    else None
                                )
                            )
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
                    ui.post(
                        lambda lvl=level: (hud.set_audio_level(lvl), widget.set_audio_level(lvl))
                    )

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

                if (
                    hud._current_voice_state in ("sleep", "sleeping")
                    and master_config.voice.close_mic_on_sleep
                ):
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
                        ui.post(
                            lambda: hud.append_message(
                                "System",
                                "Voice detection stopped. I'll restart it on the next audio block; you can still type here.",
                            )
                        )
                    return
                if vad_state == SpeechState.ENDED:
                    utterance = vad.get_and_reset_utterance()
                    if len(utterance) >= 2400:  # >= 150ms (includes 500ms pre-roll + "Hey LORA")
                        _process_utterance_async(utterance)

            capture.register_chunk_callback(_on_mic_chunk)
            capture.start()

            def _drain_deferred_voice() -> None:
                streamer = current_streamer[0]
                if (
                    turns.busy
                    or speech_active.is_set()
                    or (streamer is not None and streamer.has_pending)
                ):
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
                logger.info(
                    "Push-to-talk released after %.1fs (mic streaming: %s)",
                    dur,
                    capture.is_streaming,
                )
                if not capture.is_streaming:
                    reason = capture.status_text()
                    ui.post(
                        lambda: (
                            hud.append_message("L.O.R.A", f"My microphone isn't running: {reason}"),
                            hud.set_voice_state("idle"),
                        )
                    )
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
                    on_killswitch=lambda: ui.post(
                        lambda: _on_emergency_stop("Hotkey pressed (Ctrl+Esc)")
                    ),
                )
                hotkey.on_sleep_toggle = lambda: ui.post(
                    lambda: on_sleep_toggled(hud._current_voice_state not in ("sleep", "sleeping"))
                )
                hotkey.start()
            except Exception as e:
                logger.warning("Global push-to-talk hotkey unavailable: %s", e)
        except Exception as e:
            voice_init_error[0] = f"{type(e).__name__}: {e}"
            logger.warning("Voice capture could not be initialized: %s", e, exc_info=True)

    return capture, transcriber, vad, hotkey
