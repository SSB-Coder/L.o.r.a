"""Desktop output voice wiring with explicit shared-state/callback dependencies.

No Qt objects or native backends are constructed merely by importing this module.
The desktop retains ownership of the returned handles and their shutdown.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

logger = logging.getLogger("lora")


def create_speech_output(
    *,
    sound_cue,
    widget_state,
    active_voice,
    args,
    audio,
    current_streamer,
    hud,
    is_exiting,
    on_sleep_toggled,
    speech_active,
    speech_cooldown_until,
    tts_engine,
    tts_error_reported,
    ui,
    voice_enabled,
    widget,
):
    """Wire desktop voice output without changing turn or GUI-thread ownership."""
    WidgetState = widget_state  # noqa: N806 - class-valued dependency alias
    SoundCue = sound_cue  # noqa: N806 - class-valued dependency alias

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
                ui.post(
                    lambda: hud.append_message(
                        "System",
                        "My voice stopped, but text chat is still available. I'll restart the voice worker for the next reply.",
                    )
                )
            return None
        if is_exiting.is_set():
            return None
        if chunk.error and not tts_error_reported.is_set():
            tts_error_reported.set()
            ui.post(
                lambda: hud.append_message(
                    "L.O.R.A",
                    "My Kokoro voice isn't available. Run python scripts/download_models.py --kokoro-only; you can still chat with me here.",
                )
            )
        return chunk

    def play_chunk(chunk, stop_event: threading.Event) -> None:
        from lora.voice.playback import play_audio

        speech_active.set()
        try:
            play_audio(
                chunk.samples,
                chunk.sample_rate,
                stop_event,
                lambda rms: ui.post(
                    lambda lvl=min(1.0, rms * 8.0): (
                        hud.set_audio_level(lvl),
                        widget.set_audio_level(lvl),
                    )
                ),
                is_exiting,
            )
        finally:
            speech_cooldown_until[0] = time.monotonic() + 0.45

    def make_streamer(
        turn_done: threading.Event, is_sleep_turn: list[bool] | None = None, is_current=lambda: True
    ):
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
                post_voice(
                    lambda: (
                        (hud.set_voice_state("speaking"), widget.set_state(WidgetState.SPEAKING))
                        if hud._current_voice_state not in ("sleep", "sleeping")
                        else None
                    )
                )

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
                    post_voice(
                        lambda: (
                            on_sleep_toggled(True, play_sound=False),
                            hud.set_audio_level(0.0),
                            widget.set_audio_level(0.0),
                        )
                    )
                else:
                    post_voice(
                        lambda: (
                            hud.set_voice_state("idle"),
                            widget.set_state(WidgetState.IDLE),
                            hud.set_audio_level(0.0),
                            widget.set_audio_level(0.0),
                            audio.play_cue(SoundCue.COMPLETE) if spoke else None,
                        )
                    )

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
            ui.post(
                lambda: (
                    hud.set_voice_state("idle"),
                    widget.set_state(WidgetState.IDLE),
                    hud.set_audio_level(0.0),
                    widget.set_audio_level(0.0),
                )
            )

    return synth_sentence, play_chunk, make_streamer, stop_current_speech
