"""Speak a reply while it is still being written.

``feed(text_so_far)`` is called as tokens stream in. Completed sentences are cleaned, synthesised
and played one after another in the background, so speech starts after the first sentence rather
than after the whole answer. Synthesis of sentence N+1 overlaps playback of sentence N.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from lora.dialogue import speakable

logger = logging.getLogger(__name__)

_BOUNDARY = re.compile(r"(?<=[.!?…])[\"')\]]*\s+|\n{1,}")
_SENTINEL = object()


def split_complete(text: str) -> tuple[list[str], str]:
    """Split ``text`` into finished sentences and the unfinished tail."""
    sentences: list[str] = []
    last = 0
    for m in _BOUNDARY.finditer(text):
        piece = text[last : m.end()].strip()
        if piece:
            sentences.append(piece)
        last = m.end()
    return sentences, text[last:]


class SpeechStreamer:
    def __init__(
        self,
        synthesize: Callable[[str], Any],
        play: Callable[[Any, threading.Event], None],
        on_start: Callable[[], None] | None = None,
        on_done: Callable[[bool], None] | None = None,
        max_chars: int = 900,
    ) -> None:
        """
        synthesize(sentence) -> audio chunk or None.
        play(chunk, stop_event) -> blocks while playing, returns early if stop_event is set.
        on_done(spoke_anything) is called once, after the last sentence finished or after stop().
        """
        self._synthesize = synthesize
        self._play = play
        self._on_start = on_start
        self._on_done = on_done
        self.max_chars = max_chars
        self._text_q: queue.Queue = queue.Queue()
        self._audio_q: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._consumed = 0  # raw characters already split off
        self._spoken_chars = 0
        self._fence_open = False
        self._finished = False
        self._spoke = False
        self._done_called = False
        self._lock = threading.Lock()
        self._threads = [
            threading.Thread(target=self._synth_loop, daemon=True),
            threading.Thread(target=self._play_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()

    # ----- producer side -----
    def feed(self, text_so_far: str) -> None:
        if self._stop.is_set() or self._finished:
            return
        sentences, _tail = split_complete(text_so_far[self._consumed :])
        for raw in sentences:
            self._consumed = text_so_far.find(raw, self._consumed) + len(raw)
            self._enqueue(raw)

    def finish(self, text_so_far: str | None = None) -> None:
        """No more text: flush the unfinished tail, then end the queue."""
        if self._finished:
            return
        if text_so_far is not None and not self._stop.is_set():
            tail = text_so_far[self._consumed :].strip()
            if tail:
                self._enqueue(tail)
        self._finished = True
        self._text_q.put(_SENTINEL)

    def stop(self) -> None:
        """Barge-in: drop everything queued and cut the current sentence."""
        self._stop.set()
        for q in (self._text_q, self._audio_q):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
        self._text_q.put(_SENTINEL)
        try:
            self._audio_q.put_nowait(_SENTINEL)
        except queue.Full:
            pass  # Consumer checks _stop; never block a GUI interrupt.

    def wait(self, timeout: float = 2.0) -> bool:
        """Bounded shutdown join; call after interrupting any pending native work."""
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(max(0.0, deadline - time.monotonic()))
        return all(not thread.is_alive() for thread in self._threads)

    @property
    def has_pending(self) -> bool:
        """True if anything was queued to be spoken (so audio is on its way)."""
        return self._spoken_chars > 0 and not self._stop.is_set() and not self._done_called

    @property
    def active(self) -> bool:
        return not self._stop.is_set() and not self._done_called

    # ----- internals -----
    def _enqueue(self, raw: str) -> None:
        if raw.count("```") % 2 == 1:
            self._fence_open = not self._fence_open
            return
        if self._fence_open:
            return  # never read code blocks aloud
        clean = speakable(raw)
        if len(clean) < 2 or not re.search(r"\w", clean):
            return
        if self._spoken_chars >= self.max_chars:
            return  # long detailed answers: read the opening, the rest is on screen
        clean = clean[:self.max_chars - self._spoken_chars].rstrip()
        self._spoken_chars += len(clean)
        self._text_q.put(clean)

    def _synth_loop(self) -> None:
        while not self._stop.is_set():
            item = self._text_q.get()
            if item is _SENTINEL or self._stop.is_set():
                break
            try:
                chunk = self._synthesize(item)
            except Exception as exc:
                logger.warning("TTS synthesis failed: %s", exc)
                chunk = None
            samples = getattr(chunk, "samples", None)
            if chunk is not None and (samples is None or len(samples)) and not self._stop.is_set():
                self._put_audio(chunk)
        self._put_audio(_SENTINEL)

    def _put_audio(self, value: Any) -> None:
        while not self._stop.is_set():
            try:
                self._audio_q.put(value, timeout=0.05)
                return
            except queue.Full:
                continue

    def _play_loop(self) -> None:
        while True:
            chunk = self._audio_q.get()
            if chunk is _SENTINEL or self._stop.is_set():
                break
            if not self._spoke:
                self._spoke = True
                if self._on_start:
                    try:
                        self._on_start()
                    except Exception:
                        logger.exception("TTS start callback failed")
            try:
                self._play(chunk, self._stop)
            except Exception as exc:
                logger.warning("TTS playback failed: %s", exc)
        self._finish_once()

    def _finish_once(self) -> None:
        with self._lock:
            if self._done_called:
                return
            self._done_called = True
        if self._on_done:
            try:
                self._on_done(self._spoke)
            except Exception:
                logger.exception("on_done failed")
