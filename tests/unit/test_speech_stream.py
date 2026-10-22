import threading
import time

from lora.voice.speech_stream import SpeechStreamer, split_complete


def test_split_complete_keeps_unfinished_tail():
    done, tail = split_complete("Hello there. How are you? I am fi")
    assert done == ["Hello there.", "How are you?"] and tail == "I am fi"
    assert split_complete("Pi is 3.14 roughly. Next")[0] == ["Pi is 3.14 roughly."]


def _run(feeds, finish_text, max_chars=900):
    spoken, started, done = [], threading.Event(), threading.Event()
    result = {}

    def synth(s):
        spoken.append(s)
        return s

    def play(chunk, stop):
        time.sleep(0.01)

    st = SpeechStreamer(synth, play, on_start=started.set, on_done=lambda spoke: (result.update(spoke=spoke), done.set()), max_chars=max_chars)
    for f in feeds:
        st.feed(f)
    st.finish(finish_text)
    assert done.wait(3)
    return spoken, result, started, st


def test_speaks_sentences_as_they_complete_not_after_the_end():
    got = []
    first_spoken = threading.Event()

    def synth(s):
        got.append(s)
        first_spoken.set()
        return s

    done = threading.Event()
    st = SpeechStreamer(synth, lambda c, stop: time.sleep(0.01), on_done=lambda _s: done.set())
    st.feed("Hello there. And")  # only the first sentence is complete
    assert first_spoken.wait(2) and got == ["Hello there."]
    st.feed("Hello there. And more words. Fin")
    st.finish("Hello there. And more words. Final bit")
    assert done.wait(3)
    assert got == ["Hello there.", "And more words.", "Final bit"]


def test_markdown_and_code_are_not_read_aloud():
    text = "## Title\nSure, here is **code**.\n```python\nprint('x')\n```\nDone [S1]."
    spoken, result, _, _ = _run([text], text)
    joined = " ".join(spoken)
    assert "print" not in joined and "**" not in joined and "[S1]" not in joined and "Done" in joined
    assert result["spoke"] is True


def test_long_answers_only_read_the_opening():
    long_text = " ".join(f"Sentence number {i} is here." for i in range(200))
    spoken, _, _, _ = _run([long_text], long_text, max_chars=150)
    assert 1 <= len(spoken) <= 8


def test_stop_cuts_everything_and_reports_done():
    done = threading.Event()
    played = []

    def play(chunk, stop):
        played.append(chunk)
        stop.wait(2)

    st = SpeechStreamer(lambda s: s, play, on_done=lambda _s: done.set())
    st.feed("First sentence. Second sentence. Third sentence. ")
    time.sleep(0.1)
    st.stop()
    assert done.wait(3)
    assert len(played) == 1


def test_nothing_to_say_still_signals_done():
    done = threading.Event()
    res = {}
    st = SpeechStreamer(lambda s: s, lambda c, stop: None, on_done=lambda spoke: (res.update(spoke=spoke), done.set()))
    st.finish("```code only```")
    assert done.wait(2) and res["spoke"] is False


def test_shutdown_drops_queued_speech_and_joins_both_threads(caplog):
    entered, closed = threading.Event(), threading.Event()
    calls, played = [], []

    def synthesize(text):
        calls.append(text)
        entered.set()
        closed.wait(3)
        return text

    stream = SpeechStreamer(synthesize, lambda text, stop: played.append(text))
    stream.feed("First sentence. Second sentence. Third sentence. ")
    try:
        assert entered.wait(2)
        stream.stop()
        assert not stream.wait(timeout=.01), "Pending native synthesis should keep the producer alive"
    finally:
        closed.set()
        stream.stop()
    assert stream.wait(timeout=2)
    assert calls == ["First sentence."] and played == []
    assert not stream.has_pending and not stream.active
    assert not caplog.records
