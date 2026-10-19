from lora.dialogue import ConversationHistory, build_messages, speakable, strip_think


def test_strip_think_complete_and_unclosed():
    assert strip_think("<think>plan</think>Hello") == "Hello"
    assert strip_think("Hi <think>still thinking", streaming=True) == "Hi "
    assert strip_think("Hi <think>still thinking").strip() == "Hi"
    assert strip_think("<THINK>x</THINK>\nAnswer") == "Answer"


def test_speakable_removes_markdown_urls_and_citations():
    text = "## Title\n**Bold** text [S1] see [docs](https://x.io/a) and https://y.io\n- item one\n```py\ncode\n```"
    out = speakable(text)
    assert "http" not in out and "[S1]" not in out and "```" not in out and "**" not in out
    assert "Bold text" in out and "docs" in out and "item one" in out


def test_history_keeps_last_turns_only():
    h = ConversationHistory(max_turns=2)
    for i in range(4):
        h.add(f"q{i}", f"a{i}")
    assert [m["content"] for m in h.messages()] == ["q2", "a2", "q3", "a3"]


def test_build_messages_order_and_profile():
    h = ConversationHistory()
    h.add("hi", "hello")
    msgs = build_messages("next?", h, profile_context="likes brevity")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "likes brevity" in msgs[0]["content"] and msgs[-1]["content"] == "next?"


def test_day_period_and_time_awareness():
    from datetime import datetime
    from lora.dialogue import day_period, time_awareness

    # 1:50 AM is strictly middle of the night, not morning
    assert day_period(1) == "middle of the night"
    assert day_period(4) == "middle of the night"
    # Morning begins at 5 AM
    assert day_period(5) == "morning"
    assert day_period(11) == "morning"
    assert day_period(12) == "afternoon"

    t_150am = datetime(2026, 10, 19, 1, 50)
    awareness = time_awareness(t_150am)
    assert "middle of the night" in awareness
    assert "Morning is strictly 5:00 AM" in awareness

