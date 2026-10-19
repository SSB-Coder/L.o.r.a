import pytest

from lora.agent.commands import is_research, note_title, parse_command, quick_reply


@pytest.mark.parametrize(
    "text, kind, target",
    [
        ("open chrome", "open", "chrome"),
        ("Hey LORA, open the downloads folder", "open", "downloads"),
        ("can you open youtube", "open", "youtube"),
        ("open https://example.com", "open", "https://example.com"),
        ("launch notepad please", "open", "notepad"),
        ("show me the downloads folder", "open", "downloads"),
        ("google best ramen in bengaluru", "google", "best ramen in bengaluru"),
        ("list the files in documents", "list_dir", "documents"),
    ],
)
def test_open_style_commands(text, kind, target):
    cmd = parse_command(text)
    assert cmd and cmd.kind == kind and cmd.target == target


def test_note_and_save_commands():
    assert parse_command("create a markdown note about quantum tunneling").args["ext"] == "md"
    assert parse_command("write a text file called groceries").args["ext"] == "txt"
    assert parse_command("make a note of that").kind == "save_last"
    assert parse_command("save that as a txt file").args["ext"] == "txt"


def test_ask_ai_command():
    cmd = parse_command("ask chatgpt to explain transformers simply")
    assert cmd.kind == "ask_ai" and cmd.args["service"] == "chatgpt" and "transformers" in cmd.args["prompt"]


@pytest.mark.parametrize("text", ["what's up", "how was your day today", "explain attention", "tell me a joke"])
def test_ordinary_chat_is_not_a_command_or_research(text):
    assert parse_command(text) is None
    assert not is_research(text)


@pytest.mark.parametrize(
    "text", ["what is the latest news on spacex", "research mamba state space models", "search the web for rtx 5060 price"]
)
def test_research_detection(text):
    assert is_research(text)


from datetime import datetime

def test_quick_replies():
    # Greetings flow directly to the LLM so answers are never hardcoded or canned
    assert quick_reply("hello") is None
    assert quick_reply("Hey LORA") is None
    assert quick_reply("good morning") is None
    assert quick_reply("good evening") is None
    assert quick_reply("hello, can you explain gradient descent") is None

    t_fixed = datetime(2026, 10, 18, 23, 38)
    assert quick_reply("what time is it", now=t_fixed) == "It's 11:38 PM."
    assert quick_reply("what's the time", now=t_fixed) == "It's 11:38 PM."
    assert quick_reply("Hey Lora, what time is it?", now=t_fixed) == "It's 11:38 PM."
    assert quick_reply("tell me the time", now=t_fixed) == "It's 11:38 PM."
    assert quick_reply("time check", now=t_fixed) == "It's 11:38 PM."
    assert quick_reply("what is the current time", now=t_fixed) == "It's 11:38 PM."
    assert quick_reply("what day is it today", now=t_fixed) == "Today is Sunday, October 18, 2026."
    assert quick_reply("what's today's date", now=t_fixed) == "Today is Sunday, October 18, 2026."
    assert quick_reply("what is today's date", now=t_fixed) == "Today is Sunday, October 18, 2026."

    # World time queries should NOT be answered by local quick reply
    assert quick_reply("what time is it in Tokyo", now=t_fixed) is None
    assert quick_reply("what time is it in London", now=t_fixed) is None


def test_temporal_context_injection():
    from lora.dialogue import ConversationHistory, build_messages, build_research_messages, temporal_context

    t_fixed = datetime(2026, 10, 18, 23, 38)
    tc = temporal_context(now=t_fixed)
    assert "Sunday, October 18, 2026" in tc
    assert "11:38 PM" in tc

    h = ConversationHistory()
    msgs = build_messages("summarize my day", h)
    assert "Current local time:" in msgs[0]["content"]

    r_msgs = build_research_messages("quantum computing", "[S1] test", h)
    assert "Current local time:" in r_msgs[0]["content"]


def test_note_title():
    assert note_title("write a note called Trip Ideas") == "Trip Ideas"
    assert note_title("create a note about quantum tunneling") == "quantum tunneling"


@pytest.mark.asyncio
async def test_runtime_sleep_command_lifecycle():
    from lora.agent.runtime import AgentRuntime
    from lora.dialogue import ConversationHistory

    power_calls = []
    texts_spoken = []

    rt = AgentRuntime("http://127.0.0.1:11434", None, ConversationHistory(), None)
    rt.on_power_command = power_calls.append
    cmd = parse_command("go to sleep")
    assert cmd is not None and cmd.kind == "sleep"

    res = await rt._run_command(cmd, "go to sleep", lambda s: None, lambda t, sp=True: texts_spoken.append(t))
    assert res is not None
    assert res.kind == "command"
    assert res.meta.get("power_action") == "sleep"
    assert "Going to sleep" in res.text
    assert power_calls == ["sleep"]

