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


def test_quick_replies():
    assert quick_reply("hello") and quick_reply("Hey LORA") and quick_reply("thanks")
    assert quick_reply("hello, can you explain gradient descent") is None


def test_note_title():
    assert note_title("write a note called Trip Ideas") == "Trip Ideas"
    assert note_title("create a note about quantum tunneling") == "quantum tunneling"
