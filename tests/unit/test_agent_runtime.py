"""AgentRuntime with the real ToolRegistry/policy/gate, a fake Ollama (respx) and no real launches."""

import json
from unittest.mock import MagicMock, patch

import pytest
import respx

from lora.agent.research import ResearchResult, Source
from lora.agent.runtime import AgentRuntime
from lora.dialogue import ConversationHistory
from lora.safety.policy import PolicyEngine

URL = "http://127.0.0.1:11434"
TAGS = {"models": [{"name": "ornith-1.5:9b"}, {"name": "SparkLLM/Spark-X2.5-4B:latest"}]}


def _nd(*tokens):
    objs = [{"message": {"content": t}, "done": False} for t in tokens] + [{"message": {"content": ""}, "done": True}]
    return ("\n".join(json.dumps(o) for o in objs) + "\n").encode()


@pytest.fixture
def notes_dir(tmp_path, monkeypatch):
    d = tmp_path / "notes"
    monkeypatch.setenv("LORA_NOTES_DIR", str(d))
    return d


@pytest.fixture
def runtime(tmp_path, notes_dir):
    def factory():
        from lora.tools.registry import ToolRegistry

        reg = ToolRegistry(media_db=MagicMock())
        reg.action_gate.policy_engine = PolicyEngine(allowed_roots=[str(tmp_path)])
        return reg

    rt = AgentRuntime(URL, "ornith-1.5:9b", ConversationHistory(), factory)
    rt.retry_delay = 0
    return rt


class Sink:
    def __init__(self):
        self.status, self.texts = [], []

    def on_status(self, t):
        self.status.append(t)

    def on_text(self, full, speak=True):
        self.texts.append((full, speak))


async def run(rt, q):
    s = Sink()
    res = await rt.run_turn(q, s.on_status, s.on_text)
    return res, s


@pytest.mark.asyncio
@respx.mock
async def test_greeting_is_instant_and_never_touches_the_model(runtime):
    res, sink = await run(runtime, "hello")
    assert res.kind == "smalltalk" and res.text and sink.texts[0][0] == res.text
    assert respx.calls.call_count == 0


@pytest.mark.asyncio
@respx.mock
async def test_chat_uses_spark_not_the_stale_default_and_streams(runtime):
    respx.get(f"{URL}/api/tags").respond(200, json=TAGS)
    route = respx.post(f"{URL}/api/chat").respond(200, content=_nd("Sure", ", boss."))
    res, sink = await run(runtime, "explain attention in one line")
    assert res.text == "Sure, boss." and res.kind == "chat"
    assert json.loads(route.calls[0].request.content)["model"] == "SparkLLM/Spark-X2.5-4B:latest"
    assert [t for t, _ in sink.texts][-1] == "Sure, boss."


@pytest.mark.asyncio
async def test_open_desktop_app_through_the_real_registry(runtime):
    with patch("subprocess.Popen") as popen:
        res, _ = await run(runtime, "open chrome")
    popen.assert_called_once_with(["chrome.exe"], shell=False)
    assert res.kind == "command" and res.text == "Opening Google Chrome."


@pytest.mark.asyncio
async def test_open_web_app_and_url(runtime):
    with patch("webbrowser.open") as wb:
        r1, _ = await run(runtime, "open youtube")
        r2, _ = await run(runtime, "go to https://example.com".replace("go to", "open"))
        r3, _ = await run(runtime, "google best ramen in bengaluru")
    urls = [c.args[0] for c in wb.call_args_list]
    assert "youtube.com" in urls[0] and urls[1] == "https://example.com" and "google.com/search?q=best+ramen" in urls[2]
    assert all(r.text and r.kind == "command" for r in (r1, r2, r3))


@pytest.mark.asyncio
async def test_blocked_and_unknown_targets_are_refused_politely(runtime):
    with patch("webbrowser.open") as wb, patch("subprocess.Popen") as popen:
        r1, _ = await run(runtime, "open http://127.0.0.1:8765/api/hud/media/list")
        r2, _ = await run(runtime, "open powershell")
        r3, _ = await run(runtime, "open C:\\Windows\\System32\\cmd.exe")
    wb.assert_not_called()
    popen.assert_not_called()
    assert "on my list" in r2.text
    assert r1.text and r3.text


@pytest.mark.asyncio
async def test_open_and_list_folder_inside_allowed_roots(runtime, tmp_path):
    (tmp_path / "a.txt").write_text("x")
    with patch("os.startfile", create=True) as sf, patch("subprocess.Popen"), patch("webbrowser.open"):
        res, _ = await run(runtime, f"list the files in {tmp_path}")
    assert "a.txt" in res.text
    res2, _ = await run(runtime, "list the files in C:\\Windows")
    assert "not allowed" in res2.text or "couldn't" in res2.text
    del sf


@pytest.mark.asyncio
@respx.mock
async def test_ambiguous_verb_falls_through_to_chat(runtime):
    respx.get(f"{URL}/api/tags").respond(200, json=TAGS)
    respx.post(f"{URL}/api/chat").respond(200, content=_nd("Cats are great."))
    res, _ = await run(runtime, "start a conversation about cats")
    assert res.kind == "chat" and res.text == "Cats are great."


@pytest.mark.asyncio
@respx.mock
async def test_create_note_end_to_end_and_spoken_summary(runtime, notes_dir):
    respx.get(f"{URL}/api/tags").respond(200, json=TAGS)
    respx.post(f"{URL}/api/chat").respond(200, content=_nd("# Quantum tunneling\n", "A particle crosses a barrier."))
    res, sink = await run(runtime, "create a markdown note about quantum tunneling")
    files = list(notes_dir.glob("*.md"))
    assert len(files) == 1 and "crosses a barrier" in files[0].read_text(encoding="utf-8")
    assert res.kind == "note" and "Saved to" in res.text and res.spoken.startswith("Done. I saved it")
    assert all(speak is False for _, speak in sink.texts)  # the document body is not read aloud


@pytest.mark.asyncio
@respx.mock
async def test_save_last_answer_with_sources(runtime, notes_dir):
    respx.get(f"{URL}/api/tags").respond(200, json=TAGS)
    respx.post(f"{URL}/api/chat").respond(200, content=_nd("Mamba is linear-time [S1]."))
    src = ResearchResult("q", [Source("S1", "Mamba paper", "https://arxiv.org/abs/2312.00752", text="t" * 300)])

    async def fake_research(q, status):
        status("Searching the web…")
        return src

    runtime._research_fn = fake_research
    res, sink = await run(runtime, "research mamba models")
    assert res.kind == "research" and res.sources[0]["url"].startswith("https://arxiv.org") and "Searching the web…" in sink.status and "Planning the search…" in sink.status
    saved, _ = await run(runtime, "save that as a markdown file")
    note = next(notes_dir.glob("*.md")).read_text(encoding="utf-8")
    assert "Mamba is linear-time" in note and "[Mamba paper](https://arxiv.org/abs/2312.00752)" in note
    assert "Saved it to" in saved.text


@pytest.mark.asyncio
@respx.mock
async def test_research_prompt_carries_sources_and_rules(runtime):
    respx.get(f"{URL}/api/tags").respond(200, json=TAGS)
    route = respx.post(f"{URL}/api/chat").respond(200, content=_nd("ok"))

    async def fake_research(q, status):
        return ResearchResult(q, [Source("S1", "T", "https://example.org/x", text="Evidence text.")])

    runtime._research_fn = fake_research
    await run(runtime, "what is the latest on spacex starship")
    bodies = [json.loads(c.request.content)["messages"] for c in route.calls]
    msgs = next(m for m in bodies if "ONLY the numbered sources" in m[0]["content"])  # the writer call, after the planning call
    assert "[S1] T (example.org)" in msgs[-1]["content"]


@pytest.mark.asyncio
@respx.mock
async def test_engine_down_message_is_friendly_and_model_free(runtime):
    import httpx

    respx.get(f"{URL}/api/tags").mock(side_effect=httpx.ConnectError("refused"))
    res, _ = await run(runtime, "explain attention in one line")
    assert res.failure and "local engine" in res.text
    assert "spark" not in res.text.lower() and "ornith" not in res.text.lower()


@pytest.mark.asyncio
async def test_registry_has_no_delete_shell_or_overwrite_tools(runtime):
    runtime._get_registry()
    names = [t["name"] for t in runtime._registry.list_tools()]
    assert "create_note" in names
    assert not [n for n in names if any(w in n for w in ("delete", "remove_file", "rm", "exec", "shell", "run_command", "write_file"))
                and n not in ("remove_media_file",)]
