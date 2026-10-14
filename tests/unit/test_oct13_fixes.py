"""Fixes after the Oct 12 logs: voice punctuation, capability answers, fuzzy app names, GPU voice env, MCP, skills."""

import json
import sys
import textwrap
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from lora.agent.commands import capability_reply, is_research, normalize_speech, parse_command, quick_reply
from lora.agent.fuzzy import best_key
from lora.agent.runtime import AgentRuntime
from lora.dialogue import PERSONA, ConversationHistory
from lora.safety.policy import PolicyEngine

URL = "http://127.0.0.1:11434"


def make_rt(tmp_path, monkeypatch, **kw):
    monkeypatch.setenv("LORA_NOTES_DIR", str(tmp_path / "notes"))

    def factory():
        from lora.tools.registry import ToolRegistry

        reg = ToolRegistry(media_db=MagicMock())
        reg.action_gate.policy_engine = PolicyEngine(allowed_roots=[str(tmp_path)])
        return reg

    rt = AgentRuntime(URL, None, ConversationHistory(), factory, **kw)
    rt.retry_delay = 0
    return rt


async def turn(rt, q):
    return await rt.run_turn(q, lambda s: None, lambda t, sp=True: None)


def nd(text):
    return ("\n".join(json.dumps(o) for o in ({"message": {"content": text}, "done": False}, {"message": {"content": ""}, "done": True})) + "\n").encode()


# ---- the exact transcripts from the Oct 12 terminal log ---------------------------------------------------------
@pytest.mark.parametrize(
    "heard, kind, target",
    [
        ("Can You Open, VSCORE?", "open", "vscore"),  # the comma used to stop the command from matching at all
        ("Can You, Open the downloads folder?", "open", "downloads"),
        ("Please, open my documents.", "open", "documents"),
        ("Hey, Laura, open Chrome.", "open", "chrome"),
        ("Open, notepad!", "open", "notepad"),
        ("Search, YouTube, for lofi beats?", "site_search", "youtube"),
    ],
)
def test_whisper_punctuation_no_longer_breaks_commands(heard, kind, target):
    cmd = parse_command(heard)
    assert cmd and cmd.kind == kind and cmd.target.lower() == target  # case is kept (paths), matching ignores it


def test_normalize_keeps_urls_and_filenames():
    assert normalize_speech("Open, example.com/a.b?x=1.") == "Open example.com/a.b x=1"
    assert normalize_speech("Open resume.pdf.") == "Open resume.pdf"


@pytest.mark.parametrize(
    "heard, expect_research",
    [
        ("Can You, Browse the internet?", False),
        ("I have checked for sources, but you see you can't browse the internet. What do you mean by that?", False),
        ("Can you tell me more about thermonuclear physics?", False),
        ("Can you tell me the latest news about SpaceX?", True),
        ("Can you search for rtx 5060 prices?", True),
    ],
)
def test_questions_about_lora_are_not_web_searches(heard, expect_research):
    assert is_research(heard) is expect_research


def test_can_you_browse_gets_a_truthful_yes_not_a_denial():
    assert "search the web" in capability_reply("Can You, Browse the internet?")
    assert "open apps" in capability_reply("can you open files")
    assert "Research" in capability_reply("What can you do?")
    assert capability_reply("can you search for cats") is None  # a real request, not a question about ability
    assert quick_reply("Hey, Laura, what's up?") is None


def test_persona_no_longer_tells_the_model_it_cannot_browse():
    low = PERSONA.lower()
    assert "cannot browse" not in low.replace("never say you cannot browse", "")
    assert "research the web" in low and "open apps" in low


# ---- fuzzy names, launching ------------------------------------------------------------------------------------------
def test_misheard_app_names_resolve():
    import yaml

    apps = yaml.safe_load(open("config/apps.yaml", encoding="utf-8"))["applications"]
    assert best_key("VSCORE", apps) == "vscode" and best_key("crome", apps) == "chrome"
    assert best_key("note pad", apps) == "notepad" and best_key("banana", apps) is None


@pytest.mark.asyncio
async def test_open_vscore_by_voice_launches_vs_code(tmp_path, monkeypatch):
    rt = make_rt(tmp_path, monkeypatch)
    with patch("subprocess.Popen") as popen:
        res = await turn(rt, "Can You Open, VSCORE?")
    assert res.text.startswith("Opening") and "Code" in res.text
    popen.assert_called_once()


def test_launching_code_cmd_uses_pathext_lookup(tmp_path):
    from lora.tools.control.apps import AppLauncher

    launcher = AppLauncher()
    calls = []

    def popen(argv, shell=False):
        calls.append(argv)
        if argv == ["code"]:
            raise FileNotFoundError  # CreateProcess cannot find "code" (it is code.cmd)
        return MagicMock()

    with patch("subprocess.Popen", side_effect=popen), patch("shutil.which", return_value=r"C:\Users\x\bin\code.cmd"):
        res = launcher.open_app("vscode")
    assert res["status"] == "success" and calls[-1] == [r"C:\Users\x\bin\code.cmd"]


@pytest.mark.asyncio
@respx.mock
async def test_asking_about_browsing_gets_an_instant_answer_with_no_model_call(tmp_path, monkeypatch):
    rt = make_rt(tmp_path, monkeypatch)
    res = await turn(rt, "Can You, Browse the internet?")
    assert res.kind == "smalltalk" and "search the web" in res.text and respx.calls.call_count == 0


# ---- web client: Windows certificate store, no text-less sites -----------------------------------------------------------
def test_http_client_verifies_with_the_os_certificate_store_when_available():
    import ssl

    from lora import net

    net._CTX = None
    ctx = net.ssl_context()
    assert isinstance(ctx, ssl.SSLContext) and ctx.verify_mode == ssl.CERT_REQUIRED  # verification is never disabled


def test_falls_back_to_default_certificates_without_truststore(monkeypatch):
    import ssl

    from lora import net

    monkeypatch.setitem(sys.modules, "truststore", None)
    net._CTX = None
    assert net.ssl_context().verify_mode == ssl.CERT_REQUIRED
    net._CTX = None


@pytest.mark.asyncio
@respx.mock
async def test_video_and_login_walled_sites_are_not_fetched():
    from lora.agent.research import research

    html = "".join(
        f'<div class="result"><a class="result__a" href="{u}">T{i}</a></div>'
        for i, u in enumerate(["https://www.youtube.com/watch?v=1", "https://example.org/a"])
    )
    respx.post("https://html.duckduckgo.com/html/").respond(200, text=html)
    page = respx.get("https://example.org/a").respond(200, text="<p>" + "word " * 100 + "</p>", headers={"content-type": "text/html"})
    yt = respx.get("https://www.youtube.com/watch").respond(200, text="x")
    res = await research("q")
    assert page.called and not yt.called and [s.domain for s in res.sources] == ["example.org"]


# ---- the GPU voice runs in its own environment ------------------------------------------------------------------------
@pytest.fixture
def voice_env(monkeypatch):
    from lora.voice.tts import worker

    monkeypatch.setattr(worker, "_EXTERNAL", None)
    monkeypatch.setenv("LORA_TTS_PYTHON", sys.executable)
    monkeypatch.delenv("LORA_TTS_ENGINE", raising=False)
    yield worker
    if worker._EXTERNAL is not None:
        worker._EXTERNAL.stop()
    worker._EXTERNAL = None


def test_voice_server_speaks_through_the_separate_environment(voice_env, monkeypatch):
    monkeypatch.setenv("LORA_TTS_FAKE", "1")
    w = voice_env.Qwen3TtsWorkerProcess(models_dir=r"C:\nowhere")
    assert w.preload() is True
    chunk = w.synthesize("Hello there, boss.", voice="vivian")
    assert len(chunk.samples) > 1000 and chunk.sample_rate == 22050
    assert abs(float(chunk.samples.max())) <= 1.0
    diag = w.diagnosis()
    assert diag["ok"] and "ready" in diag["summary"]
    w.shutdown()


def test_voice_server_failure_reports_reason_and_fix_and_falls_back(voice_env, monkeypatch, tmp_path):
    monkeypatch.delenv("LORA_TTS_FAKE", raising=False)
    # the real server in an interpreter without torch/qwen-tts: a clear message, then the Windows voice
    w = voice_env.Qwen3TtsWorkerProcess(models_dir=str(tmp_path))
    assert w.preload() is False
    diag = w.diagnosis()
    assert not diag["ok"] and "install_tts.ps1" in diag["fix"] and "Windows voice" in diag["summary"]
    with patch.object(voice_env, "_windows_speech_synth", return_value=None):
        chunk = w.synthesize("Hi.")
    assert chunk is not None  # no crash even when every voice is unavailable


def test_installer_creates_a_separate_environment_and_never_touches_the_main_one():
    script = open("scripts/install_tts.ps1", encoding="utf-8").read()
    assert ".venv-tts" in script and "uv pip install --python $Py" in script
    assert "uv sync" not in script and '".venv\\Scripts' not in script


# ---- MCP: a real server over stdio ------------------------------------------------------------------------------------
FAKE_MCP = textwrap.dedent(
    """
    import sys, json
    TOOLS = [
      {"name": "echo", "description": "Echo text back", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
      {"name": "write_note", "description": "Write a note", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string"}}}},
      {"name": "delete_everything", "description": "danger", "inputSchema": {"type": "object"}},
    ]
    for line in sys.stdin:
        msg = json.loads(line)
        if "id" not in msg:
            continue
        m = msg["method"]
        if m == "initialize":
            res = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "1"}}
        elif m == "tools/list":
            res = {"tools": TOOLS}
        elif m == "tools/call":
            n, a = msg["params"]["name"], msg["params"]["arguments"]
            res = {"content": [{"type": "text", "text": "echo: " + a.get("text", "")}]} if n == "echo" else {"content": [{"type": "text", "text": "wrote"}]}
        else:
            res = {}
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": res}), flush=True)
    """
)


@pytest.fixture
def mcp_rt(tmp_path, monkeypatch):
    script = tmp_path / "fake_mcp.py"
    script.write_text(FAKE_MCP)
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "fake": {"command": sys.executable, "args": [str(script)]},
        "off": {"command": "nonexistent-program", "disabled": True},
    }}))
    rt = make_rt(tmp_path, monkeypatch, mcp_config_path=cfg)
    rt.start_mcp()
    yield rt
    rt.mcp_hub.close()


def test_mcp_server_connects_registers_safe_tools_and_blocks_dangerous_ones(mcp_rt):
    hub = mcp_rt.mcp_hub
    assert list(hub.servers) == ["fake"] and not hub.errors
    names = set(mcp_rt.mcp_tool_defs())
    assert names == {"fake__echo", "fake__write_note"} and "fake.delete_everything" in hub.skipped
    from lora.safety.policy import ActionRiskTier

    assert mcp_rt.mcp_tool_defs()["fake__echo"].tier == ActionRiskTier.TIER1
    assert mcp_rt.mcp_tool_defs()["fake__write_note"].tier == ActionRiskTier.TIER2  # looks like it changes things


@pytest.mark.asyncio
async def test_mcp_tool_call_goes_through_the_gate(mcp_rt, tmp_path):
    ok = await mcp_rt.call_mcp("fake__echo", {"text": "hi"}, tainted=False)
    assert ok == {"status": "success", "text": "echo: hi"}
    tainted = await mcp_rt.call_mcp("fake__write_note", {"path": str(tmp_path / "n.md"), "text": "x"}, tainted=True)
    assert tainted["status"] in ("denied", "error")  # a write after reading untrusted content is refused
    outside = await mcp_rt.call_mcp("fake__write_note", {"path": "C:/Windows/evil.txt", "text": "x"}, tainted=False)
    assert outside["status"] in ("denied", "error")
    assert (await mcp_rt.call_mcp("fake__nope", {}, False))["status"] == "error"


@pytest.mark.asyncio
@respx.mock
async def test_free_form_loop_can_use_an_mcp_tool_and_is_tainted_afterwards(mcp_rt, tmp_path):
    script = [
        '{"tool": "fake__echo", "args": {"text": "ping"}}',
        '{"tool": "open_app", "args": {"name": "chrome"}}',
        '{"final": "The tool said echo: ping."}',
    ]
    seen = []

    def llm(request):
        seen.append(json.loads(request.content)["messages"])
        return httpx.Response(200, content=nd(script.pop(0)))

    respx.get(f"{URL}/api/tags").respond(200, json={"models": [{"name": "m:1"}]})
    respx.post(f"{URL}/api/chat").mock(side_effect=llm)
    with patch("subprocess.Popen") as popen:
        res = await turn(mcp_rt, "use fake to echo ping")
    popen.assert_not_called()  # opening an app after an external tool's output is refused
    assert res.kind == "agent" and "echo: ping" in res.text
    assert "fake__echo(text)" in seen[0][0]["content"] and "untrusted" in seen[0][0]["content"].lower()


@pytest.mark.asyncio
async def test_broken_mcp_server_is_reported_not_fatal(tmp_path, monkeypatch):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"bad": {"command": "definitely-not-installed-xyz"}}}))
    rt = make_rt(tmp_path, monkeypatch, mcp_config_path=cfg)
    rt.start_mcp()
    assert "bad" in rt.mcp_hub.errors and not rt.mcp_tool_defs()
    rt.diag_extras = lambda: {}
    from lora.agent.diagnostics import run_diagnostics

    with respx.mock:
        respx.get(f"{URL}/api/tags").mock(side_effect=httpx.ConnectError("x"))
        respx.post("https://html.duckduckgo.com/html/").respond(500)
        respx.get("https://lite.duckduckgo.com/lite/").respond(500)
        respx.get("https://www.bing.com/search").respond(500)
        respx.get("https://en.wikipedia.org/w/api.php").respond(500)
        text = await run_diagnostics(rt)
    assert "Other tools" in text and "bad" in text


# ---- skills, persona, roles -----------------------------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_skill_trigger_adds_instructions_and_persona_file_replaces_the_personality(tmp_path, monkeypatch):
    skills = tmp_path / "skills" / "paper-summary"
    skills.mkdir(parents=True)
    (skills / "skill.yaml").write_text("name: paper-summary\ntriggers: [summarise this paper]\nsystem_prompt_addon: Always end with a one-line TL;DR.\ndefault_depth: quick\n")
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "persona.md").write_text("You are Jarvis. Address the user as sir.")
    rt = make_rt(tmp_path, monkeypatch, skills_dir=tmp_path / "skills", agents_dir=agents)
    bodies = []

    def llm(request):
        bodies.append(json.loads(request.content)["messages"])
        return httpx.Response(200, content=nd("Sure, sir."))

    respx.get(f"{URL}/api/tags").respond(200, json={"models": [{"name": "m:1"}]})
    respx.post(f"{URL}/api/chat").mock(side_effect=llm)
    await turn(rt, "please summarise this paper about attention")
    system = bodies[-1][0]["content"]
    assert system.startswith("You are Jarvis") and "one-line TL;DR" in system
    await turn(rt, "explain gradient descent briefly")
    assert "TL;DR" not in bodies[-1][0]["content"]  # only when the trigger phrase matches


def test_roles_from_lora_yaml_choose_the_other_model_families():
    from lora.agent.deep import pick_judge_model
    from lora.models.selection import candidate_names, load_role

    role = load_role("verifier")
    assert role.get("name")
    assert candidate_names(None, role, "verifier")[0] == role["name"]  # that role's own model comes first
    assert candidate_names(None, load_role("writer"))[0] == "spark-x25-4b"  # the writer stays Spark
    installed = ["spark-x25-4b-8k:latest", role["name"], "granite4.2:8b"]
    judge, independent = pick_judge_model(installed, "spark-x25-4b-8k:latest")
    assert independent and judge != "spark-x25-4b-8k:latest"
