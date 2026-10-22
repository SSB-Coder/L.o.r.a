"""Fixes after the Oct 12 logs: voice punctuation, capability answers, fuzzy app names, GPU voice env, MCP, skills."""

import json
import sys
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
    assert "tony stark" not in low and 'never address them as "stark"' in low


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


# ---- CPU voice and deliberately deferred MCP ----
def test_voice_failure_keeps_text_working_without_another_speech_backend(monkeypatch):
    from lora.voice.tts import worker
    monkeypatch.setattr(worker, "get_kokoro_model", lambda path: None)
    w = worker.KokoroTtsWorkerProcess()
    assert not w.preload()
    assert not w.diagnosis()["ok"]
    assert "--kokoro-only" in w.diagnosis()["fix"]
    chunk = w.synthesize("Hi.")
    assert chunk.error and len(chunk.samples) == 0


def test_installer_uses_locked_runtime_and_kokoro_cpu_only():
    from pathlib import Path
    script = Path("scripts/install_tts.ps1").read_text(encoding="utf-8")
    assert "uv sync --locked" in script and "--kokoro-only" in script
    assert "KokoroTtsWorkerProcess" in script
    assert "qwen" not in script.lower() and "cuda" not in script.lower()


@pytest.mark.parametrize("command", [sys.executable, "definitely-not-installed-xyz"])
def test_mcp_configuration_never_launches_servers(tmp_path, monkeypatch, command):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"fake": {"command": command}}}))
    rt = make_rt(tmp_path, monkeypatch, mcp_config_path=cfg)
    with patch("subprocess.Popen") as popen:
        rt.start_mcp()
        rt.start_mcp()
    popen.assert_not_called()
    assert rt.mcp_hub is None and rt.mcp_tool_defs() == {}
    assert not rt._mentions_mcp("use fake to echo ping")


@pytest.mark.asyncio
@pytest.mark.parametrize("tainted", [True, False])
async def test_deferred_mcp_cannot_execute_any_tool(tmp_path, monkeypatch, tainted):
    rt = make_rt(tmp_path, monkeypatch)
    with patch("subprocess.Popen") as popen:
        result = await rt.call_mcp("fake__echo", {"text": "hi"}, tainted=tainted)
    assert result["status"] == "error" and "Unknown tool" in result["message"]
    popen.assert_not_called()


# ---- skills, persona, roles -----------------------------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_skill_trigger_adds_instructions_and_persona_file_replaces_the_personality(tmp_path, monkeypatch):
    skills = tmp_path / "skills" / "paper-summary"
    skills.mkdir(parents=True)
    (skills / "skill.yaml").write_text("name: paper-summary\ntriggers: [summarise this paper]\nsystem_prompt_addon: Always end with a one-line TL;DR.\ndefault_depth: quick\n")
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "persona.md").write_text("You are a chill study buddy. Keep replies casual.")
    rt = make_rt(tmp_path, monkeypatch, skills_dir=tmp_path / "skills", agents_dir=agents)
    bodies = []

    def llm(request):
        bodies.append(json.loads(request.content)["messages"])
        return httpx.Response(200, content=nd("Got it!"))

    respx.get(f"{URL}/api/tags").respond(200, json={"models": [{"name": "m:1"}]})
    respx.post(f"{URL}/api/chat").mock(side_effect=llm)
    query = "please summarise this paper about attention"
    kwargs = rt._persona_kwargs(query)
    assert kwargs["persona"].startswith("You are a chill study buddy")
    assert "one-line TL;DR" in kwargs["addon"]
    await turn(rt, "tell me a joke about cats")
    system = bodies[-1][0]["content"]
    assert system.startswith("You are a chill study buddy") and "TL;DR" not in system
    await turn(rt, "how are you doing")
    assert "TL;DR" not in bodies[-1][0]["content"]  # only when the trigger phrase matches


def test_roles_from_lora_yaml_choose_the_other_model_families():
    from lora.agent.deep import pick_judge_model
    from lora.models.selection import candidate_names, load_role

    role = load_role("verifier")
    assert role.get("name")
    assert candidate_names(None, role, "verifier")[0] == role["name"]  # that role's own model comes first
    assert candidate_names(None, load_role("writer"))[0] == "spark-x25-4b"  # the base writer stays Spark 4B
    installed = ["spark-x25-4b-8k:latest", role["name"], "granite4.2:8b"]
    judge, independent = pick_judge_model(installed, "spark-x25-4b-8k:latest")
    assert independent and judge != "spark-x25-4b-8k:latest"


# ---- v5 Agentic Harness, Kokoro TTS (af_sky), Model Switching, 8-Way Verification & Artifacts -------------------------
def test_kokoro_tts_default_voice_is_af_sky(tmp_path, monkeypatch):
    import numpy as np
    import types
    from lora.voice.tts import worker as tts_worker

    assert tts_worker.DEFAULT_KOKORO_VOICE == "af_sky"
    assert tts_worker.resolve_kokoro_voice(None) == "af_sky"
    assert tts_worker.resolve_kokoro_voice("vivian") == "af_sky"
    assert tts_worker.resolve_kokoro_voice("bm_george") == "bm_george"

    kokoro_dir = tmp_path / "kokoro"
    kokoro_dir.mkdir()
    (kokoro_dir / "kokoro-v1.0.onnx").write_bytes(b"fake_onnx")
    (kokoro_dir / "voices-v1.0.bin").write_bytes(b"fake_voices")

    calls = {}

    class FakeKokoro:
        @classmethod
        def from_session(cls, session, voices_path):
            calls["voices_path"] = voices_path
            return cls()

        def get_voices(self):
            return ["af_sky", "af_heart", "bm_george"]

        def create(self, text: str, voice: str = "af_sky", speed: float = 1.0, lang: str = "en-us"):
            calls["voice"] = voice
            calls["lang"] = lang
            return np.ones(2400, dtype=np.float32) * 0.2, 24000

    monkeypatch.setattr(tts_worker, "_CACHED_KOKORO_MODEL", None)
    monkeypatch.setattr(tts_worker, "_KOKORO_STATUS", {"state": "unloaded", "reason": "", "fix": "", "dir": ""})
    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=FakeKokoro))

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(
        SessionOptions=lambda: types.SimpleNamespace(),
        InferenceSession=lambda *a, **kw: calls.update(providers=kw["providers"]),
    ))
    w = tts_worker.KokoroTtsWorkerProcess(models_dir=str(kokoro_dir))
    assert w.default_voice == "af_sky"
    chunk = w.synthesize("Hello from Kokoro Sky!")
    assert len(chunk.samples) == 2400 and chunk.sample_rate == 24000
    assert calls["voice"] == "af_sky" and calls["lang"] == "en-us"
    assert w.diagnosis()["ok"] is True and calls["providers"] == ["CPUExecutionProvider"]


@pytest.mark.asyncio
async def test_open_wikipedia_on_page_and_open_chatgpt_and_ask(tmp_path, monkeypatch, public_dns):
    c_wiki = parse_command("Open wikipedia on Alan Turing")
    assert c_wiki is not None and c_wiki.kind == "site_search"
    assert c_wiki.target == "wikipedia" and c_wiki.args["query"] == "Alan Turing"

    c_wiki_rev = parse_command("Open quantum computing on wikipedia")
    assert c_wiki_rev is not None and c_wiki_rev.kind == "site_search"
    assert c_wiki_rev.target == "wikipedia" and c_wiki_rev.args["query"] == "quantum computing"

    c_ai = parse_command("open chatgpt and ask how transformers work")
    assert c_ai is not None and c_ai.kind == "ask_ai"
    assert c_ai.args["service"] == "chatgpt" and "transformers" in c_ai.args["prompt"]

    rt = make_rt(tmp_path, monkeypatch)
    with patch("webbrowser.open") as wb:
        res_wiki = await turn(rt, "Open wikipedia on Alan Turing")
    assert res_wiki.kind == "command"
    assert "Searching Wikipedia for “Alan Turing”" in res_wiki.text
    assert "en.wikipedia.org/wiki/Special:Search?search=Alan+Turing" in wb.call_args.args[0]


def test_manual_and_automatic_model_selection():
    from lora.models.selection import QWYTHOS_MODEL, candidate_names, is_complex_query, pick_escalation_model

    cands_manual = candidate_names("ornith-1.5:9b", {"name": "spark-x25-4b"}, role="writer", manual_model="granite4.2:8b")
    assert cands_manual[0] == "granite4.2:8b"

    cands_complex = candidate_names(None, {}, role="complex")
    assert cands_complex[0] == "granite4.2:8b"
    assert QWYTHOS_MODEL in cands_complex

    cands_vision = candidate_names(None, {}, role="vision")
    assert "gemma4:e4b" in cands_vision[:2] and "gemma3:4b" in cands_vision[:3]

    esc = pick_escalation_model(["SparkLLM/Spark-X2.5-4B:latest", "granite4.2:8b"], "SparkLLM/Spark-X2.5-4B:latest")
    assert esc == "granite4.2:8b"
    assert is_complex_query("Compare the architectural trade-offs between Mamba and Transformers in depth")


@pytest.mark.asyncio
async def test_simple_questions_get_clean_prompts_not_bloated_rubrics():
    from lora.dialogue import is_simple_question
    from lora.tools.browser.prompt_agent import PromptOptimizerAgent, PromptStyle, is_simple_prompt

    assert is_simple_prompt("what is 2 + 2?")
    assert is_simple_question("what is the capital of France?")
    assert not is_simple_prompt("Research the literature on quantum error correction architectures")

    opt = PromptOptimizerAgent()
    res_simple = await opt.optimize("what is the capital of Japan?", target_ai="chatgpt", style="auto", deep=False)
    assert res_simple.style == PromptStyle.SIMPLE
    assert res_simple.optimized_prompt == "what is the capital of Japan?"
    assert "Executive Summary" not in res_simple.optimized_prompt


@pytest.mark.asyncio
@respx.mock
async def test_research_displays_reference_check_without_scorecard(tmp_path, monkeypatch):
    from lora.agent.research import ResearchResult, Source

    rt = make_rt(tmp_path, monkeypatch)
    page_text = "Mamba is a selective state space architecture that scales linearly with sequence length. " * 4

    async def _fake_res(q, st, queries=None, topic_type="background"):
        return ResearchResult(q, [Source("S1", "Mamba Paper", "https://arxiv.org/abs/2312.00752", text=page_text)])

    rt._research_fn = _fake_res

    respx.get(f"{URL}/api/tags").respond(200, json={"models": [{"name": "SparkLLM/Spark-X2.5-4B:latest"}, {"name": "granite4.2:8b"}]})
    respx.post(f"{URL}/api/chat").respond(200, content=nd("Mamba scales linearly with sequence length [S1]."))

    res = await turn(rt, "verified research on mamba state space models")
    assert res.meta["source_check"]["status"] == "references_checked"
    assert "verification_loop" not in res.meta and "checked" not in res.meta
    assert "score" not in res.meta["source_check"]


def test_builtin_skills_and_subagents_loaded():
    from lora.agent.runtime import SUBAGENTS
    from lora.skills.loader import SkillLoader

    loader = SkillLoader(include_builtin=True)
    skills = loader.load_skills()
    expected = {"deep-research", "web-navigator", "code-architect", "artifact-studio", "visual-creator", "truth-auditor"}
    assert expected.issubset(set(skills.keys()))
    assert set(SUBAGENTS.keys()) == {
        "supervisor", "system_navigator", "research", "workspace_code", "artifact_studio", "critic_verifier"
    }


def test_multimodal_attachments_and_artifact_generation(tmp_path):
    import base64
    from pathlib import Path
    from lora.agent.notes import create_artifact, generate_visual, process_attachments

    art_res = create_artifact("fibonacci", "def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\n", ext="py", folder=tmp_path)
    assert art_res["status"] == "success"
    assert art_res["artifact"]["ext"] == "py" and Path(art_res["path"]).is_file()

    vis_res = generate_visual("Agent Pipeline", "1. Planner\n2. Specialist\n3. 8-Way Verifier", folder=tmp_path)
    assert vis_res["status"] == "success"
    assert vis_res["artifact"]["kind"] == "image"
    assert vis_res["artifact"]["data_url"].startswith("data:image/svg+xml;base64,")

    tiny_png = base64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    ).decode("ascii")
    imgs, ctx, _ = process_attachments([
        {"name": "paste.png", "mime": "image/png", "data_url": f"data:image/png;base64,{tiny_png}"},
        {"name": "notes.py", "mime": "text/x-python", "text": "print('hello world')"},
    ])
    assert len(imgs) == 1 and imgs[0] == tiny_png
    assert "notes.py" in ctx and "print('hello world')" in ctx
