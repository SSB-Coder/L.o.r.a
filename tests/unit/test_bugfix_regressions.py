"""Regression tests for the problems reported after the Oct 11 patch."""

import json
import sys
import types
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from lora.agent.runtime import AgentRuntime
from lora.dialogue import ConversationHistory
from lora.safety.policy import PolicyEngine

URL = "http://127.0.0.1:11434"


def _runtime(tmp_path, monkeypatch, roots=None):
    monkeypatch.setenv("LORA_NOTES_DIR", str(tmp_path / "notes"))

    def factory():
        from lora.tools.registry import ToolRegistry

        reg = ToolRegistry(media_db=MagicMock())
        reg.action_gate.policy_engine = PolicyEngine(allowed_roots=roots or [str(tmp_path)])
        return reg

    rt = AgentRuntime(URL, None, ConversationHistory(), factory)
    rt.retry_delay = 0
    return rt


async def _turn(rt, q):
    return await rt.run_turn(q, lambda s: None, lambda t, sp=True: None)


# ---- "it can't open applications": the gate's per-task budget never reset ----------------------------
@pytest.mark.asyncio
async def test_opening_the_same_app_many_times_keeps_working(tmp_path, monkeypatch):
    rt = _runtime(tmp_path, monkeypatch)
    with patch("subprocess.Popen"):
        results = [await _turn(rt, "open chrome") for _ in range(8)]  # 4th used to be denied as a "loop"
    assert all(r.text == "Opening Google Chrome." for r in results), [r.text for r in results]


@pytest.mark.asyncio
async def test_action_budget_does_not_run_out_over_a_long_session(tmp_path, monkeypatch):
    rt = _runtime(tmp_path, monkeypatch)
    with patch("subprocess.Popen"), patch("webbrowser.open"):
        for i in range(60):  # budget is 40 per task; it used to be 40 per app run
            r = await _turn(rt, "open notepad" if i % 2 else "open calculator")
            assert r.text.startswith("Opening"), (i, r.text)


# ---- folders: OneDrive-redirected locations ------------------------------------------------------------
@pytest.mark.asyncio
async def test_open_folder_uses_the_real_known_folder_location(tmp_path, monkeypatch):
    onedrive_docs = tmp_path / "OneDrive" / "Documents"
    onedrive_docs.mkdir(parents=True)
    rt = _runtime(tmp_path, monkeypatch, roots=[str(tmp_path)])
    from lora.tools.files import known

    monkeypatch.setattr(known, "_windows_known_folder", lambda name: onedrive_docs if name == "documents" else None)
    with patch("os.startfile", create=True) as startfile:
        res = await _turn(rt, "open my documents folder")
    assert res.text == "Opening Documents." and startfile.call_args.args[0] == str(onedrive_docs)


def test_known_folder_roots_feed_the_default_policy(tmp_path, monkeypatch):
    from lora.tools import registry as registry_mod
    from lora.tools.files import known

    redirected = tmp_path / "OneDrive" / "Desktop"
    redirected.mkdir(parents=True)
    monkeypatch.setattr(known, "_windows_known_folder", lambda name: redirected if name == "desktop" else None)
    monkeypatch.setattr(registry_mod, "known_folder_roots", known.known_folder_roots)
    reg = registry_mod.ToolRegistry(media_db=MagicMock())
    assert reg.action_gate.policy_engine.check_path(str(redirected / "x.txt"), origin="desktop", mode="read").allowed


# ---- open a file by name, compound browse commands ----------------------------------------------------------
@pytest.mark.asyncio
async def test_open_file_by_name_finds_it_and_opens_it(tmp_path, monkeypatch):
    docs = tmp_path / "Documents"
    (docs / "sub").mkdir(parents=True)
    (docs / "sub" / "resume.pdf").write_text("x")
    rt = _runtime(tmp_path, monkeypatch, roots=[str(tmp_path)])
    from lora.tools.files import known

    monkeypatch.setattr(known, "_windows_known_folder", lambda name: docs if name == "documents" else tmp_path / "none")
    with patch("os.startfile", create=True) as startfile:
        res = await _turn(rt, "open resume.pdf")
    assert res.text == "Opening resume.pdf." and "resume.pdf" in str(startfile.call_args)
    missing = await _turn(rt, "open nothere.pdf")
    assert "couldn't find" in missing.text


@pytest.mark.asyncio
async def test_open_site_and_search_in_one_sentence(tmp_path, monkeypatch):
    rt = _runtime(tmp_path, monkeypatch)
    with patch("webbrowser.open") as wb:
        r1 = await _turn(rt, "open youtube and search for lofi beats")
        await _turn(rt, "search youtube for cats")
        r3 = await _turn(rt, "search ramen on reddit")  # no shortcut: falls back to a Google search
    urls = [c.args[0] for c in wb.call_args_list]
    assert "results?search_query=lofi+beats" in urls[0] and "search_query=cats" in urls[1] and "google.com/search" in urls[2]
    assert r1.text.startswith("Searching YouTube") and "ramen" in r3.text


# ---- "it can't browse": search backends fall through ------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_search_falls_through_to_bing_when_duckduckgo_blocks():
    from lora.agent.research import search_with_report

    respx.post("https://html.duckduckgo.com/html/").respond(202, text="bot check")
    respx.get("https://lite.duckduckgo.com/lite/").respond(403)
    bing = '<li class="b_algo"><h2><a href="https://example.org/a">Alpha</a></h2><div class="b_caption"><p>Alpha snippet</p></div></li>'
    respx.get("https://www.bing.com/search").respond(200, text=f"<ol>{bing}</ol>")
    async with httpx.AsyncClient() as client:
        hits, report = await search_with_report("alpha", client)
    assert [h.url for h in hits] == ["https://example.org/a"] and report["backend"] == "bing"
    assert any("duckduckgo: RuntimeError: HTTP 202" in e for e in report["errors"])


def test_bing_redirect_links_are_decoded():
    import base64

    from lora.agent.research import parse_bing_html

    real = "https://example.org/deep/page?x=1"
    wrapped = "https://www.bing.com/ck/a?!&&p=abc&u=a1" + base64.urlsafe_b64encode(real.encode()).decode().rstrip("=") + "&ntb=1"
    html = f'<li class="b_algo"><h2><a href="{wrapped}">T</a></h2></li>'
    assert parse_bing_html(html)[0].url == real


@pytest.mark.asyncio
@respx.mock
async def test_read_url_summarises_a_page(tmp_path, monkeypatch):
    rt = _runtime(tmp_path, monkeypatch)
    page = "<html><body><article>" + "<p>The new engine doubles throughput on long inputs. </p>" * 12 + "</article></body></html>"
    respx.get("https://example.org/post").respond(200, text=page, headers={"content-type": "text/html"})
    respx.get(f"{URL}/api/tags").respond(200, json={"models": [{"name": "m:1"}]})
    nd = "\n".join(json.dumps(o) for o in ({"message": {"content": "- Doubles throughput."}, "done": False}, {"message": {"content": ""}, "done": True})) + "\n"
    chat = respx.post(f"{URL}/api/chat").respond(200, content=nd.encode())
    res = await _turn(rt, "summarize https://example.org/post")
    assert res.kind == "research" and res.sources[0]["url"] == "https://example.org/post"
    assert "doubles throughput" in json.loads(chat.calls[0].request.content)["messages"][-1]["content"].lower()


# ---- Qwen TTS loader --------------------------------------------------------------------------------------------------
def _fake_torch(cuda=True, free_gb=5.0, version="2.8.0"):
    torch = types.ModuleType("torch")
    torch.__version__ = version
    torch.bfloat16 = "bf16"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda, mem_get_info=lambda: (int(free_gb * 1e9), int(6e9)), empty_cache=lambda: None
    )
    return torch


@pytest.fixture
def tts(monkeypatch):
    from lora.voice.tts import worker

    monkeypatch.setattr(worker, "_CACHED_QWEN_MODEL", None)
    monkeypatch.setattr(worker, "_QWEN_STATUS", {"state": "unloaded", "reason": "", "fix": "", "dir": ""})
    monkeypatch.delenv("LORA_TTS_ENGINE", raising=False)
    monkeypatch.delenv("LORA_TTS_MODEL_DIR", raising=False)
    return worker


def _model_dir(tmp_path, name, gb):
    d = tmp_path / name
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"\0" * int(gb * 1e6))  # scaled: 1 'GB' == 1 MB here
    return d


def test_cpu_only_torch_gives_a_clear_reason_and_fix(tts, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, version="2.8.0+cpu"))
    assert tts.get_qwen_model(str(tmp_path)) is None
    diag = tts.tts_diagnosis(str(tmp_path))
    assert not diag["ok"] and "no GPU support" in diag["summary"] and "install_tts.ps1" in diag["fix"]


def test_model_loads_in_bf16_not_float32(tts, monkeypatch, tmp_path):
    """Default float32 loading needs ~7 GB for the 1.7B model, which cannot fit in 6 GB."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    calls = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kw):
            calls.update(path=path, **kw)
            return cls()

    monkeypatch.setitem(sys.modules, "qwen_tts", types.SimpleNamespace(Qwen3TTSModel=FakeModel))
    monkeypatch.setattr(tts, "_weights_gb", lambda p: 3.8)
    d = _model_dir(tmp_path, "Qwen3-TTS-12Hz-1.7B-CustomVoice", 1)
    assert tts.get_qwen_model(str(d)) is not None
    assert calls["dtype"] == "bf16" and calls["device_map"] == "cuda:0"
    assert tts.tts_diagnosis(str(d))["ok"]


def test_smaller_model_is_chosen_when_the_big_one_does_not_fit(tts, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free_gb=3.0))
    loaded = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kw):
            loaded.append(path)
            return cls()

    monkeypatch.setitem(sys.modules, "qwen_tts", types.SimpleNamespace(Qwen3TTSModel=FakeModel))
    big = _model_dir(tmp_path, "Qwen3-TTS-12Hz-1.7B-CustomVoice", 1)
    small = _model_dir(tmp_path, "Qwen3-TTS-12Hz-0.6B-CustomVoice", 1)
    monkeypatch.setattr(tts, "_weights_gb", lambda p: 3.8 if "1.7B" in str(p) else 1.3)
    assert tts.get_qwen_model(str(big)) is not None
    assert loaded == [str(small)]


def test_not_enough_vram_says_so_and_does_not_retry_every_sentence(tts, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free_gb=1.0))
    monkeypatch.setitem(sys.modules, "qwen_tts", types.SimpleNamespace(Qwen3TTSModel=MagicMock()))
    d = _model_dir(tmp_path, "Qwen3-TTS-12Hz-1.7B-CustomVoice", 1)
    monkeypatch.setattr(tts, "_weights_gb", lambda p: 3.8)
    assert tts.get_qwen_model(str(d)) is None
    diag = tts.tts_diagnosis(str(d))
    assert "not enough free GPU memory" in diag["summary"] and "0.6B" in diag["fix"]
    sys.modules["qwen_tts"].Qwen3TTSModel.from_pretrained.assert_not_called()


def test_speaker_names_are_canonicalised(tts):
    model = MagicMock()
    model.get_supported_speakers.return_value = ["Vivian", "Uncle_Fu"]
    assert tts._canonical_speaker(model, "uncle_fu") == "Uncle_Fu"
    assert tts._canonical_speaker(model, "nobody") == "Vivian"
    plain = types.SimpleNamespace()
    assert tts._canonical_speaker(plain, "ono_anna") == "Ono_Anna"


# ---- diagnostics ------------------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_diagnostics_reports_problems_with_fixes_and_no_model_names(tmp_path, monkeypatch):
    rt = _runtime(tmp_path, monkeypatch)
    rt.diag_extras = lambda: {
        "mic": (False, "no input device found"),
        "tts": {"ok": False, "summary": "using the Windows voice (this PyTorch has no GPU support).", "fix": "run scripts\\install_tts.ps1"},
    }
    respx.get(f"{URL}/api/tags").respond(200, json={"models": [{"name": "SparkLLM/Spark-X2.5-4B:latest"}]})
    respx.post("https://html.duckduckgo.com/html/").respond(202)
    respx.get("https://lite.duckduckgo.com/lite/").respond(403)
    respx.get("https://www.bing.com/search").respond(200, text="<html></html>")
    respx.get("https://en.wikipedia.org/w/api.php").mock(side_effect=httpx.ConnectError("offline"))
    res = await _turn(rt, "run diagnostics")
    assert "need attention" in res.text
    assert "Browsing: I can't reach any search engine" in res.text
    assert "install_tts.ps1" in res.text and "microphone problem" in res.text
    assert "spark" not in res.text.lower()



def test_v8_sleep_wakeword_word_boundaries_and_inline_query():
    from lora.voice.wakeword import WakeWordDetector

    det = WakeWordDetector()
    # Words containing "awaken" as a substring must NOT false-trigger from sleep
    matched_false, _ = det.match_text("The awakening of spring was loud", is_sleeping=True)
    assert matched_false is False

    # Saying "Awaken LORA, open Chrome" from sleep wakes and extracts the inline command
    matched_true, inline_q = det.match_text("Awaken LORA, open Chrome", is_sleeping=True)
    assert matched_true is True
    assert inline_q == "open Chrome"


def test_v8_ssrf_blocks_172_16_and_link_local_and_ipv6():
    from lora.safety.egress import EgressFilter

    ef = EgressFilter()
    assert ef.check_egress("http://172.16.0.5:8080/admin").allowed is False
    assert ef.check_egress("http://169.254.169.254/latest/meta-data/").allowed is False
    assert ef.check_egress("http://[fe80::1]/").allowed is False
    assert ef.check_egress("http://[fc00::1]/").allowed is False


def test_v8_mobile_hud_media_routes_blocked(tmp_path):
    from fastapi.testclient import TestClient
    from lora.gateway.auth import TokenValidator
    from lora.gateway.server import create_gateway_app

    tv = TokenValidator(desktop_token="desk_tok", mobile_token="mob_tok")
    mob_app = create_gateway_app(origin="mobile", token_validator=tv)
    client = TestClient(mob_app, base_url="http://192.168.50.10:8765")

    assert client.get("/api/hud/status").status_code == 401
    assert client.get("/api/hud/status", headers={"Authorization": "Bearer mob_tok"}).status_code == 200
    assert client.post("/api/hud/media/add", json={"path": "C:\\Users\\Public\\secret.txt"}, headers={"Authorization": "Bearer mob_tok"}).status_code == 403
    assert client.get("/api/hud/media/list", headers={"Authorization": "Bearer mob_tok"}).status_code == 403
    assert client.post("/api/hud/media/report", json={"query": "secret"}, headers={"Authorization": "Bearer mob_tok"}).status_code == 403


def test_v8_hybrid_session_index_rrf_and_chunk_cap():
    from lora.grounding.chunk import DocumentChunk
    from lora.grounding.embed import EmbeddingEngine
    from lora.grounding.session_index import SessionVectorIndex

    eng = EmbeddingEngine(model_name_or_path="nonexistent_fallback_model")
    idx = SessionVectorIndex(session_id="test-v8", embedding_engine=eng, max_chunks=16)
    chunks = [
        DocumentChunk(
            chunk_id=f"c{i}",
            text=f"Passage number {i} discussing quantum error correction surface codes and syndrome decoding.",
            source_url="https://arxiv.org/abs/2401.00001",
            title="Quantum Paper",
            token_count=14,
        )
        for i in range(25)
    ]
    idx.add_chunks(chunks)
    assert idx.chunk_count == 16
    hybrid_hits = idx.search_hybrid("syndrome decoding surface codes", top_k=3)
    assert len(hybrid_hits) == 3


def test_v8_contradiction_detection_and_preprint_flag():
    from lora.grounding.cards import EvidenceCard, EvidenceCardExtractor
    from lora.grounding.chunk import DocumentChunk
    from lora.verify.corroborate import CorroborationChecker

    extractor = EvidenceCardExtractor()
    cards = extractor.extract_cards([
        DocumentChunk(
            chunk_id="c1",
            text="Room temperature superconductivity was observed in copper-substituted apatite samples.",
            source_url="https://arxiv.org/abs/2307.12008",
            title="Preprint Claim",
            token_count=12,
            tier=1,
        )
    ])
    assert len(cards) == 1
    assert cards[0].is_preprint is True
    assert cards[0].reliability_tier == "T1-P"

    c_neg = EvidenceCard(
        card_id="S2",
        claim_fact="Room temperature superconductivity was not observed in copper-substituted apatite samples.",
        exact_quote="Room temperature superconductivity was not observed in copper-substituted apatite samples.",
        source_url="https://nature.com/articles/s41586-023-06522",
        title="Replication Study",
        tier=1,
    )
    cor = CorroborationChecker().check_corroboration([cards[0], c_neg])
    assert cor.is_corroborated is True
    assert len(cor.contradictions) >= 1


def test_v8_csv_formula_injection_blocked_in_open_path(tmp_path):
    from lora.safety.policy import PolicyEngine
    from lora.tools.files.paths import open_path

    evil_csv = tmp_path / "payload.csv"
    evil_csv.write_text("Name,Formula\nAlice,=cmd|'/c calc'!A0\n", encoding="utf-8")
    policy = PolicyEngine(allowed_roots=[str(tmp_path)])
    res = open_path(evil_csv, policy_engine=policy)
    assert res["status"] == "error"
    assert "formula" in res["message"].lower() or "dde" in res["message"].lower()
