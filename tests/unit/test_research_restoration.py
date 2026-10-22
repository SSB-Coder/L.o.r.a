"""User-reported desktop regressions: routing, source priority and verifier integrity."""

import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lora.agent.deep import opinion_prompt, parse_opinion, refine
from lora.agent.research import ResearchResult, Source
from lora.agent.runtime import AgentRuntime
from lora.agent.verification import Verification, VerifierLLM, verify_answer
from lora.dialogue import ConversationHistory
from lora.verify.llm_check import LlmFactChecker


def runtime(media=None):
    rt = AgentRuntime("http://127.0.0.1:11434", None, ConversationHistory(), MagicMock(), media_db=media)
    rt._stream = AsyncMock(return_value="Redshift stretches light to longer wavelengths [S1].")
    rt._collect = AsyncMock(return_value="{}")
    rt._plan = AsyncMock(return_value=(SimpleNamespace(), SimpleNamespace()))
    rt._research_fn = AsyncMock(return_value=ResearchResult("redshift", [Source(
        "S1", "Redshift", "https://example.org", text="Redshift stretches light to longer wavelengths.",
    )]))
    return rt


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["Good night", "Goodnight Lora", "okay", "see you tomorrow", "thanks"])
async def test_research_then_casual_reply_does_not_search(reply):
    rt = runtime()
    first = await rt.run_turn("search the web for redshift", lambda _: None, lambda *_: None)
    assert first.meta["verdict"] == "unchecked" and "checked" not in first.meta
    rt._research_fn.reset_mock()
    rt.active_skill = "deep-research"
    second = await rt.run_turn(reply, lambda _: None, lambda *_: None)
    assert second.kind == "chat"
    rt._research_fn.assert_not_called()


@pytest.mark.asyncio
async def test_added_sources_precede_web_and_keep_page_reference():
    db = SimpleNamespace(file_count=1, search=MagicMock(return_value=[{
        "filename": "astronomy.pdf", "page_number": 7, "text": "Redshift stretches light to longer wavelengths.",
    }]))
    rt = runtime(db)
    result = await rt.run_turn("What causes redshift?", lambda _: None, lambda *_: None)
    rt._research_fn.assert_not_called()
    assert result.meta["retrieval"] == "local"
    assert "page 7" in result.sources[0]["title"]
    assert "Redshift stretches" in rt._stream.call_args.args[0][-1]["content"]


@pytest.mark.asyncio
async def test_source_miss_does_not_silently_search_web():
    rt = runtime(SimpleNamespace(file_count=1, search=lambda *_: []))
    result = await rt.run_turn("What are the conclusions?", lambda _: None, lambda *_: None)
    assert "couldn't find a relevant passage" in result.text
    rt._research_fn.assert_not_called()
    rt._stream.assert_not_called()


@pytest.mark.asyncio
async def test_standard_research_uses_local_sources_and_simple_reference_check(monkeypatch):
    rt = runtime(SimpleNamespace(file_count=1, search=lambda *_: [{
        "filename": "astronomy.pdf", "text": "Redshift stretches light to longer wavelengths.",
    }]))
    verify = AsyncMock(side_effect=AssertionError("Heavy verification must not run"))
    monkeypatch.setattr("lora.agent.runtime.verify_answer", verify)
    result = await rt.run_turn("What causes redshift?", lambda _: None, lambda *_: None, mode="standard")
    rt._research_fn.assert_not_called()
    verify.assert_not_called()
    assert result.meta["source_check"]["status"] == "references_checked"
    assert "Your file:" in result.sources[0]["title"]


def test_missing_nli_weights_are_not_reloaded_per_pipeline(monkeypatch):
    from lora.verify import nli

    monkeypatch.setenv("LORA_VERIFY_OFFLINE", "0")
    monkeypatch.setattr(nli, "_MODEL_CACHE", {})
    calls = []
    monkeypatch.setattr(nli.NliGroundednessChecker, "_load_uncached", lambda self: calls.append(self.model_name))
    for _ in range(3):
        nli.NliGroundednessChecker(model_name="missing-model").check_entailment("Light stretches.", "Light stretches.")
    assert calls == ["missing-model"]


@pytest.mark.asyncio
async def test_explicit_web_still_searches_with_added_sources():
    rt = runtime(SimpleNamespace(file_count=1, search=lambda *_: []))
    result = await rt.run_turn("search the web for redshift", lambda _: None, lambda *_: None)
    assert result.kind == "research"
    rt._research_fn.assert_awaited_once()


def test_second_opinion_contains_later_source_ids():
    sources = "\n\n".join(f"[S{i}] Source {i}\n" + "Redshift source evidence. " * 120 for i in range(1, 7))
    prompt = opinion_prompt("redshift", "Light stretches [S4].", sources)
    assert all(f"[S{i}] Source {i}" in prompt for i in range(1, 7))
    assert len(prompt) < 12000


@pytest.mark.parametrize("response", ["not JSON", '{"verdict":"approve"', '{"foo":[]}'])
def test_malformed_second_opinion_cannot_approve(response):
    result = parse_opinion(response)
    assert result.verdict != "approve" and not result.independent


@pytest.mark.asyncio
async def test_refinement_does_not_recheck_unchanged_draft():
    baseline = Verification("abstain", 0, 0, 1, unverified=["An unsupported claim."])
    verifier = AsyncMock(return_value=baseline)
    await refine("Original draft.", "Question?", [], verifier, AsyncMock(return_value="New candidate."), [],
                 max_rounds=1, initial_verification=baseline)
    verifier.assert_awaited_once_with("New candidate.")


@pytest.mark.asyncio
async def test_batch_verification_covers_more_than_two_claims_and_preserves_score():
    class Client:
        calls = 0

        async def stream_chat(self, model, messages, **kwargs):
            self.calls += 1
            assert kwargs["format"] == "json"
            items = json.loads(messages[0]["content"].split("\n", 1)[1])
            yield json.dumps({"results": [{"id": item["id"], "supported": True,
                    "supporting_quote": item["claim"].replace(" [S1]", ""), "reason": "Direct evidence"}
                    for item in items]})

    facts = ["Redshift stretches light.", "Gravity bends light.", "Stars emit radiation.",
             "Planets orbit stars.", "Galaxies contain stars.", "Nebulae contain gas."]
    source = Source("S1", "Astronomy", "https://example.org", text=" ".join(facts))
    client = Client()
    result = await verify_answer(" ".join(f"{fact[:-1]} [S1]." for fact in facts), [source],
                                 VerifierLLM(client, "independent", max_calls=3))
    assert result.total == 6 and result.supported == 6
    assert result.score == 1 and client.calls == 2
    assert result.stages[2]["status"] == "UNAVAILABLE"
    empty = await verify_answer("Unicorns emit radiation [S1].", [source])
    assert empty.supported == 0 and empty.score == 0


@pytest.mark.asyncio
async def test_malformed_batch_is_unavailable_not_completed():
    client = SimpleNamespace(generate=AsyncMock(return_value='{"results": broken}'))
    result = await LlmFactChecker(client).verify_claims([("Light stretches.", "Light stretches.")])
    assert not result[0].completed and not result[0].is_supported


@pytest.mark.asyncio
async def test_page_index_progress_and_embedding_run_off_event_loop(monkeypatch):
    import lora.agent.research as module

    owner = threading.get_ident()
    worker_threads = []

    class Embedder:
        def encode(self, texts):
            worker_threads.append(threading.get_ident())
            return np.ones((len(texts), 4), dtype=np.float32) / 2

    async def search(*_):
        return [Source("S1", "Astronomy", "https://example.org/article")]

    async def fetch(*_, **kwargs):
        return "Redshift stretches light to longer wavelengths. " * 100

    monkeypatch.setattr(module, "web_search", search)
    monkeypatch.setattr(module, "fetch_text", fetch)
    monkeypatch.setattr(module, "_egress_ok", lambda _: True)
    monkeypatch.setattr(module, "_get_rag_components", lambda: (Embedder(), None))
    status = []
    result = await module.research("redshift", status.append)
    assert result.session_index.chunk_count > 0
    assert worker_threads and owner not in worker_threads
    progress = [s for s in status if s.startswith("Embedding sources:")]
    assert len(progress) >= 2 and progress[-1].split()[2].split("/")[0] == progress[-1].split()[2].split("/")[1]
    assert len(result.sources[0].text) <= module.PAGE_CHARS


def test_sleep_animation_renders_grey_before_pausing():
    html = (Path(__file__).resolve().parents[2] / "src/lora/ui/droplet.html").read_text(encoding="utf-8")
    start = html.index("  if(glLost){requestAnimationFrame(frame);return}")
    end = html.index(" })(last);", start)
    frame = html[start:end]
    setup = """
let mode='sleep',dim=0,tm=0,tn=[0,1,1],tgtDim=1,tgtTm=1,tgtTn=[.4,.4,.4];
let glLost=false,last=0,t=0,slowT=0,slowN=0,renderScale=1,ampT=0,freq=0,impulse=0;
let amp=0,phase=0,tw=0,th=0,sp=0,sv=0,sq=0,nextRing=0,rings=[],rip=new Float32Array(8);
let draws=0;const document={hidden:false},cv={width:100,height:100},U={};
const gl=new Proxy({}, {get:(_,name)=>name==='drawArrays'?()=>draws++:()=>{}});
const dark=()=>true,ripple=()=>{},fit=()=>{},requestAnimationFrame=()=>{};
"""
    script = setup + "function frame(now){" + frame + "}\n" + """
for(let i=1;i<=250;i++)frame(i*34);
if(draws===0||dim<.998||tn.some(x=>Math.abs(x-.4)>.002))throw Error('Sleep did not render grey');
const before=draws;frame(10000);if(draws!==before)throw Error('Settled sleep still renders');
mode='idle';frame(10034);if(draws<=before)throw Error('Wake did not render');
"""
    completed = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr


def test_desktop_source_check_does_not_display_truth_percentage():
    html = (Path(__file__).resolve().parents[2] / "src/lora/ui/droplet.html").read_text(encoding="utf-8")
    start = html.index("function renderSourceCheck")
    end = html.index("function renderAgentTrace", start)
    assert "check.note" in html[start:end] and "references_checked" in html[start:end]
    assert "Verification Scorecard" not in html and "Composite Truthfulness" not in html
