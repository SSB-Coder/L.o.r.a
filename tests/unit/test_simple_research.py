"""User workflow checks for citation-only research and complete document summaries."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lora.agent.jobs import JobAborted
from lora.agent.research import ResearchResult, Source
from lora.agent.runtime import AgentRuntime
from lora.agent.source_check import check_references, mobile_answer
from lora.dialogue import ConversationHistory
from lora.grounding.media_vector_db import MediaVectorDB


def runtime(media=None, mobile=False):
    rt = AgentRuntime("http://127.0.0.1:11434", None, ConversationHistory(), MagicMock(),
                      media_db=media, allow_actions=not mobile)
    rt._plan = AsyncMock(return_value=(SimpleNamespace(), SimpleNamespace()))
    rt._collect = AsyncMock(return_value="Section notes [S1].")
    rt._stream = AsyncMock(return_value="The author's conclusion is that conservation matters [S1].")
    rt._research_fn = AsyncMock(return_value=ResearchResult("redshift", [Source(
        "S1", "Astronomy", "https://example.org", text="Redshift stretches light.",
    )]))
    return rt


def test_reference_integrity_is_not_a_truth_score():
    sources = [Source("S1", "Astronomy", "", text="Redshift stretches light.")]
    good = check_references("An unsupported assertion [S1].", sources)
    assert good["status"] == "references_checked"  # Reference exists; fact support is unknown.
    assert "not independently" in good["note"] and "score" not in good
    unknown = check_references("Text [S1, S9].", sources)
    assert unknown["status"] == "invalid_references" and unknown["unknown_ids"] == ["S9"]
    assert check_references("Uncited answer.", sources)["status"] == "missing_citations"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["quick", "standard", "deep", "deep_with_refinement"])
async def test_every_research_mode_stays_on_writer_without_verification(monkeypatch, mode):
    rt = runtime()
    expensive = AsyncMock(side_effect=AssertionError("Heavy verification ran"))
    monkeypatch.setattr("lora.agent.runtime.verify_answer", expensive)
    rt.run_rsi_outer_loop = expensive
    result = await rt.run_turn("search the web for redshift", lambda _: None, lambda *_: None, mode=mode)
    assert result.kind == "research" and result.text == rt._stream.return_value
    assert result.meta["source_check"]["status"] == "references_checked"
    assert "checked" not in result.meta and "verification_loop" not in result.meta
    assert rt._plan.await_count == 1 and rt._stream.await_count == 1
    assert all(call.kwargs.get("role", "writer") in ("planner", "writer") for call in rt._collect.await_args_list)
    expensive.assert_not_called()


@pytest.mark.asyncio
async def test_phone_chat_after_research_has_no_research_badges():
    rt = runtime(mobile=True)
    research = await rt.run_job("search the web for redshift", "quick", lambda: None, depth="expert")
    assert research["kind"] == "research"
    prompt = rt._stream.call_args.args[0][0]["content"]
    assert "Requested answer style: expert" in prompt and "1 to 4 sentences" not in prompt
    rt._research_fn.reset_mock()
    for text in ("how are you", "I feel tired", "good night", "let's chat"):
        reply = await rt.run_job(text, "quick", lambda: None)
        assert reply["kind"] in ("chat", "smalltalk")
        event = mobile_answer(reply)
        assert not event["source_check"] and not event["citations"]
        assert "confidence" not in event and "unverified" not in event
    rt._research_fn.assert_not_called()


@pytest.mark.asyncio
async def test_web_research_does_not_append_unrelated_local_files():
    local = SimpleNamespace(file_count=1, search=MagicMock(return_value=[{
        "filename": "essay.docx", "text": "An unrelated opinion about pollution.",
    }]))
    rt = runtime(local)
    result = await rt.run_turn("search the web for redshift", lambda _: None, lambda *_: None)
    assert [source["title"] for source in result.sources] == ["Astronomy"]
    local.search.assert_not_called()


def source_db(tmp_path):
    embedder = SimpleNamespace(encode=lambda texts: np.ones((len(texts), 4), dtype=np.float32) / 2,
                              model_path="test-only", _use_fallback=False)
    db = MediaVectorDB(embedder, tmp_path / "index")
    text = "# Opening\n\n" + "The opening introduces the argument. " * 800
    text += "\n\n# Conclusion\n\nFINAL_FINDING: Community conservation protects the habitat."
    path = tmp_path / "essay.md"
    path.write_text(text, encoding="utf-8")
    assert db.add_file(path)["status"] == "success"
    return db


@pytest.mark.asyncio
async def test_document_summary_reads_conclusion_and_never_searches(tmp_path):
    db = source_db(tmp_path)
    rt = runtime(db)
    result = await rt.run_turn("Summarize my sources", lambda _: None, lambda *_: None)
    inputs = [call.args[0][-1]["content"] for call in rt._collect.await_args_list]
    inputs.extend(call.args[0][-1]["content"] for call in rt._stream.await_args_list)
    assert any("FINAL_FINDING" in text for text in inputs)
    rt._research_fn.assert_not_called()
    assert result.text == rt._stream.return_value
    assert result.meta["summary_coverage"] == {
        "included_chunks": db.chunk_count, "total_chunks": db.chunk_count, "partial": False,
    }
    assert result.sources[0]["title"] == "essay.md"
    assert "sample passage" not in result.text and "extracted characters" not in result.text


def test_summary_snapshot_reports_partial_coverage(tmp_path):
    db = source_db(tmp_path)
    snapshot = db.summary_batches(max_chars=1500, max_batches=1)
    assert len(snapshot["batches"]) == 1 and snapshot["partial"]
    assert 0 < snapshot["included_chunks"] < snapshot["total_chunks"]


@pytest.mark.asyncio
async def test_summary_cancellation_stops_before_next_writer_call(tmp_path):
    rt = runtime(source_db(tmp_path))
    rt._should_abort = lambda: "cancelled"
    with pytest.raises(JobAborted):
        await rt._summarize_sources("Summarize my sources", lambda _: None, lambda *_: None)
    rt._collect.assert_not_called()
    rt._stream.assert_not_called()
