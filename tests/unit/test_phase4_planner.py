"""Unit tests for Phase 4 planning and fast-path routing."""

from unittest.mock import AsyncMock, patch

import pytest

from lora.core.controller import Controller
from lora.core.planner import ControlPlanner


@pytest.fixture
def planner():
    return ControlPlanner()


def test_fast_path_ask_ai(planner: ControlPlanner):
    plan_claude = planner.plan("ask claude what is quantum entanglement")
    assert len(plan_claude) == 1
    assert plan_claude[0].tool == "ask_other_ai"
    assert plan_claude[0].params["service"] == "claude"
    assert "quantum entanglement" in plan_claude[0].params["prompt"]

    plan_gemini = planner.plan("ask gemini to summarize general relativity")
    assert len(plan_gemini) == 1
    assert plan_gemini[0].tool == "ask_other_ai"
    assert plan_gemini[0].params["service"] == "gemini"
    assert "summarize general relativity" in plan_gemini[0].params["prompt"]

    plan_perp = planner.plan("ask perplexity for latest quantum benchmarks")
    assert len(plan_perp) == 1
    assert plan_perp[0].tool == "ask_other_ai"
    assert plan_perp[0].params["service"] == "perplexity"
    assert "latest quantum benchmarks" in plan_perp[0].params["prompt"]


def test_fast_path_search_specific_source(planner: ControlPlanner):
    plan = planner.plan("search arxiv for superconducting qubits")
    assert len(plan) == 1
    assert plan[0].tool == "web_search"
    assert plan[0].params["source"] == "arxiv"
    assert plan[0].params["query"] == "superconducting qubits"


def test_fast_path_search_wikipedia(planner: ControlPlanner):
    plan = planner.plan("search wikipedia for general relativity")
    assert len(plan) == 1
    assert plan[0].tool == "web_search"
    assert plan[0].params["source"] == "wikipedia"
    assert plan[0].params["query"] == "general relativity"


def test_fast_path_search_general_web(planner: ControlPlanner):
    plan = planner.plan("search web for latest AI benchmarks")
    assert len(plan) == 1
    assert plan[0].tool == "web_search"
    assert plan[0].params["source"] == "auto"
    assert plan[0].params["query"] == "latest AI benchmarks"


def test_fast_path_fetch_webpage(planner: ControlPlanner):
    plan = planner.plan("fetch webpage https://en.wikipedia.org/wiki/Physics")
    assert len(plan) == 1
    assert plan[0].tool == "fetch_page"
    assert plan[0].params["url"] == "https://en.wikipedia.org/wiki/Physics"


def test_fast_path_fetch_pdf(planner: ControlPlanner):
    plan = planner.plan("fetch pdf https://arxiv.org/pdf/2401.00001.pdf")
    assert len(plan) == 1
    assert plan[0].tool == "pdf_fetch"
    assert plan[0].params["url"] == "https://arxiv.org/pdf/2401.00001.pdf"


@pytest.mark.asyncio
async def test_controller_async_execution_web_search():
    controller = Controller()
    mock_search_res = {
        "status": "success",
        "query": "quantum computing",
        "count": 1,
        "hits": [{"title": "Quantum computing", "url": "https://en.wikipedia.org/wiki/Quantum_computing"}],
    }

    with patch.object(controller.tool_registry, "aexecute", new_callable=AsyncMock) as mock_aexec:
        mock_aexec.return_value = mock_search_res
        res = await controller.aexecute_user_request("search wikipedia for quantum computing")
        assert res["status"] == "success"
        assert len(res["steps"]) == 1
        assert res["steps"][0]["tool"] == "web_search"
        assert res["steps"][0]["result"]["count"] == 1
