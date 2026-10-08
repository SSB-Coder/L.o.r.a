"""Unit tests for PromptOptimizerAgent and PrivacyScrubber."""

from unittest.mock import AsyncMock

import pytest

from lora.tools.browser.prompt_agent import (
    OptimizedPrompt,
    PrivacyScrubber,
    PromptOptimizerAgent,
    PromptStyle,
)


def test_privacy_scrubber_paths():
    scrubber = PrivacyScrubber(username="alice")
    raw = "Check the file at C:\\Users\\alice\\Documents\\secret.py and D:/repos/data.csv"
    cleaned, redactions = scrubber.scrub(raw)

    assert "C:\\Users\\alice\\Documents\\secret.py" not in cleaned
    assert "D:/repos/data.csv" not in cleaned
    assert "[LOCAL_PATH]" in cleaned
    assert len(redactions) >= 2


def test_privacy_scrubber_ips_and_secrets():
    scrubber = PrivacyScrubber()
    raw = (
        "Server at 192.168.1.45 and loopback 127.0.0.1 using key "
        "sk-1234567890abcdef12345678 and token ghp_1234567890abcdef12345678"
    )
    cleaned, redactions = scrubber.scrub(raw)

    assert "192.168.1.45" not in cleaned
    assert "127.0.0.1" not in cleaned
    assert "sk-1234567890abcdef12345678" not in cleaned
    assert "ghp_1234567890abcdef12345678" not in cleaned
    assert "[PRIVATE_IP]" in cleaned
    assert "[SECRET_KEY]" in cleaned
    assert "[GITHUB_TOKEN]" in cleaned
    assert len(redactions) == 4


def test_privacy_scrubber_username():
    scrubber = PrivacyScrubber(username="johndoe")
    raw = "Hey johndoe, please explain quantum teleportation."
    cleaned, redactions = scrubber.scrub(raw)

    assert "johndoe" not in cleaned
    assert "[USER]" in cleaned
    assert any("Username" in r for r in redactions)


@pytest.mark.asyncio
async def test_prompt_optimizer_deterministic_styles():
    optimizer = PromptOptimizerAgent(scrubber=PrivacyScrubber(username="tester"))

    # 1. Code style detection and injection
    code_res = await optimizer.optimize("Write a Python script for binary search")
    assert code_res.style == PromptStyle.CODE
    assert "expert software engineer" in code_res.optimized_prompt.lower()
    assert "conversational preamble" in code_res.optimized_prompt.lower()

    # 2. Research style detection and injection
    research_res = await optimizer.optimize("Research the literature on quantum error correction architectures")
    assert research_res.style == PromptStyle.RESEARCH
    assert "principal research scientist" in research_res.optimized_prompt.lower()
    assert "Executive Summary & Core Mechanisms" in research_res.optimized_prompt

    # 3. Fact check style
    fact_res = await optimizer.optimize("Is it true that room temperature superconductors were verified?")
    assert fact_res.style == PromptStyle.FACT_CHECK
    assert "strict empirical objectivity" in fact_res.optimized_prompt.lower()


@pytest.mark.asyncio
async def test_prompt_optimizer_model_specific_directives():
    optimizer = PromptOptimizerAgent()

    # Claude target
    res_claude = await optimizer.optimize("What is quantum computing?", target_ai="claude", style=PromptStyle.GENERAL)
    assert "Markdown headings" in res_claude.optimized_prompt

    # ChatGPT target
    res_chatgpt = await optimizer.optimize("What is quantum computing?", target_ai="chatgpt", style=PromptStyle.GENERAL)
    assert "bullet points" in res_chatgpt.optimized_prompt

    # Gemini target
    res_gemini = await optimizer.optimize("What is quantum computing?", target_ai="gemini", style=PromptStyle.GENERAL)
    assert "multi-dimensional overview" in res_gemini.optimized_prompt

    # Perplexity target
    res_perp = await optimizer.optimize("What is quantum computing?", target_ai="perplexity", style=PromptStyle.GENERAL)
    assert "verifiable information" in res_perp.optimized_prompt


@pytest.mark.asyncio
async def test_prompt_optimizer_deep_llm_with_fallback():
    mock_ollama = AsyncMock()
    mock_ollama.generate.return_value = "Optimized deep prompt with advanced constraints."

    optimizer = PromptOptimizerAgent(ollama_client=mock_ollama)

    # When deep is True and Ollama succeeds
    res_deep = await optimizer.optimize("Explain backpropagation", deep=True)
    assert res_deep.optimized_prompt == "Optimized deep prompt with advanced constraints."
    mock_ollama.generate.assert_called_once()

    # When Ollama fails, fall back to deterministic prompt gracefully
    mock_ollama.generate.side_effect = RuntimeError("Ollama connection failed")
    res_fallback = await optimizer.optimize("Explain backpropagation", deep=True)
    assert "Explain backpropagation" in res_fallback.optimized_prompt
    assert "conversational preamble" in res_fallback.optimized_prompt.lower()


def test_optimized_prompt_immutability():
    res = OptimizedPrompt(
        original_prompt="hello",
        optimized_prompt="Hello world",
        target_ai="claude",
        style=PromptStyle.GENERAL,
        redacted_count=0,
    )
    with pytest.raises(Exception):
        res.optimized_prompt = "modified"  # type: ignore


@pytest.mark.asyncio
async def test_prompt_optimizer_edge_cases():
    optimizer = PromptOptimizerAgent()

    # Concise detection
    concise_res = await optimizer.optimize("Summarize in 3 points briefly the main ideas")
    assert concise_res.style == PromptStyle.CONCISE
    assert "bullet points" in concise_res.optimized_prompt.lower()

    # Invalid string style falls back to general
    invalid_style_res = await optimizer.optimize("Explain entropy", style="nonexistent_style")
    assert invalid_style_res.style == PromptStyle.GENERAL

    # Unknown bot target falls back to general template
    unknown_bot_res = await optimizer.optimize("Explain entropy", target_ai="unknown_bot")
    assert "comprehensive and structured answer" in unknown_bot_res.optimized_prompt.lower()

