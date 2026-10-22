"""Prompt Engineering and Privacy Scrubber Agent for External AI Interactions."""

import os
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PromptStyle(StrEnum):
    """Target stylistic and depth constraints for external prompt generation."""

    SIMPLE = "simple"
    RESEARCH = "research"
    CODE = "code"
    CONCISE = "concise"
    FACT_CHECK = "fact_check"
    GENERAL = "general"


class OptimizedPrompt(BaseModel):
    """Immutable result of prompt optimization and privacy scrubbing."""

    model_config = ConfigDict(frozen=True)

    original_prompt: str
    optimized_prompt: str
    target_ai: str
    style: PromptStyle
    redacted_count: int
    redactions: list[str] = Field(default_factory=list)


def is_simple_prompt(prompt: str) -> bool:
    """True when the user's question is short and straightforward and should NOT be wrapped in a heavy multi-section rubric."""
    clean = (prompt or "").strip()
    words = clean.split()
    if len(words) > 14 or "\n" in clean or "```" in clean:
        return False
    complex_markers = (
        "research", "literature", "paper", "architecture", "mechanism", "state of the art",
        "in depth", "in-depth", "deep", "thorough", "detailed", "comprehensive",
        "write code", "implement", "script", "function", "class", "bug", "traceback",
        "verify", "is it true", "fact check", "controversy", "debunk",
        "compare", "contrast", "step by step", "step-by-step", "analyze", "analysis",
    )
    low = clean.lower()
    return not any(w in low for w in complex_markers)


class PrivacyScrubber:
    """Detects and redacts local file paths, usernames, private IPs, and API tokens."""

    PATH_PATTERNS = [
        re.compile(r"\b[a-zA-Z]:[\\/](?:[a-zA-Z0-9_\-.]+[\\/])*[a-zA-Z0-9_\-.]+", re.IGNORECASE),
        re.compile(r"\\\\[a-zA-Z0-9_.\-]+(?:\\[a-zA-Z0-9_.\-]+)+"),
    ]

    PRIVATE_IP_PATTERN = re.compile(
        r"\b(?:127\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"192\.168\.\d{1,3}\.\d{1,3}|"
        r"172\.(?:1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3})\b"
    )

    SECRET_PATTERNS = [
        (re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b"), "[SECRET_KEY]"),
        (re.compile(r"\bghp_[a-zA-Z0-9]{20,}\b"), "[GITHUB_TOKEN]"),
        (re.compile(r"\bBearer\s+[a-zA-Z0-9_\-.]{16,}\b", re.IGNORECASE), "Bearer [REDACTED_TOKEN]"),
        (re.compile(r"(?:password|passwd|pwd|secret)\s*[:=]\s*['\"]?([^\s'\"]{4,})['\"]?", re.IGNORECASE), "[REDACTED_SECRET]"),
    ]

    def __init__(self, username: str | None = None) -> None:
        self.username = username or os.environ.get("USERNAME") or os.environ.get("USER") or ""

    def scrub(self, text: str) -> tuple[str, list[str]]:
        """Scrub private paths, user identifiers, private IPs, and secrets from text."""
        redactions: list[str] = []
        cleaned = text

        for pattern, replacement in self.SECRET_PATTERNS:
            matches = pattern.findall(cleaned)
            if matches:
                for _ in matches:
                    redactions.append(f"Secret: {pattern.pattern[:15]}...")
                cleaned = pattern.sub(replacement, cleaned)

        for p_regex in self.PATH_PATTERNS:
            matches = p_regex.findall(cleaned)
            if matches:
                for match in matches:
                    redactions.append(f"Local Path: {match}")
                cleaned = p_regex.sub("[LOCAL_PATH]", cleaned)

        if self.username and len(self.username) > 2:
            user_pattern = re.compile(rf"\b{re.escape(self.username)}\b", re.IGNORECASE)
            matches = user_pattern.findall(cleaned)
            if matches:
                redactions.append(f"Username: {self.username}")
                cleaned = user_pattern.sub("[USER]", cleaned)

        ip_matches = self.PRIVATE_IP_PATTERN.findall(cleaned)
        if ip_matches:
            for ip_m in ip_matches:
                redactions.append(f"Private IP: {ip_m}")
            cleaned = self.PRIVATE_IP_PATTERN.sub("[PRIVATE_IP]", cleaned)

        return cleaned, redactions


class PromptOptimizerAgent:
    """Engineers high-leverage prompts for external AI chatbots with privacy defenses."""

    def __init__(
        self,
        scrubber: PrivacyScrubber | None = None,
        ollama_client: Any | None = None,
    ) -> None:
        self.scrubber = scrubber or PrivacyScrubber()
        self.ollama_client = ollama_client

    def _detect_style(self, prompt: str, allow_simple: bool = False) -> PromptStyle:
        """Infer suitable prompt style if set to general or auto."""
        lower = prompt.lower()
        if any(w in lower for w in ["write code", "implement", "script", "function", "class", "bug", "traceback"]):
            return PromptStyle.CODE
        if any(
            w in lower
            for w in [
                "research", "literature", "paper", "architecture", "mechanism", "state of the art",
                "in depth", "in-depth", "deep", "thorough", "detailed", "comprehensive",
            ]
        ):
            return PromptStyle.RESEARCH
        if any(w in lower for w in ["verify", "is it true", "fact check", "controversy", "claim", "debunk"]):
            return PromptStyle.FACT_CHECK
        if any(w in lower for w in ["briefly", "short", "tl;dr", "tldr", "summarize in 3 points", "quick"]):
            return PromptStyle.CONCISE
        if allow_simple and is_simple_prompt(prompt):
            return PromptStyle.SIMPLE
        return PromptStyle.GENERAL

    def _build_deterministic_prompt(
        self,
        clean_text: str,
        target_ai: str,
        style: PromptStyle,
    ) -> str:
        """Construct a structured, high-leverage prompt with zero conversational filler."""
        target = target_ai.strip().lower()

        if style == PromptStyle.SIMPLE:
            return clean_text

        no_fluff = "Do not include conversational preamble, pleasantries, or polite introductions. Go directly to the answer."

        if style == PromptStyle.CODE:
            return (
                f"You are an expert software engineer. Provide a clean, robust, and idiomatic implementation for:\n"
                f"{clean_text}\n\n"
                f"Requirements:\n"
                f"- Include full type annotations and handle edge cases.\n"
                f"- Provide clear comments for critical architectural choices.\n"
                f"- {no_fluff}"
            )

        if style == PromptStyle.RESEARCH:
            return (
                f"You are a principal research scientist. Provide a rigorous, systematic, and well-structured analysis of:\n"
                f"{clean_text}\n\n"
                f"Structure your response with:\n"
                f"1. Executive Summary & Core Mechanisms\n"
                f"2. Theoretical Foundations & Key Formulations\n"
                f"3. Practical Applications & SOTA Benchmarks\n"
                f"4. Key Limitations & Open Problems\n"
                f"- {no_fluff}"
            )

        if style == PromptStyle.FACT_CHECK:
            return (
                f"Assess the following topic/claim with strict empirical objectivity:\n"
                f"{clean_text}\n\n"
                f"Requirements:\n"
                f"- Identify what is established consensus vs. what is actively contested.\n"
                f"- Highlight primary empirical evidence and counter-arguments.\n"
                f"- State your confidence level and primary caveats.\n"
                f"- {no_fluff}"
            )

        if style == PromptStyle.CONCISE:
            return (
                f"Provide a direct, high-density summary for:\n"
                f"{clean_text}\n\n"
                f"Format strictly as bullet points highlighting key facts. {no_fluff}"
            )

        if target == "claude":
            return (
                f"Provide an authoritative, detailed, and logically structured analysis of:\n"
                f"{clean_text}\n\n"
                f"Use clear Markdown headings, maintain analytical nuance, and {no_fluff.lower()}"
            )
        elif target == "chatgpt":
            return (
                f"Provide a direct, authoritative, and practical explanation of:\n"
                f"{clean_text}\n\n"
                f"Use clear section headers, concise bullet points, and {no_fluff.lower()}"
            )
        elif target == "gemini":
            return (
                f"Provide a clear, multi-dimensional overview of:\n"
                f"{clean_text}\n\n"
                f"Synthesize core principles, practical examples, and notable modern developments. {no_fluff}"
            )
        elif target == "perplexity":
            return (
                f"Synthesize current, verifiable information regarding:\n"
                f"{clean_text}\n\n"
                f"Focus on recent empirical findings, verified sources, and concrete data points. {no_fluff}"
            )
        else:
            return (
                f"Provide a comprehensive and structured answer to:\n"
                f"{clean_text}\n\n"
                f"{no_fluff}"
            )

    async def optimize(
        self,
        prompt: str,
        target_ai: str = "claude",
        style: PromptStyle | str = PromptStyle.GENERAL,
        deep: bool = False,
    ) -> OptimizedPrompt:
        """Sanitize privacy identifiers and engineer an optimized prompt."""
        allow_simple = isinstance(style, str) and style.lower() == "auto"
        if isinstance(style, str):
            if style.lower() == "auto":
                style_enum = self._detect_style(prompt, allow_simple=True)
            else:
                try:
                    style_enum = PromptStyle(style.lower())
                except ValueError:
                    style_enum = PromptStyle.GENERAL
        else:
            style_enum = style

        if style_enum == PromptStyle.GENERAL and not allow_simple:
            style_enum = self._detect_style(prompt, allow_simple=False)

        scrubbed_text, redactions = self.scrubber.scrub(prompt.strip())

        # Simple questions bypass heavy LLM prompt rewriting so they stay natural and fast
        if style_enum == PromptStyle.SIMPLE and not deep:
            return OptimizedPrompt(
                original_prompt=prompt,
                optimized_prompt=scrubbed_text,
                target_ai=target_ai,
                style=style_enum,
                redacted_count=len(redactions),
                redactions=redactions,
            )

        if deep:
            raise ValueError("Deep prompt refinement must run in L2 before this tool")

        optimized = self._build_deterministic_prompt(
            clean_text=scrubbed_text,
            target_ai=target_ai,
            style=style_enum,
        )

        return OptimizedPrompt(
            original_prompt=prompt,
            optimized_prompt=optimized,
            target_ai=target_ai,
            style=style_enum,
            redacted_count=len(redactions),
            redactions=redactions,
        )
