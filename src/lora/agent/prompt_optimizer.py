"""L2 prompt refinement, never invoked by an L4 tool."""
from typing import Any

from lora.tools.browser.prompt_agent import (
    OptimizedPrompt,
    PrivacyScrubber,
    PromptStyle,
    is_simple_prompt,
)


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

        if deep and self.ollama_client is not None:
            try:
                from lora.dialogue import strip_think

                meta_system = (
                    f"You are an expert prompt engineer writing a clear prompt to send to {target_ai}. "
                    "Match the complexity of the user's question: if the question is simple, keep the prompt short and direct; "
                    "if it is complex, structure it clearly without adding unwanted bloat. "
                    "Do NOT answer the question yourself. Output ONLY the prompt text."
                )
                raw_refined = await self.ollama_client.generate(
                    prompt=f"User request to turn into a prompt ({style_enum.value} mode):\n{scrubbed_text}",
                    system=meta_system,
                    temperature=0.2,
                )
                refined = strip_think(str(raw_refined or "")).strip().strip('"')
                if refined and len(refined) > 5:
                    return OptimizedPrompt(
                        original_prompt=prompt,
                        optimized_prompt=refined,
                        target_ai=target_ai,
                        style=style_enum,
                        redacted_count=len(redactions),
                        redactions=redactions,
                    )
            except Exception:
                pass

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
