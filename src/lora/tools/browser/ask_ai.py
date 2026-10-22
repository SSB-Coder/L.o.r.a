"""External Chatbot Automation Tool for L.O.R.A (ask_other_ai)."""

import asyncio
import time
from typing import Any

from lora.safety.egress import EgressFilter
from lora.safety.gate import DataProvenance
from lora.tools.browser.adapters.chatgpt import ChatGPTAdapter
from lora.tools.browser.adapters.claude import ClaudeAdapter
from lora.tools.browser.adapters.gemini import GeminiAdapter
from lora.tools.browser.adapters.perplexity import PerplexityAdapter
from lora.tools.browser.driver import PlaywrightBrowserDriver
from lora.tools.browser.prompt_agent import PromptOptimizerAgent


class AskOtherAiTool:
    """Safely automates external AI chatbot interfaces under strict typing, provenance, and prompt optimization controls."""

    MIN_ACTION_INTERVAL_SECONDS = 5.0  # ToS politeness rate-limiting

    def __init__(
        self,
        driver: PlaywrightBrowserDriver | None = None,
        egress_filter: EgressFilter | None = None,
        prompt_optimizer: PromptOptimizerAgent | None = None,
    ) -> None:
        self.driver = driver or PlaywrightBrowserDriver()
        self.egress_filter = egress_filter or EgressFilter()
        prompt_optimizer = prompt_optimizer or PromptOptimizerAgent()
        self.prompt_optimizer = prompt_optimizer
        self.adapters = {
            "chatgpt": ChatGPTAdapter(),
            "claude": ClaudeAdapter(),
            "gemini": GeminiAdapter(),
            "perplexity": PerplexityAdapter(),
        }
        self._last_call_time = 0.0
        self._lock = asyncio.Lock()

    async def ask(
        self,
        service: str,
        prompt: str,
        provenance: DataProvenance = DataProvenance.USER,
        optimize_prompt: bool = True,
        style: str = "general",
        deep_optimize: bool = False,
    ) -> dict[str, Any]:
        """Ask question to third-party AI chatbot via browser DOM automation.

        Enforces:
        1. Provenance: Web text (UNTRUSTED) can NEVER be dispatched or typed into external AI.
        2. Prompt Engineering & Privacy: Scrubs local paths/tokens and refines the prompt.
        3. Egress: Destination domain must be on typing allowlist.
        4. Rate limit: Minimum 5-second interval between actions.
        5. Classification: Response is classified as Tier 4 untrusted lead.
        """
        # 1. Provenance Guard
        if provenance not in {DataProvenance.USER, DataProvenance.SYSTEM}:
            return {
                "status": "error",
                "message": (
                    "Security Violation: Arguments derived from untrusted web/file content "
                    "cannot be typed into external AI chat interfaces."
                ),
            }

        # 2. Service resolution
        svc_key = service.strip().lower()
        adapter = self.adapters.get(svc_key)
        if not adapter:
            available = ", ".join(self.adapters.keys())
            return {
                "status": "error",
                "message": f"Unsupported AI service '{service}'. Supported services: {available}",
            }

        # 3. Domain Allowlist Verification
        egress_res = self.egress_filter.check_egress(adapter.url, require_typing_allowlist=True)
        if not egress_res.allowed:
            return {
                "status": "error",
                "message": f"External AI domain typing blocked by policy: {egress_res.reason}",
            }

        # 4. Prompt Optimization and Privacy Scrubbing
        effective_prompt = prompt.strip()
        redactions_count = 0
        redactions: list[str] = []
        applied_style = style

        if optimize_prompt:
            opt_res = await self.prompt_optimizer.optimize(
                prompt=prompt,
                target_ai=svc_key,
                style=style,
                deep=False,
            )
            effective_prompt = opt_res.optimized_prompt
            redactions_count = opt_res.redacted_count
            redactions = opt_res.redactions
            applied_style = opt_res.style.value
        # The deterministic egress scrubber also runs for pre-refined L2 text.
        effective_prompt, final_redactions = self.prompt_optimizer.scrubber.scrub(effective_prompt)
        redactions.extend(final_redactions)
        redactions_count += len(final_redactions)

        # 5. Politeness rate limiting & Playwright automation (serialized with lock)
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            if elapsed < self.MIN_ACTION_INTERVAL_SECONDS:
                await asyncio.sleep(self.MIN_ACTION_INTERVAL_SECONDS - elapsed)
            self._last_call_time = time.monotonic()

            try:
                context = await self.driver._ensure_context()
                try:
                    page = await context.new_page()
                except Exception:
                    # Recover if user closed the browser window or pipe broke between turns
                    if hasattr(self.driver, "_reset_context"):
                        await self.driver._reset_context()
                    else:
                        self.driver._context = None
                        self.driver._playwright = None
                    context = await self.driver._ensure_context()
                    page = await context.new_page()

                try:
                    if hasattr(page, "bring_to_front"):
                        try:
                            await page.bring_to_front()
                        except Exception:
                            pass
                    await page.goto(adapter.url, wait_until="domcontentloaded", timeout=25000)
                    await asyncio.sleep(1.5)
                    reply = await adapter.submit_prompt(page, prompt=effective_prompt, timeout_seconds=75.0)
                finally:
                    # Only close the tab in headless background mode; in visible desktop mode keep ChatGPT open!
                    if getattr(self.driver, "headless", True):
                        try:
                            await page.close()
                        except Exception:
                            pass

                return {
                    "status": "success",
                    "service": adapter.name,
                    "url": adapter.url,
                    "original_prompt": prompt,
                    "prompt": effective_prompt,
                    "optimized": optimize_prompt,
                    "style": applied_style,
                    "redacted_count": redactions_count,
                    "redactions": redactions,
                    "response": reply,
                    "reliability_tier": "T4",
                    "is_untrusted_lead": True,
                    "notice": "External AI-generated lead — unverified against primary sources.",
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to automate {adapter.name}: {e}",
                }
