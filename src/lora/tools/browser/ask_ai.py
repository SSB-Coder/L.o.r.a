"""External Chatbot Automation Tool for L.O.R.A (ask_other_ai)."""

import asyncio
import time
from typing import Any

from lora.safety.egress import EgressFilter
from lora.safety.gate import DataProvenance
from lora.tools.browser.adapters.chatgpt import ChatGPTAdapter
from lora.tools.browser.adapters.claude import ClaudeAdapter
from lora.tools.browser.driver import PlaywrightBrowserDriver


class AskOtherAiTool:
    """Safely automates external AI chatbot interfaces under strict typing and provenance controls."""

    MIN_ACTION_INTERVAL_SECONDS = 5.0  # ToS politeness rate-limiting

    def __init__(
        self,
        driver: PlaywrightBrowserDriver | None = None,
        egress_filter: EgressFilter | None = None,
    ) -> None:
        self.driver = driver or PlaywrightBrowserDriver()
        self.egress_filter = egress_filter or EgressFilter()
        self.adapters = {
            "chatgpt": ChatGPTAdapter(),
            "claude": ClaudeAdapter(),
        }
        self._last_call_time = 0.0
        self._lock = asyncio.Lock()

    async def ask(
        self,
        service: str,
        prompt: str,
        provenance: DataProvenance = DataProvenance.USER,
    ) -> dict[str, Any]:
        """Ask question to third-party AI chatbot via browser DOM automation.

        Enforces:
        1. Provenance: Web text (UNTRUSTED) can NEVER be dispatched or typed into external AI.
        2. Egress: Destination domain must be on typing allowlist.
        3. Rate limit: Minimum 5-second interval between actions.
        4. Classification: Response is classified as Tier 4 untrusted lead.
        """
        # 1. Provenance Guard
        if provenance == DataProvenance.UNTRUSTED:
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

        # 4. Politeness rate limiting & Playwright automation (serialized with lock)
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            if elapsed < self.MIN_ACTION_INTERVAL_SECONDS:
                await asyncio.sleep(self.MIN_ACTION_INTERVAL_SECONDS - elapsed)
            self._last_call_time = time.monotonic()

            try:
                context = await self.driver._ensure_context()
                page = await context.new_page()
                try:
                    await page.goto(adapter.url, wait_until="domcontentloaded", timeout=20000)
                    await asyncio.sleep(1.0)
                    reply = await adapter.submit_prompt(page, prompt=prompt.strip(), timeout_seconds=30.0)
                finally:
                    await page.close()

                return {
                    "status": "success",
                    "service": adapter.name,
                    "url": adapter.url,
                    "prompt": prompt,
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
