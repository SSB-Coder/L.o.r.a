"""Perplexity DOM Adapter for L.O.R.A."""

import asyncio
from typing import Any


class PerplexityAdapter:
    """DOM Adapter for Perplexity AI (perplexity.ai) using Playwright role and CSS locators."""

    name = "perplexity"
    domain = "perplexity.ai"
    url = "https://www.perplexity.ai"

    INPUT_SELECTORS = [
        'textarea[placeholder*="Ask"]',
        'textarea[placeholder*="Search"]',
        'div[contenteditable="true"]',
        'textarea',
    ]

    SEND_BUTTON_SELECTORS = [
        'button[aria-label*="Submit"]',
        'button[aria-label*="Ask"]',
        'button[type="submit"]',
        'button:has(svg)',
    ]

    RESPONSE_SELECTORS = [
        'div.prose',
        'div[dir="auto"]',
        '.markdown',
        'div.default.font-sans.text-base',
    ]

    async def submit_prompt(self, page: Any, prompt: str, timeout_seconds: float = 30.0) -> str:
        """Find input, type prompt, press Enter/submit, and wait for Perplexity's response."""
        # 1. Locate textarea
        target_input = None
        for sel in self.INPUT_SELECTORS:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                target_input = loc
                break

        if not target_input:
            raise RuntimeError("Could not find Perplexity input box on page.")

        # 2. Fill and submit
        await target_input.click()
        await target_input.fill(prompt)
        await asyncio.sleep(0.5)

        submitted = False
        for btn_sel in self.SEND_BUTTON_SELECTORS:
            btn = page.locator(btn_sel).first
            if await btn.count() > 0 and await btn.is_enabled():
                await btn.click()
                submitted = True
                break

        if not submitted:
            await target_input.press("Enter")

        # 3. Wait for response to stabilize
        await asyncio.sleep(2.0)
        start_time = asyncio.get_event_loop().time()
        last_text = ""

        while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
            for resp_sel in self.RESPONSE_SELECTORS:
                resps = page.locator(resp_sel)
                count = await resps.count()
                if count > 0:
                    last_msg = resps.nth(count - 1)
                    curr_text = await last_msg.inner_text()
                    if curr_text and len(curr_text.strip()) > 0:
                        if curr_text == last_text and len(curr_text) > 10:
                            return curr_text.strip()
                        last_text = curr_text
            await asyncio.sleep(1.5)

        if last_text:
            return last_text.strip()
        return "Perplexity response timed out or could not be captured from DOM."
