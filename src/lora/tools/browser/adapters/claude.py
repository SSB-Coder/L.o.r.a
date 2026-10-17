"""Claude.ai DOM Adapter for L.O.R.A."""

import asyncio
from typing import Any

from lora.tools.browser.adapters.base import ChatbotAdapter


class ClaudeAdapter(ChatbotAdapter):
    """DOM Adapter for Claude (claude.ai) using Playwright role and CSS locators."""

    name = "claude"
    domain = "claude.ai"
    url = "https://claude.ai/new"

    INPUT_SELECTORS = [
        'div[contenteditable="true"]',
        'div.ProseMirror',
        'fieldset textarea',
        'textarea',
    ]

    SEND_BUTTON_SELECTORS = [
        'button[aria-label="Send Message"]',
        'button[aria-label="Send message"]',
        'button:has-text("Send")',
    ]

    RESPONSE_SELECTORS = [
        '[data-is-streaming="false"]',
        'div.font-claude-message',
        'div[data-test-render-count]',
        'div.standard-markdown',
    ]

    async def submit_prompt(self, page: Any, prompt: str, timeout_seconds: float = 30.0) -> str:
        """Find input, type prompt, press Enter/submit, and wait for Claude's response."""
        target_input = None
        for sel in self.INPUT_SELECTORS:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                target_input = loc
                break

        if not target_input:
            raise RuntimeError("Could not find Claude input box on page.")

        await target_input.click()
        try:
            await target_input.fill(prompt)
        except Exception:
            if hasattr(page, "keyboard"):
                await page.keyboard.insert_text(prompt)
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

        # Wait for streaming to finish and text to stabilize
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
        return "Claude response timed out or could not be captured from DOM."
