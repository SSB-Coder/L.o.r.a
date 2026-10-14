"""Gemini DOM Adapter for L.O.R.A."""

import asyncio
from typing import Any


class GeminiAdapter:
    """DOM Adapter for Google Gemini (gemini.google.com) using Playwright role and CSS locators."""

    name = "gemini"
    domain = "gemini.google.com"
    url = "https://gemini.google.com/app"

    INPUT_SELECTORS = [
        'div[role="textbox"]',
        'rich-textarea div[contenteditable="true"]',
        'textarea[aria-label*="prompt"]',
        'div[contenteditable="true"]',
    ]

    SEND_BUTTON_SELECTORS = [
        'button[aria-label*="Send"]',
        'button.send-button',
        'button[data-test-id="send-button"]',
        'button[aria-label*="Submit"]',
    ]

    RESPONSE_SELECTORS = [
        'message-content',
        '.model-response-text',
        'div[data-test-id="model-response"]',
        'div.response-container-content',
        'div.markdown',
    ]

    async def submit_prompt(self, page: Any, prompt: str, timeout_seconds: float = 30.0) -> str:
        """Find input, type prompt, press Enter/submit, and wait for Gemini's response."""
        # 1. Locate textbox
        target_input = None
        for sel in self.INPUT_SELECTORS:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                target_input = loc
                break

        if not target_input:
            raise RuntimeError("Could not find Gemini input textbox on page.")

        # 2. Fill and submit
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
        return "Gemini response timed out or could not be captured from DOM."
