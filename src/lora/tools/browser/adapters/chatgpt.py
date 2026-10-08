"""ChatGPT DOM Adapter for L.O.R.A."""

import asyncio
from typing import Any


class ChatGPTAdapter:
    """DOM Adapter for ChatGPT (chatgpt.com) using Playwright role and CSS locators."""

    name = "chatgpt"
    domain = "chatgpt.com"
    url = "https://chatgpt.com"

    # Known locators
    INPUT_SELECTORS = [
        '#prompt-textarea',
        'div[contenteditable="true"]',
        'textarea[placeholder*="Ask"]',
        'textarea',
    ]

    SEND_BUTTON_SELECTORS = [
        'button[data-testid="send-button"]',
        'button[aria-label="Send prompt"]',
        'button[aria-label="Send"]',
    ]

    RESPONSE_SELECTORS = [
        'div[data-message-author-role="assistant"]',
        '.agent-turn',
        'div.markdown',
    ]

    async def submit_prompt(self, page: Any, prompt: str, timeout_seconds: float = 30.0) -> str:
        """Find input, type prompt, press Enter/submit, and wait for response."""
        # 1. Locate textarea
        target_input = None
        for sel in self.INPUT_SELECTORS:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                target_input = loc
                break

        if not target_input:
            raise RuntimeError("Could not find ChatGPT input textarea on page.")

        # 2. Fill and submit
        await target_input.click()
        await target_input.fill(prompt)
        await asyncio.sleep(0.5)

        # Try clicking send button or press Enter
        submitted = False
        for btn_sel in self.SEND_BUTTON_SELECTORS:
            btn = page.locator(btn_sel).first
            if await btn.count() > 0 and await btn.is_enabled():
                await btn.click()
                submitted = True
                break

        if not submitted:
            await target_input.press("Enter")

        # 3. Wait for assistant message to appear and settle
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
                            # Text has stabilized
                            return curr_text.strip()
                        last_text = curr_text
            await asyncio.sleep(1.5)

        if last_text:
            return last_text.strip()
        return "ChatGPT response timed out or could not be captured from DOM."
