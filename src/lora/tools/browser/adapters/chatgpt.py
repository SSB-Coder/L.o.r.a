"""ChatGPT DOM Adapter for L.O.R.A."""

import asyncio
from typing import Any

from lora.tools.browser.adapters.base import ChatbotAdapter


class ChatGPTAdapter(ChatbotAdapter):
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
        'div[data-message-author-role="assistant"] .markdown',
        'div[data-message-author-role="assistant"]',
        'article[data-testid^="conversation-turn-"] .markdown',
        '.agent-turn',
        'div.markdown',
    ]

    STOP_STREAMING_SELECTORS = [
        'button[data-testid="stop-button"]',
        'button[aria-label*="Stop"]',
    ]

    async def submit_prompt(self, page: Any, prompt: str, timeout_seconds: float = 75.0) -> str:
        """Find input, type prompt, press Enter/submit, and wait for response."""
        if hasattr(page, "wait_for_selector"):
            try:
                await page.wait_for_selector(", ".join(self.INPUT_SELECTORS), timeout=8000)
            except Exception:
                pass

        # 1. Locate textarea / ProseMirror contenteditable
        target_input = None
        for sel in self.INPUT_SELECTORS:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                target_input = loc
                break

        if not target_input:
            raise RuntimeError("Could not find ChatGPT input textarea on page.")

        # 2. Fill and submit (with keyboard.insert_text fallback for ProseMirror divs)
        await target_input.click()
        try:
            await target_input.fill(prompt)
        except Exception:
            if hasattr(page, "keyboard"):
                await page.keyboard.insert_text(prompt)
        await asyncio.sleep(0.4)

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

        # 3. Wait for assistant message to appear and finish streaming
        await asyncio.sleep(2.0)
        start_time = asyncio.get_event_loop().time()
        last_text = ""

        while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
            for resp_sel in self.RESPONSE_SELECTORS:
                resps = page.locator(resp_sel)
                count = await resps.count()
                if count > 0:
                    last_msg = resps.nth(count - 1)
                    curr_text = (await last_msg.inner_text() or "").strip()
                    if curr_text:
                        if curr_text == last_text and len(curr_text) > 10:
                            return curr_text
                        last_text = curr_text
                        break
            await asyncio.sleep(1.5)

        if last_text:
            return last_text.strip()
        return "ChatGPT is open in the browser window (log in or complete any verification prompt in the browser to view the reply)."
