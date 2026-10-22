"""Async Playwright Browser Driver for L.O.R.A.

v1.5 Plan §4.9: Separates ephemeral clean browser contexts (for general web reading `fetch_page`,
with zero cookies/session state) from the persistent user profile context (`_ensure_context`, used
exclusively for `ask_other_ai`).
"""

import asyncio
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext, async_playwright

from lora.safety.egress import EgressFilter
from lora.tools.browser.extractor import ContentExtractor


class PlaywrightBrowserDriver:
    """Headless Chromium browser controller with ephemeral vs. persistent profile isolation and egress policy."""

    @staticmethod
    def _default_user_data_dir() -> Path:
        repo_root = Path(__file__).resolve().parents[4]
        if (repo_root / "config").is_dir():
            return repo_root / "cache" / "browser_profile"
        return Path.home() / ".lora" / "cache" / "browser_profile"

    def __init__(
        self,
        user_data_dir: Path | str | None = None,
        headless: bool = True,
        egress_filter: EgressFilter | None = None,
        extractor: ContentExtractor | None = None,
    ) -> None:
        self.user_data_dir = Path(user_data_dir) if user_data_dir else self._default_user_data_dir()
        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.egress_filter = egress_filter or EgressFilter()
        self.extractor = extractor or ContentExtractor()

        self._playwright = None
        self._context: BrowserContext | None = None
        self._ephemeral_browser = None
        self._loop = None

    async def _reset_context(self) -> None:
        """Safely discard broken or closed Playwright handles."""
        ctx, br, pw = self._context, self._ephemeral_browser, self._playwright
        self._context = None
        self._ephemeral_browser = None
        self._playwright = None
        self._loop = None
        if ctx is not None:
            try:
                await ctx.close()
            except Exception:
                pass
        if br is not None:
            try:
                await br.close()
            except Exception:
                pass
        if pw is not None:
            try:
                await pw.stop()
            except Exception:
                pass

    async def _ensure_context(self) -> BrowserContext:
        """Initialize or reuse persistent browser context (for `ask_other_ai`), recovering if loop or window closed."""
        import asyncio

        current_loop = asyncio.get_running_loop()
        if self._context is not None:
            if self._loop is not None and self._loop is not current_loop:
                self._context = None
                self._playwright = None
                self._loop = None
            else:
                try:
                    _ = self._context.pages
                    br = getattr(self._context, "browser", None)
                    if br is not None and hasattr(br, "is_connected") and not br.is_connected():
                        await self._reset_context()
                except Exception:
                    await self._reset_context()

        if self._context is None:
            if self._playwright is None:
                self._playwright = await async_playwright().start()
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.user_data_dir),
                accept_downloads=False,
                headless=self.headless,
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            )
            self._loop = current_loop
        return self._context

    async def _new_ephemeral_context(self) -> tuple[BrowserContext, bool]:
        """Return a clean, zero-cookie ephemeral BrowserContext for general web reading (`fetch_page`)."""
        import asyncio

        current_loop = asyncio.get_running_loop()
        if self._playwright is None or (self._loop is not None and self._loop is not current_loop):
            await self._reset_context()
            self._playwright = await async_playwright().start()
            self._loop = current_loop
        try:
            if self._ephemeral_browser is None or not self._ephemeral_browser.is_connected():
                self._ephemeral_browser = await self._playwright.chromium.launch(headless=True)
            ctx = await self._ephemeral_browser.new_context(accept_downloads=False,
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            )
            return ctx, True
        except Exception:
            raise RuntimeError("Cannot create isolated research browser context") from None

    async def fetch_page(self, url: str) -> dict[str, Any]:
        """Navigate to URL in a clean ephemeral context, render page, and extract clean text and metadata."""
        egress_check = self.egress_filter.check_egress(url, require_typing_allowlist=False)
        if not egress_check.allowed:
            return {
                "status": "error",
                "message": f"Navigation blocked by egress security policy: {egress_check.reason}",
                "url": url,
            }

        target_url = egress_check.cleaned_url

        try:
            context, is_ephemeral = await self._new_ephemeral_context()
            page = await context.new_page()

            async def intercept_route(route):
                check = self.egress_filter.check_egress(route.request.url, require_typing_allowlist=False)
                if not check.allowed or route.request.method != "GET" or route.request.resource_type in ["image", "media", "font"]:
                    await route.abort()
                else:
                    await route.continue_()

            await page.route("**/*", intercept_route)

            try:
                await page.goto(target_url, wait_until="domcontentloaded", timeout=15000)
                html = await page.content()
                page_title = await page.title()
            finally:
                await page.close()
                if is_ephemeral:
                    try:
                        await context.close()
                    except Exception:
                        pass

            extracted = await asyncio.to_thread(self.extractor.extract_from_html, html, source_url=target_url)

            return {
                "status": "success",
                "url": target_url,
                "title": extracted.get("title") or page_title,
                "text": extracted.get("text", ""),
                "framed_content": extracted.get("framed_content", ""),
                "is_suspicious": extracted.get("is_suspicious", False),
                "flagged_patterns": extracted.get("flagged_patterns", []),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Browser navigation to '{url}' failed: {e}",
                "url": url,
            }

    async def close(self) -> None:
        """Close browser context and playwright instance."""
        await self._reset_context()

    async def __aenter__(self) -> "PlaywrightBrowserDriver":
        await self._ensure_context()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
