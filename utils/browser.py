from __future__ import annotations

from typing import Optional

from playwright.async_api import Browser, async_playwright


class BrowserSession:
    def __init__(self, headless: bool, timeout_ms: int) -> None:
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._playwright = None
        self.browser: Optional[Browser] = None

    async def __aenter__(self) -> Browser:
        self._playwright = await async_playwright().start()
        self.browser = await self._playwright.chromium.launch(headless=self.headless)
        return self.browser

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()
