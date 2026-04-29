from __future__ import annotations

import os
from typing import Optional

from playwright.async_api import Browser, async_playwright

# Chromium requires --no-sandbox when running as root inside Docker containers.
# Detected via the RUNNING_IN_DOCKER env var set in docker-compose, or by checking UID.
_IN_CONTAINER = os.getenv("RUNNING_IN_DOCKER") == "1" or os.getuid() == 0


class BrowserSession:
    def __init__(self, headless: bool, timeout_ms: int) -> None:
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._playwright = None
        self.browser: Optional[Browser] = None

    async def __aenter__(self) -> Browser:
        self._playwright = await async_playwright().start()
        args = ["--no-sandbox", "--disable-setuid-sandbox"] if _IN_CONTAINER else []
        self.browser = await self._playwright.chromium.launch(
            headless=self.headless, args=args
        )
        return self.browser

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()
