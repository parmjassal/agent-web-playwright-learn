# browser_manager.py
from playwright.async_api import async_playwright
import asyncio

class BrowserManager:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.pages = {}
        self._lock = asyncio.Lock()

    async def start(self):
        if self.browser is None:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)

    async def get_page(self, session_id: str):
        if self.browser is None:
            await self.start()

        async with self._lock:
            if session_id not in self.pages:
                context = await self.browser.new_context()
                page = await context.new_page()
                self.pages[session_id] = page
            return self.pages[session_id]

    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

browser_manager = BrowserManager()
browser_manager.start()