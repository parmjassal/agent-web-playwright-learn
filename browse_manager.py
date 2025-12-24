import asyncio
import threading
import logging
from playwright.async_api import async_playwright, Page

logging.basicConfig(level=logging.INFO)


class BrowserManager:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_loop,
            daemon=True
        )
        self.thread.start()

        self.playwright = None
        self.browser = None
        self.contexts = {}

        # Start Playwright in the same loop
        asyncio.run_coroutine_threadsafe(self._start(), self.loop).result()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False, slow_mo=100)
        logging.info("Playwright started")

    # ---------- ASYNC API (same loop only) ----------

    async def get_context(self, session_id: str):
        if session_id not in self.contexts:
            logging.info(f"Creating new context: {session_id}")
            context = await self.browser.new_context()
            self.contexts[session_id] = context
        return self.contexts[session_id]

    async def new_page(self, session_id: str):
        context = await self.get_context(session_id)
        return await context.new_page()

    async def close_page(self, page: Page):
        if page.is_closed():
            return
        await page.close()

    # ---------- SYNC BRIDGE (for tools) ----------

    def new_page_sync(self, session_id: str):
        future = asyncio.run_coroutine_threadsafe(
            self.new_page(session_id),
            self.loop
        )
        return future.result()

    def close_page_sync(self, page: Page):
        future = asyncio.run_coroutine_threadsafe(
            self.close_page(page),
            self.loop
        )
        return future.result()


browser_manager = BrowserManager()
