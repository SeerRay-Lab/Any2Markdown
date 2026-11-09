"""Playwright-based fallback fetcher for sites with JS challenges or SPAs.

Used when ``url_fetcher.fetch_bytes`` would be defeated by a JavaScript
anti-bot challenge (e.g. Zhihu's zse-ck) or when the page is a client-side
rendered SPA whose meaningful content only appears after hydration.

A single Chromium browser is launched lazily on first use and shared across
requests; each request gets a fresh browser context so cookies / storage do
not leak between sources.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Reasonable desktop Chrome UA. Real-Chromium-running-real-JS is already a
# high bar for anti-bot — UA mainly matters for sites that gate behavior on it.
DEFAULT_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Chromium launch args. --no-sandbox is required when running as root in a
# container; AutomationControlled flag removal hides the most obvious
# Playwright fingerprint.
_CHROMIUM_ARGS = [
    "--no-sandbox",
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
]


class HeadlessFetchError(Exception):
    """Raised when the headless browser cannot fetch the URL."""


class BrowserPool:
    """Singleton-ish wrapper around a long-lived Chromium browser."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._playwright: Any = None
        self._browser: Any = None

    async def get(self) -> Any:
        if self._browser is not None and self._browser.is_connected():
            return self._browser
        async with self._lock:
            if self._browser is not None and self._browser.is_connected():
                return self._browser
            try:
                from playwright.async_api import async_playwright
            except ImportError as e:
                raise HeadlessFetchError(
                    "playwright is not installed; rebuild the image with the "
                    "playwright + chromium dependencies"
                ) from e
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True, args=_CHROMIUM_ARGS,
            )
            logger.info("headless chromium launched")
            return self._browser

    async def close(self) -> None:
        async with self._lock:
            if self._browser is not None:
                try:
                    await self._browser.close()
                except Exception:
                    logger.exception("browser close failed")
                self._browser = None
            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                except Exception:
                    logger.exception("playwright stop failed")
                self._playwright = None


async def fetch_html_via_browser(
    browser: Any,
    url: str,
    *,
    timeout: float = 30.0,
    wait_for_selector: str | None = None,
    selector_timeout: float = 8.0,
    user_agent: str = DEFAULT_BROWSER_UA,
) -> str:
    """Navigate to ``url`` in a fresh context and return rendered HTML.

    Args:
        browser: Pre-launched Playwright Browser.
        url: Target URL.
        timeout: Total navigation timeout (seconds).
        wait_for_selector: If set, wait for this CSS selector to appear before
            grabbing HTML. Falls through silently if it never appears.
        selector_timeout: Timeout for the selector wait (seconds).
        user_agent: UA string for the browser context.
    """
    from playwright.async_api import TimeoutError as PWTimeoutError

    context = await browser.new_context(
        user_agent=user_agent,
        viewport={"width": 1280, "height": 900},
        locale="zh-CN",
        timezone_id="Asia/Shanghai",
        extra_http_headers={
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    try:
        page = await context.new_page()
        try:
            await page.goto(url, timeout=int(timeout * 1000), wait_until="domcontentloaded")
        except PWTimeoutError as e:
            raise HeadlessFetchError(f"navigation timeout after {timeout}s") from e
        except Exception as e:
            raise HeadlessFetchError(f"navigation failed: {e}") from e

        if wait_for_selector:
            try:
                await page.wait_for_selector(
                    wait_for_selector, timeout=int(selector_timeout * 1000)
                )
            except PWTimeoutError:
                # Selector didn't show up — return whatever we have. Caller
                # decides whether content is sufficient.
                logger.info("selector %r not found within %ss for %s",
                            wait_for_selector, selector_timeout, url)

        return await page.content()
    finally:
        try:
            await context.close()
        except Exception:
            logger.exception("context close failed")


# Module-level pool — used by server via lazy access.
pool = BrowserPool()
