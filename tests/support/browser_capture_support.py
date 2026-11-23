"""The stage the capture flows share: the stand-in app, its page and the service.

Chromium runs the built package on the stand-in app's page, under webauth's real
headers, against the real service over TCP with a fake hearing engine.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from playwright.async_api import async_playwright

from tests.support.browser_support import MESSAGING_HOP_PATH, AppSide, stand_in_app
from tests.support.hear_support import FakeHearing, hear_runtime
from tests.support.page_coverage_support import PageCoverage
from tests.support.tcp_server_support import serving_asgi, serving_over_tcp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from playwright.async_api import Page

CHROMIUM_FLAGS = (
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    "--autoplay-policy=document-user-activation-required",
)
WAIT_MS = 20_000
FRAMES_BEFORE_STOP = 3
# The page passes `onMessage`, and the app's route writes messages of its own.
MESSAGING = {"hop": MESSAGING_HOP_PATH, "messages": "1"}


@dataclass(frozen=True)
class Stage:
    """One test's service, app side and page."""

    side: AppSide
    page: Page
    base_url: str
    service_url: str
    coverage: PageCoverage

    async def open(self, **query: str) -> None:
        """Load the app's page for the scenario `query` names."""
        await self.page.goto(f"{self.base_url}/capture.html?{urlencode(query)}")

    async def switch(self) -> None:
        """Click the app's hearing switch."""
        await self.page.click("#switch")

    async def until(self, condition: str) -> None:
        """Wait until `condition` holds on `seen`, what the page saw."""
        await self.page.wait_for_function(
            f"() => (seen => {condition})(window.capturePage)", timeout=WAIT_MS
        )

    async def seen(self) -> dict[str, object]:
        """What the page saw."""
        return await self.page.evaluate("window.capturePage")

    async def switch_on_until_streaming(self, **query: str) -> None:
        """Open the page, switch hearing on and wait until frames leave."""
        await self.open(**query)
        await self.switch()
        await self.until(f"seen.sentBytes.length >= {FRAMES_BEFORE_STOP}")

    def send(self, *messages: str | bytes | None) -> None:
        """Have the messaging route write `messages`; `None` ends the hop."""
        for message in messages:
            self.side.outbox.put_nowait(message)

    async def held(self) -> str:
        """Every value the page holds, the messages the app was handed included."""
        held = await self.page.evaluate(
            "() => [document.documentElement.outerHTML, location.href,"
            " JSON.stringify(localStorage), JSON.stringify(sessionStorage),"
            " document.cookie, JSON.stringify(window.capturePage)].join('\\n')"
        )
        return held + repr(await self.page.context.cookies())

    async def switch_off(self) -> dict[str, object]:
        """Switch hearing off and return what the page saw once the capture ended."""
        await self.coverage.take_worklets()
        await self.switch()
        await self.until("seen.ended !== null")
        return await self.seen()


@asynccontextmanager
async def capture_stage(
    tmp_path: Path, *, hearing_available: bool = True
) -> AsyncIterator[Stage]:
    engine = FakeHearing() if hearing_available else None
    async with serving_over_tcp(hear_runtime(engine)) as service:
        side = AppSide.new()
        app = stand_in_app(service.base_url, side, tmp_path)
        async with serving_asgi(app) as base_url, async_playwright() as playwright:
            browser = await playwright.chromium.launch(args=CHROMIUM_FLAGS)
            try:
                page = await browser.new_page()
                coverage = PageCoverage.from_environment()
                await coverage.watch(page)
                yield Stage(side, page, base_url, service.base_url, coverage)
                await coverage.write()
            finally:
                await browser.close()
