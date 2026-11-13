"""The stage the player flows share: the stand-in app, its page and the instrument.

Chromium runs the built package on the stand-in app's page, under webauth's real
headers and the autoplay policy. The app's server speaks through the real service
over TCP with a fake voice; the sounds that need a chosen length, a hold, a failure
or a cut-off are test-written and served by the stand-in app. The page's measuring
instrument records the graph, the decodes and the resumes, and reads the output's
energy.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from speech_client.wire import VoiceId
from speech_service.service import Runtime
from tests.support.browser_support import (
    CUT_LATE_PATH,
    CUT_PATH,
    HELD_PATH,
    SOUNDS_PATH,
    STREAMED_PATH,
    VOICE_PATH,
    AppSide,
    chromium_page,
    stand_in_app,
    write_sound,
)
from tests.support.fakes_support import FakeSpeaking
from tests.support.tcp_server_support import serving_asgi, serving_over_tcp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from playwright.async_api import Page

WAIT_MS = 20_000
WAIT_SECONDS = 20
STOP_BOUND_MS = 1000
SETTLE_MS = 300
SOUND_SECONDS = {
    "short.wav": 0.4,
    "other.wav": 0.4,
    "long.wav": 4.0,
    "natural.wav": 1.2,
}
SILENT_RMS = 0.001
AUDIBLE_RMS = 0.05
CLOSED = "closed"
RESUMES_REJECTED_THEN_PASSED = 2
ALL_ENDED = "s.sounds.every(sound => sound.endedValue !== null)"
SHORT = f"{SOUNDS_PATH}/short.wav"
OTHER = f"{SOUNDS_PATH}/other.wav"
LONG = f"{SOUNDS_PATH}/long.wav"
NATURAL = f"{SOUNDS_PATH}/natural.wav"
HELD = f"{HELD_PATH}/short.wav"
CUT = f"{CUT_PATH}/short.wav"
STREAMED = f"{STREAMED_PATH}/long.wav"
CUT_LATE = f"{CUT_LATE_PATH}/long.wav"
ABSENT = "/absent.wav"
VOICE_BYTES = "voice"
TAP_NAMES = re.compile(r"analys|tap|level|meter|energy", re.IGNORECASE)


def url(path: str) -> dict[str, str]:
    """A sound the app hands over as the URL of its own route."""
    return {"url": path}


def bytes_of(key: str) -> dict[str, str]:
    """A sound the app hands over as the bytes the page fetched earlier under `key`."""
    return {"bytes": key}


JUNK = {"junk": "bytes no decoder reads"}


@dataclass(frozen=True)
class Stage:
    """One test's app side and page."""

    side: AppSide
    page: Page
    base_url: str

    async def open(self, **query: str) -> None:
        """Load the app's page, its player there from the start.

        `query` may name a sound for the page to hand over while it loads.
        """
        await self.page.goto(f"{self.base_url}/player.html?{urlencode(query)}")

    async def activate(self) -> None:
        """Click once, which gives the page user activation."""
        await self.page.click("#click")

    async def prepare_bytes(
        self, key: str = VOICE_BYTES, route: str = VOICE_PATH
    ) -> None:
        """Fetch the bytes of the app's `route` into the page, held under `key`."""
        await self.page.evaluate(
            "([key, route]) => window.playerPage.prepareBytes(key, route)", [key, route]
        )

    async def hand_over(self, *sounds: dict[str, str]) -> list[int]:
        """Hand the sounds to the player in one task; their ids on the page."""
        return await self.page.evaluate(
            "sounds => window.playerPage.play(...sounds)", list(sounds)
        )

    async def act(self, action: str, *arguments: object) -> None:
        """Do the app's switch `action` of the page and wait until its task is over."""
        await self.page.evaluate(
            f"arguments => window.playerPage.{action}(...arguments)", list(arguments)
        )

    async def set_muted(self, *, muted: bool) -> None:
        """Flip the app's mute switch and wait until its task is over."""
        await self.act("mute", muted)

    async def until(self, condition: str) -> None:
        """Wait until `condition` holds on `s`, what the page saw."""
        await self.page.wait_for_function(
            f"() => (s => {condition})(window.playerPage.snapshot())", timeout=WAIT_MS
        )

    async def settle(self) -> None:
        """Let a bounded stretch of page time pass, to show nothing more happens."""
        await self.page.evaluate(
            "ms => new Promise(resolve => setTimeout(resolve, ms))", SETTLE_MS
        )

    async def snapshot(self) -> dict[str, object]:
        """What the page saw, after the checks that hold in every cell.

        Each sound is reported at most once each way, its start before its end.
        """
        seen = await self.page.evaluate("window.playerPage.snapshot()")
        assert seen["errors"] == []
        for sound in seen["sounds"]:
            assert sound["startedCount"] <= 1
            assert sound["endedCount"] <= 1
            events = [
                entry["event"] for entry in seen["log"] if entry["id"] == sound["id"]
            ]
            assert events in ([], ["started"], ["started", "ended"])
        return seen

    async def finish(self) -> dict[str, object]:
        """Wait until every sound has ended, then what the page saw."""
        await self.until(ALL_ENDED)
        return await self.snapshot()

    async def silent_within(self) -> float:
        """Milliseconds from the last switch of the app until the output read silent."""
        return await self.page.evaluate(
            "threshold => window.playerPage.silentAfterAction(threshold)", SILENT_RMS
        )


def ends(seen: dict[str, object]) -> list[str]:
    return [sound["endedValue"] for sound in seen["sounds"]]


def starts(seen: dict[str, object]) -> list[str]:
    return [sound["startedValue"] for sound in seen["sounds"]]


def events(seen: dict[str, object]) -> list[tuple[str, int]]:
    return [(entry["event"], entry["id"]) for entry in seen["log"]]


@asynccontextmanager
async def unactivated(tmp_path: Path) -> AsyncIterator[Stage]:
    for name, seconds in SOUND_SECONDS.items():
        write_sound(tmp_path / name, seconds)
    runtime = Runtime({VoiceId.THORSTEN: FakeSpeaking()}, memory_probe=lambda: 0)
    async with serving_over_tcp(runtime) as service:
        side = AppSide.new()
        app = stand_in_app(service.base_url, side, tmp_path)
        async with serving_asgi(app) as base_url, chromium_page() as (page, _):
            yield Stage(side, page, base_url)


@asynccontextmanager
async def activated(tmp_path: Path) -> AsyncIterator[Stage]:
    async with unactivated(tmp_path) as stage:
        await stage.open()
        await stage.activate()
        yield stage
