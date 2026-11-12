"""An app's browser plays sounds through the package's player, one after another.

Chromium runs the built package on the stand-in app's page, under webauth's real
headers and the autoplay policy. The app's server speaks through the real service
over TCP with a fake voice; the sounds that need a chosen length, a hold, a failure
or a cut-off are test-written and served by the stand-in app. The page's measuring
instrument records the graph, the decodes and the resumes, and reads the output's
energy.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import pytest
from webauth.policies import default_content_security_policy

from speech_client.wire import VoiceId
from speech_service.service import Runtime
from tests.support.browser_support import (
    CUT_PATH,
    HELD_PATH,
    SOUNDS_PATH,
    VOICE_PATH,
    AppSide,
    chromium_page,
    stand_in_app,
    write_sound,
)
from tests.support.fakes_support import FakeSpeaking
from tests.support.speak_support import info_comment_before_data
from tests.support.tcp_server_support import serving_asgi, serving_over_tcp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from playwright.async_api import Page

pytestmark = pytest.mark.browser

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
SOUNDS_STARTED_AROUND_THE_MUTE = 2
RESUMES_REJECTED_THEN_PASSED = 2
ALL_ENDED = "s.sounds.every(sound => sound.endedValue !== null)"
SHORT = f"{SOUNDS_PATH}/short.wav"
OTHER = f"{SOUNDS_PATH}/other.wav"
LONG = f"{SOUNDS_PATH}/long.wav"
NATURAL = f"{SOUNDS_PATH}/natural.wav"
HELD = f"{HELD_PATH}/short.wav"
CUT = f"{CUT_PATH}/short.wav"
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


def _ends(seen: dict[str, object]) -> list[str]:
    return [sound["endedValue"] for sound in seen["sounds"]]


def _starts(seen: dict[str, object]) -> list[str]:
    return [sound["startedValue"] for sound in seen["sounds"]]


def _events(seen: dict[str, object]) -> list[tuple[str, int]]:
    return [(entry["event"], entry["id"]) for entry in seen["log"]]


@asynccontextmanager
async def _stage(tmp_path: Path) -> AsyncIterator[Stage]:
    for name, seconds in SOUND_SECONDS.items():
        write_sound(tmp_path / name, seconds)
    runtime = Runtime({VoiceId.THORSTEN: FakeSpeaking()}, memory_probe=lambda: 0)
    async with serving_over_tcp(runtime) as service:
        side = AppSide.new()
        app = stand_in_app(service.base_url, side, tmp_path)
        async with serving_asgi(app) as base_url, chromium_page() as (page, _):
            yield Stage(side, page, base_url)


@asynccontextmanager
async def _activated(tmp_path: Path) -> AsyncIterator[Stage]:
    async with _stage(tmp_path) as stage:
        await stage.open()
        await stage.activate()
        yield stage


def test_three_sounds_play_in_the_order_handed_over_and_never_at_once(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with _activated(tmp_path) as stage:
            await stage.prepare_bytes()
            await stage.hand_over(url(SHORT), bytes_of(VOICE_BYTES), url(OTHER))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert _ends(seen) == ["played"] * 3
    assert _events(seen) == [
        (event, sound) for sound in range(3) for event in ("started", "ended")
    ]
    assert seen["maxActiveSources"] == 1


def test_a_sound_whose_answer_is_held_plays_before_the_later_one_that_decoded(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with _activated(tmp_path) as stage:
            await stage.prepare_bytes()
            await stage.hand_over(url(HELD), bytes_of(VOICE_BYTES))
            await stage.until("s.decodesSettled >= 1")
            while_held = await stage.snapshot()
            stage.side.held_release.set()
            return while_held, await stage.finish()

    while_held, seen = asyncio.run(scenario())

    assert while_held["sourcesStarted"] == 0
    assert _starts(while_held) == [None, None]
    assert _events(seen) == [
        (event, sound) for sound in range(2) for event in ("started", "ended")
    ]


def test_bytes_from_the_service_play_with_their_ai_mark_and_are_heard(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], AppSide]:
        async with _activated(tmp_path) as stage:
            await stage.prepare_bytes()
            await stage.hand_over(bytes_of(VOICE_BYTES))
            return await stage.finish(), stage.side

    seen, side = asyncio.run(scenario())

    assert info_comment_before_data(side.spoken[0]) is not None
    assert _ends(seen) == ["played"]
    assert seen["sounds"][0]["peakRms"] > AUDIBLE_RMS


@pytest.mark.parametrize("handed_as", ["url", "urlObject"])
def test_a_route_of_the_app_plays_once_whether_handed_as_text_or_as_a_url_object(
    tmp_path: Path, handed_as: str
) -> None:
    async def scenario() -> tuple[dict[str, object], AppSide]:
        async with _activated(tmp_path) as stage:
            await stage.hand_over({handed_as: VOICE_PATH})
            return await stage.finish(), stage.side

    seen, side = asyncio.run(scenario())

    assert _ends(seen) == ["played"]
    assert seen["sounds"][0]["peakRms"] > AUDIBLE_RMS
    assert side.requests == [VOICE_PATH]


def test_stop_ends_the_playing_and_waiting_sounds_at_once_and_aborts_the_held_fetch(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], float]:
        async with _activated(tmp_path) as stage:
            await stage.hand_over(url(LONG), url(HELD), url(SHORT))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.side.arrival(HELD)
            await stage.until(f"s.sounds[0].peakRms > {AUDIBLE_RMS}")
            await stage.act("stop")
            silent_ms = await stage.silent_within()
            await asyncio.wait_for(stage.side.held_aborted.wait(), WAIT_SECONDS)
            return await stage.snapshot(), silent_ms

    seen, silent_ms = asyncio.run(scenario())

    assert seen["endedAtAction"] == ["stopped"] * 3
    assert _starts(seen) == ["playing", "ended", "ended"]
    assert silent_ms < STOP_BOUND_MS


def test_a_stopped_sound_reports_nothing_more_at_its_natural_end_and_starts_nothing(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with _activated(tmp_path) as stage:
            await stage.hand_over(url(NATURAL), url(SHORT))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.act("stop")
            await stage.until("s.sourcesEnded >= 1")
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert _events(seen) == [
        (event, sound) for sound in range(2) for event in ("started", "ended")
    ]
    assert _ends(seen) == ["stopped", "stopped"]
    assert seen["sourcesStarted"] == 1


def test_a_sound_handed_over_after_stop_plays(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with _activated(tmp_path) as stage:
            await stage.hand_over(url(LONG))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.act("stop")
            await stage.hand_over(url(SHORT))
            await stage.until("s.sounds[1].endedValue !== null")
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert _ends(seen) == ["stopped", "played"]
    assert seen["sourcesStarted"] == len(seen["sounds"])


def test_stop_twice_on_an_idle_player_reports_nothing(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with _stage(tmp_path) as stage:
            await stage.open()
            await stage.act("stop")
            await stage.act("stop")
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert seen["log"] == []
    assert seen["sounds"] == []


def test_muting_ends_the_playing_and_waiting_sounds_at_once_and_silences_the_output(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], float]:
        async with _activated(tmp_path) as stage:
            await stage.hand_over(url(LONG), url(SHORT))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.until(f"s.sounds[0].peakRms > {AUDIBLE_RMS}")
            await stage.set_muted(muted=True)
            silent_ms = await stage.silent_within()
            return await stage.snapshot(), silent_ms

    seen, silent_ms = asyncio.run(scenario())

    assert seen["endedAtAction"] == ["muted", "muted"]
    assert _starts(seen) == ["playing", "ended"]
    assert seen["gainValue"] == 0
    assert silent_ms < STOP_BOUND_MS


def test_a_sound_handed_over_while_muted_ends_muted_and_is_not_fetched(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], AppSide]:
        async with _activated(tmp_path) as stage:
            await stage.set_muted(muted=True)
            await stage.hand_over(url(SHORT), url(OTHER))
            return await stage.finish(), stage.side

    seen, side = asyncio.run(scenario())

    assert _ends(seen) == ["muted", "muted"]
    assert _starts(seen) == ["ended", "ended"]
    assert side.requests == []
    assert seen["decodesCalled"] == 0


def test_unmuting_plays_the_next_sound_and_none_handed_over_before(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with _activated(tmp_path) as stage:
            await stage.hand_over(url(LONG), url(SHORT))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.set_muted(muted=True)
            await stage.set_muted(muted=False)
            await stage.hand_over(url(OTHER))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert _ends(seen) == ["muted", "muted", "played"]
    assert seen["sourcesStarted"] == SOUNDS_STARTED_AROUND_THE_MUTE
    assert seen["gainValue"] == 1
    assert seen["sounds"][2]["peakRms"] > AUDIBLE_RMS


@pytest.mark.parametrize(
    ("broken", "failure"),
    [(url(ABSENT), "a route answering 404"), (JUNK, "bytes that do not decode")],
)
def test_a_sound_that_cannot_load_fails_and_the_next_one_plays(
    tmp_path: Path, broken: dict[str, str], failure: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with _activated(tmp_path) as stage:
            await stage.hand_over(broken, url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert _ends(seen) == ["failed", "played"], failure
    assert _starts(seen) == ["ended", "playing"]


def test_a_route_that_aborts_mid_body_fails_and_the_next_one_plays(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with _activated(tmp_path) as stage:
            await stage.hand_over(url(CUT), url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert _ends(seen) == ["failed", "played"]


def test_a_sound_handed_over_without_a_click_is_refused_and_plays_after_one(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with _stage(tmp_path) as stage:
            ended = asyncio.Event()
            await stage.page.expose_function("noteEnded", lambda _sound: ended.set())
            await stage.open(handOver=SHORT)
            await asyncio.wait_for(ended.wait(), WAIT_SECONDS)
            refused = await stage.snapshot()
            await stage.activate()
            await stage.hand_over(url(SHORT))
            return refused, await stage.finish()

    refused, seen = asyncio.run(scenario())

    assert _ends(refused) == ["refused"]
    assert _starts(refused) == ["ended"]
    assert refused["resumesCalled"] == 0
    assert _ends(seen) == ["refused", "played"]
    assert seen["resumesCalled"] == 1


def test_a_rejected_resume_refuses_the_sound_and_the_next_one_plays(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with _activated(tmp_path) as stage:
            await stage.act("rejectNextResume")
            await stage.hand_over(url(SHORT))
            await stage.until("s.sounds[0].endedValue !== null")
            await stage.hand_over(url(OTHER))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert _ends(seen) == ["refused", "played"]
    assert seen["resumesCalled"] == RESUMES_REJECTED_THEN_PASSED


def test_every_source_connects_only_to_the_one_gain_that_is_the_outputs_only_input(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with _activated(tmp_path) as stage:
            await stage.hand_over(url(SHORT), url(OTHER), url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())
    edges = seen["edges"]
    [into_output] = [edge for edge in edges if edge["toKind"] == "AudioDestinationNode"]
    from_sources = [
        edge for edge in edges if edge["fromKind"] == "AudioBufferSourceNode"
    ]

    assert into_output["fromKind"] == "GainNode"
    assert len(from_sources) == seen["sourcesStarted"] == len(seen["sounds"])
    assert {edge["to"] for edge in from_sources} == {into_output["from"]}


def test_the_package_builds_no_tap_and_exports_none(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with _activated(tmp_path) as stage:
            await stage.hand_over(url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert seen["analysersCreated"] == 0
    assert "createPlayer" in seen["exportNames"]
    assert not [name for name in seen["exportNames"] if TAP_NAMES.search(name)]


def test_the_page_holds_none_of_the_tokens_the_apps_server_spoke_with(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[str, tuple[str, str]]:
        async with _activated(tmp_path) as stage:
            await stage.hand_over(url(VOICE_PATH))
            await stage.finish()
            held = await stage.page.evaluate(
                "() => [document.documentElement.outerHTML, location.href,"
                " JSON.stringify(localStorage), JSON.stringify(sessionStorage),"
                " document.cookie].join('\\n')"
            )
            cookies = await stage.page.context.cookies()
            return held + repr(cookies), stage.side.tokens

    held, tokens = asyncio.run(scenario())

    assert not any(token in held for token in tokens)


def test_the_player_runs_under_webauths_default_policy(tmp_path: Path) -> None:
    async def scenario() -> tuple[str, dict[str, object]]:
        async with _activated(tmp_path) as stage:
            page = await stage.page.request.get(f"{stage.base_url}/player.html")
            await stage.hand_over(url(SHORT))
            return page.headers["content-security-policy"], await stage.finish()

    policy, seen = asyncio.run(scenario())

    assert policy == default_content_security_policy()
    assert _ends(seen) == ["played"]


def test_close_ends_every_sound_closes_the_context_and_later_sounds_end_stopped(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with _activated(tmp_path) as stage:
            await stage.hand_over(url(LONG), url(SHORT))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.act("close")
            at_close = await stage.snapshot()
            await stage.until(f"s.contextStates.every(state => state === '{CLOSED}')")
            await stage.hand_over(url(SHORT))
            await stage.act("close")
            seen = await stage.finish()
            return {**seen, "atClose": at_close["endedAtAction"]}

    seen = asyncio.run(scenario())

    assert seen["atClose"] == ["stopped", "stopped"]
    assert _ends(seen) == ["stopped"] * 3
    assert _starts(seen) == ["playing", "ended", "ended"]
    assert seen["contextStates"] == [CLOSED]


def test_the_bytes_handed_over_keep_their_length(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with _activated(tmp_path) as stage:
            await stage.prepare_bytes()
            await stage.hand_over(bytes_of(VOICE_BYTES))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert seen["bytes"][VOICE_BYTES]["before"] > 0
    assert seen["bytes"][VOICE_BYTES]["after"] == seen["bytes"][VOICE_BYTES]["before"]


def test_a_stop_in_the_task_of_a_natural_end_gives_one_report_and_starts_nothing(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with _activated(tmp_path) as stage:
            await stage.act("stopOnNextSourceEnd")
            await stage.hand_over(url(SHORT), url(OTHER))
            await stage.finish()
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert seen["instrumentStops"] == 1
    assert seen["sourcesStarted"] == 1
    assert _ends(seen) == ["stopped", "stopped"]
    assert [sound["endedCount"] for sound in seen["sounds"]] == [1, 1]


@pytest.mark.parametrize("held", ["decode", "resume"])
@pytest.mark.parametrize("switch", ["stop", "mute"])
def test_a_switch_while_the_head_is_held_leaves_it_ended_and_the_new_sound_plays_once(
    tmp_path: Path, held: str, switch: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with _activated(tmp_path) as stage:
            await stage.prepare_bytes()
            await stage.act("hold", held)
            await stage.hand_over(bytes_of(VOICE_BYTES))
            await stage.until(f"s.{held}sCalled >= 1")
            if switch == "stop":
                await stage.act("stop")
            else:
                await stage.set_muted(muted=True)
                await stage.set_muted(muted=False)
            await stage.hand_over(bytes_of(VOICE_BYTES))
            await stage.until("s.sounds[1].endedValue !== null")
            await stage.act("release")
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert _ends(seen) == ["stopped" if switch == "stop" else "muted", "played"]
    assert _starts(seen) == ["ended", "playing"]
    assert seen["sourcesStarted"] == 1
    assert [sound["endedCount"] for sound in seen["sounds"]] == [1, 1]
