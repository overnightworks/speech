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
from itertools import pairwise
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from shipped_catalogue import CATALOGUE
from speech_client.wire import VoiceId
from speech_service.pcm import BYTES_PER_SAMPLE
from speech_service.service import Runtime, RuntimeDependencies
from tests.support.browser_support import (
    CUT_LATE_PATH,
    CUT_PATH,
    HEADED_LEAD_BYTES,
    HEADED_PATH,
    HELD_PATH,
    SOUNDS_PATH,
    STREAMED_PATH,
    VOICE_PATH,
    AppSide,
    chromium_page,
    stand_in_app,
    write_marker_sound,
    write_sound,
)
from tests.support.fakes_support import (
    PCM_FULL_SCALE,
    SPEAK_SAMPLE_RATE,
    TONE_AMPLITUDE,
    FakeSpeaking,
)
from tests.support.tcp_server_support import serving_asgi, serving_over_tcp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
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
# The tone's amplitude as the ear reads it: a quantum of the tone peaks here.
TONE_PEAK = TONE_AMPLITUDE / PCM_FULL_SCALE
# The ear reports the peak of each render quantum, `RENDER_QUANTUM_FRAMES` samples.
RENDER_QUANTUM_FRAMES = 128
# A quantum under this is silent: 26 dB below the tone, no step to or from it is heard.
SILENT_PEAK = 0.05 * TONE_PEAK
FULL_PEAK = 0.95 * TONE_PEAK
# The longest an edge of the volume may take; the largest step between its quanta.
EDGE_SECONDS = 0.2
CLICK_STEP = TONE_PEAK / 2
# Fewest quanta between full volume and silence that a ramp, not a cut, shows.
EDGE_QUANTA = 2
# Fewest quanta of a whole run that must pass the volume check for it to mean anything.
MIN_QUANTA_CHECKED = 100
# What a hold may take off a sound's audible length: the context renders on for
# a moment between the end of the edge and the suspend.
LOST_AT_EDGES_SECONDS = 0.25
# The marker sound's tone, gap and short tone: the gap places its position.
MARKER_SECONDS = (2.0, 1.0, 0.5)
SILENT_RMS = 0.001
AUDIBLE_RMS = 0.05
# One render quantum plus the detection sample and the tone's silent first sample.
ONSET_BOUND_FRAMES = 128 + 2
CLOSED = "closed"
RESUMES_REJECTED_THEN_PASSED = 2
# The context resume that starts the first sound, and the one after a pause.
RESUMES_FOR_START_AND_RESUME = 2
ALL_ENDED = "s.sounds.every(sound => sound.endedValue !== null)"
SHORT = f"{SOUNDS_PATH}/short.wav"
OTHER = f"{SOUNDS_PATH}/other.wav"
LONG = f"{SOUNDS_PATH}/long.wav"
NATURAL = f"{SOUNDS_PATH}/natural.wav"
HELD = f"{HELD_PATH}/short.wav"
HEADED = f"{HEADED_PATH}/short.wav"
CUT = f"{CUT_PATH}/short.wav"
STREAMED = f"{STREAMED_PATH}/long.wav"
CUT_LATE = f"{CUT_LATE_PATH}/long.wav"
MARKER_NAME = "marker.wav"
MARKER = f"{SOUNDS_PATH}/{MARKER_NAME}"
MARKER_STREAMED = f"{STREAMED_PATH}/{MARKER_NAME}"
ABSENT = "/absent.wav"
VOICE_BYTES = "voice"
TAP_NAMES = re.compile(r"analys|tap|level|meter|energy", re.IGNORECASE)


def url(path: str) -> dict[str, str]:
    """A sound the app hands over as the URL of its own route."""
    return {"url": path}


def bytes_of(key: str) -> dict[str, str]:
    """A sound the app hands over as the bytes the page fetched earlier under `key`."""
    return {"bytes": key}


def fed(key: str, *, lead_seconds: float = 0) -> dict[str, str | int]:
    """A streamed sound the page feeds: the bytes under `key`, headed at once.

    The first `lead_seconds` of its samples come with the header.
    """
    lead_samples = int(lead_seconds * SPEAK_SAMPLE_RATE)
    return {"fed": key, "lead": HEADED_LEAD_BYTES + lead_samples * BYTES_PER_SAMPLE}


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

    async def hand_over(self, *sounds: Mapping[str, str | int]) -> list[int]:
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


def quantum_seconds(seen: dict[str, object]) -> float:
    return RENDER_QUANTUM_FRAMES / seen["sampleRate"]


def peaks_between(seen: dict[str, object], since: float, until: float) -> list[float]:
    """The peaks of the quanta that start in [`since`, `until`) of the context clock."""
    return [peak for at, peak in seen["heardPeaks"] if since <= at < until]


def start_of(seen: dict[str, object], sound: int = 0) -> float:
    """The context time at which the app's sound number `sound` starts."""
    return seen["sounds"][sound]["start"]["contextTime"]


def heard_start_of(seen: dict[str, object], sound: int = 0) -> float:
    """The context time at which the app's sound number `sound` is first heard.

    A page task that stalls past the start the player chose leaves the render
    thread to play the sound from its next quantum, so the heard start is the
    announced one or later. A negative `sound` counts from the last.
    """
    return seen["heardOnsets"][sound]


def peaks_inside(seen: dict[str, object], sound: int, seconds: float) -> list[float]:
    """The peaks of sound number `sound`, `seconds` long, less 3 quanta at each end."""
    start = start_of(seen, sound)
    margin = 3 * quantum_seconds(seen)
    return peaks_between(seen, start + margin, start + seconds - margin)


def audible_seconds(seen: dict[str, object], since: float, until: float) -> float:
    """The seconds of [`since`, `until`) in which the output was not silent."""
    audible = [peak for peak in peaks_between(seen, since, until) if peak > SILENT_PEAK]
    return len(audible) * quantum_seconds(seen)


def action_times(seen: dict[str, object], name: str) -> list[float]:
    """The context times at which the app's switch `name` was thrown, in order."""
    return [action["at"] for action in seen["actions"] if action["name"] == name]


def action_gains(seen: dict[str, object], name: str) -> list[float]:
    """The volume the output had when the app's switch `name` was thrown, in order."""
    return [action["gain"] for action in seen["actions"] if action["name"] == name]


def ramp_end(seen: dict[str, object], target: float) -> float:
    """The context time the player's first ramp of the volume to `target` ends."""
    return next(ramp["endsAt"] for ramp in seen["ramps"] if ramp["target"] == target)


def edge_peaks(
    seen: dict[str, object], name: str, until: float | None = None
) -> list[float]:
    """The peaks of the quanta from the first throw of switch `name` to its edge's end.

    The edge ends at `until` of the context clock, or `EDGE_SECONDS` after the throw.
    """
    [at, *_] = action_times(seen, name)
    return peaks_between(
        seen,
        at - quantum_seconds(seen),
        at + EDGE_SECONDS if until is None else until,
    )


def largest_step(peaks: list[float]) -> float:
    steps = (abs(after - before) for before, after in pairwise(peaks))
    return max(steps, default=0)


def tone_quanta(seen: dict[str, object], start: float) -> list[tuple[float, float]]:
    """The (time, peak) of the quanta inside the tones of a marker sound at `start`."""
    tone, gap, tail = MARKER_SECONDS
    margin = 3 * quantum_seconds(seen)
    tones = [(start, start + tone), (start + tone + gap, start + tone + gap + tail)]
    return [
        (at, peak)
        for at, peak in seen["heardPeaks"]
        if any(begin + margin <= at <= end - margin for begin, end in tones)
    ]


def peaks_outside_edges(seen: dict[str, object], start: float) -> list[float]:
    """The peaks of the marker sound's quanta, leaving out each switch's edge."""
    quantum = quantum_seconds(seen)
    thrown = [action["at"] for action in seen["actions"]]
    return [
        peak
        for at, peak in tone_quanta(seen, start)
        if not any(when - quantum <= at < when + EDGE_SECONDS for when in thrown)
    ]


@asynccontextmanager
async def unactivated(tmp_path: Path) -> AsyncIterator[Stage]:
    for name, seconds in SOUND_SECONDS.items():
        write_sound(tmp_path / name, seconds)
    write_marker_sound(tmp_path / MARKER_NAME, *MARKER_SECONDS)
    runtime = Runtime(
        {VoiceId.THORSTEN: FakeSpeaking()},
        memory_probe=lambda: 0,
        dependencies=RuntimeDependencies(catalogue=CATALOGUE),
    )
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
        await stage.page.evaluate("window.playerPage.listening()")
        yield stage
