"""The stage the player flows share: the stand-in app, its page and the instrument.

Chromium runs the built package on the stand-in app's page, under webauth's real
headers and the autoplay policy. The app's server speaks through the real service
over TCP with a fake voice; the sounds that need a chosen length, a hold, a failure
or a cut-off are test-written and served by the stand-in app. The page's measuring
instrument records the graph, the decodes and the resumes, and reads the output's
energy. A cell that fails leaves what the instrument saw, with markers computed from
it, in one JSON file named by the cell, in the directory the run names. The stepped
stage runs the same page on a context the test renders quantum by quantum.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from playwright.async_api import Error as PlaywrightError

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
    MARKER_NAME,
    MARKER_SECONDS,
    PAGE_FILES,
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
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

    from playwright.async_api import Page

FAILURE_DIRECTORY_VARIABLE = "SPEECH_BROWSER_FAILURE_DIR"
STEPPED_CONTEXT = PAGE_FILES / "stepped_context.js"
WAIT_MS = 20_000
WAIT_SECONDS = 20
SOUND_SECONDS = {
    "short.wav": 0.4,
    "other.wav": 0.4,
    "long.wav": 4.0,
    "natural.wav": 1.2,
}
# The tone's amplitude as the ear reads it: a quantum of the tone peaks here.
TONE_PEAK = TONE_AMPLITUDE / PCM_FULL_SCALE
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
SILENT_RMS = 0.001
AUDIBLE_RMS = 0.05
# The tone's first sample is silent, so the ear hears a sound from its second frame.
FIRST_SAMPLE_SILENT_FRAMES = 1
CLOSED = "closed"
RESUMES_REJECTED_THEN_PASSED = 2
# The context resume that starts the first sound, and the one after a pause.
RESUMES_FOR_START_AND_RESUME = 2
ALL_ENDED = "s.sounds.every(sound => sound.endedValue !== null)"
STARTED = "s.sounds[0].startedValue === 'playing'"
HEARD = "s.heardOnsets.length > 0"
SUSPENDED = "s.contextStates[0] === 'suspended'"
# What the stepped stage plays of a sound before the cell's first switch: inside the
# first chunk of a streamed sound, which is the shortest sound a cell hears.
HEARD_FOR_SECONDS = 0.1
# A stretch the stepped stage steps to show nothing more happens: longer than an edge.
STRETCH_SECONDS = 2 * EDGE_SECONDS
# How long the stepped stage steps for a sound to end; stepping costs no page time.
FINISH_WITHIN_SECONDS = 10.0
FINISH_QUANTA = 8
SHORT = f"{SOUNDS_PATH}/short.wav"
OTHER = f"{SOUNDS_PATH}/other.wav"
LONG = f"{SOUNDS_PATH}/long.wav"
NATURAL = f"{SOUNDS_PATH}/natural.wav"
HELD = f"{HELD_PATH}/short.wav"
HEADED = f"{HEADED_PATH}/short.wav"
CUT = f"{CUT_PATH}/short.wav"
STREAMED = f"{STREAMED_PATH}/long.wav"
CUT_LATE = f"{CUT_LATE_PATH}/long.wav"
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


@dataclass
class Recording:
    """What the page last showed one cell, kept to be written if the cell fails."""

    directory: Path
    seen: dict[str, object] | None = None
    note: str | None = None

    async def take_if_none(self, page: Page) -> None:
        """Take the page's snapshot now when none was taken, and say if it cannot.

        The page has `WAIT_SECONDS` to answer, so a stalled page cannot hold on.
        """
        if self.seen is not None:
            return
        try:
            self.seen = await asyncio.wait_for(
                page.evaluate("window.playerPage.snapshot()"), WAIT_SECONDS
            )
        except (PlaywrightError, TimeoutError) as error:
            self.note = f"no snapshot: the page did not answer ({error!r})"

    def write(self, cell: str) -> Path:
        """Write the failed `cell`'s record into the directory, one file per cell."""
        record = {
            "cell": cell,
            "note": self.note,
            "markers": None if self.seen is None else markers(self.seen),
            "snapshot": self.seen,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{re.sub(r'[^\w.-]+', '_', cell)}.json"
        path.write_text(json.dumps(record, indent=1))
        return path


# The recording of the cell running now; the suite's failure hook takes it.
_running: list[Recording] = []


def take_recording() -> Recording | None:
    """The recording of the cell that ran, which no later cell then finds."""
    taken = _running[:]
    _running.clear()
    return taken[0] if taken else None


def _on_snapshot(condition: str) -> str:
    """A page function that evaluates `condition` on `s`, what the page saw."""
    return f"() => (s => {condition})(window.playerPage.snapshot())"


@dataclass(frozen=True)
class Stage:
    """One test's app side and page."""

    side: AppSide
    page: Page
    base_url: str
    recording: Recording

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
        await self.page.wait_for_function(_on_snapshot(condition), timeout=WAIT_MS)

    async def snapshot(self) -> dict[str, object]:
        """What the page saw, after the checks that hold in every cell.

        Each sound is reported at most once each way, its start before its end.
        """
        seen = await self.page.evaluate("window.playerPage.snapshot()")
        self.recording.seen = seen
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


@dataclass
class Stepping:
    """How far the stepped stage has run the page: its context's rate and clock."""

    sample_rate: int = 0
    quantum_frames: int = 0  # frames per render quantum
    clock_milliseconds: int = 0


@dataclass(frozen=True)
class SteppedStage(Stage):
    """A stage whose audio clock and page timers move only when the test steps them.

    The page's context is `stepped_context.js`'s, which renders whole render quanta
    when asked, and the page's timers run on Playwright's clock, which the stage
    advances by the time the quanta took. Waiting is for what the page does without
    time (a body that arrives, a promise that settles), never for page time: a wait
    on a timer does not end.
    """

    stepping: Stepping = field(default_factory=Stepping)

    async def open(self, **query: str) -> None:
        """Load the app's page under the stepped context and the paused clock."""
        await self.page.add_init_script(path=STEPPED_CONTEXT)
        await self.page.clock.install(time=0)
        # Paused from the start: a timer fires only as the stage runs the clock.
        await self.page.clock.pause_at(1)
        await super().open(**query)
        seen = await self.page.evaluate("window.playerPage.snapshot()")
        self.stepping.sample_rate = seen["sampleRate"]
        self.stepping.quantum_frames = seen["renderQuantumFrames"]

    def quanta_for(self, seconds: float) -> int:
        """The whole render quanta that cover `seconds` of the context's clock."""
        return math.ceil(
            seconds * self.stepping.sample_rate / self.stepping.quantum_frames
        )

    async def step(self, quanta: int) -> int:
        """Render `quanta` render quanta, then run the page's timers over their time.

        Returns the frames the context has rendered so far. The page ends its step
        once the ear's reports have arrived, whatever they say, so a repeated or
        missing report shows as a marker and never stalls the step; a page that does
        not answer in `WAIT_SECONDS` fails it.
        """
        reply = await asyncio.wait_for(
            self.page.evaluate("quanta => window.playerPage.step(quanta)", quanta),
            WAIT_SECONDS,
        )
        frames, rate = reply["frames"], reply["sampleRate"]
        # The clock takes whole milliseconds, so it lags the audio by less than one.
        milliseconds = frames * 1000 // rate
        await self.page.clock.run_for(milliseconds - self.stepping.clock_milliseconds)
        self.stepping.clock_milliseconds = milliseconds
        return frames

    async def step_until(
        self, condition: str, *, within_seconds: float, quanta: int = 1
    ) -> None:
        """Step `quanta` render quanta at a time until `condition` holds on `s`.

        Fails when it does not hold after `within_seconds` of the context's clock.
        """
        for _ in range(math.ceil(self.quanta_for(within_seconds) / quanta)):
            if await self.holds(condition):
                return
            await self.step(quanta)
        assert await self.holds(condition), (
            f"not within {within_seconds} s: {condition}"
        )

    async def step_for(self, seconds: float) -> int:
        """Render the quanta that cover `seconds` of the context's clock."""
        return await self.step(self.quanta_for(seconds))

    async def stand_still_for(self, seconds: float) -> None:
        """Step `seconds` and fail if the context's clock moved: it is held.

        A paused or closed context renders nothing, whatever the stage steps.
        """
        before = await self.page.evaluate("window.playerPage.snapshot().contextNow")
        frames = await self.step_for(seconds)
        assert frames == round(before * self.stepping.sample_rate), (
            f"the context ran on from {before} s to {frames} frames"
        )

    async def holds(self, condition: str) -> bool:
        """Whether `condition` holds on `s`, what the page saw, now."""
        return await self.page.evaluate(_on_snapshot(condition))

    async def hear(
        self, *sounds: Mapping[str, str | int], for_seconds: float = HEARD_FOR_SECONDS
    ) -> None:
        """Hand the sounds over, step until the first is heard, then on a while."""
        await self.hand_over(*sounds)
        await self.until(STARTED)
        await self.step_until(HEARD, within_seconds=EDGE_SECONDS)
        await self.step_for(for_seconds)

    async def hear_then_release(
        self, *sounds: Mapping[str, str | int]
    ) -> dict[str, object]:
        """Hear the sounds, then let the app's held body go; the page's view before."""
        await self.hear(*sounds)
        before = await self.snapshot()
        self.side.held_release.set()
        return before

    async def hand_over_decoded(self, route: str) -> int:
        """Hand over the sound of the app's `route` as bytes the page decodes.

        The decode is the page's own work and needs no page time, so the player holds
        the decoded sound when the call returns, wherever the stage steps next.
        """
        decoded = await self.page.evaluate(
            "window.playerPage.snapshot().decodesSettled"
        )
        await self.prepare_bytes(route, route)
        [sound] = await self.hand_over(bytes_of(route))
        await self.until(f"s.decodesSettled > {decoded}")
        return sound

    async def paused(self) -> None:
        """Throw the app's pause and step until the context is suspended."""
        await self.act("pause")
        await self.step_until(SUSPENDED, within_seconds=EDGE_SECONDS)

    async def finish(self) -> dict[str, object]:
        """Step until every sound has ended, then what the page saw."""
        await self.step_until(
            ALL_ENDED, within_seconds=FINISH_WITHIN_SECONDS, quanta=FINISH_QUANTA
        )
        return await self.snapshot()


def ends(seen: dict[str, object]) -> list[str]:
    return [sound["endedValue"] for sound in seen["sounds"]]


def starts(seen: dict[str, object]) -> list[str]:
    return [sound["startedValue"] for sound in seen["sounds"]]


def events(seen: dict[str, object]) -> list[tuple[str, int]]:
    return [(entry["event"], entry["id"]) for entry in seen["log"]]


def quantum_seconds(seen: dict[str, object]) -> float:
    return seen["renderQuantumFrames"] / seen["sampleRate"]


def frames_of(seen: dict[str, object], seconds: float) -> int:
    """The frames of the context's clock that make `seconds`."""
    return round(seconds * seen["sampleRate"])


def peaks_between(seen: dict[str, object], since: float, until: float) -> list[float]:
    """The peaks of the quanta that start in [`since`, `until`) of the context clock."""
    return [peak for at, peak in seen["heardPeaks"] if since <= at < until]


def loudest_peak(seen: dict[str, object], since: float) -> float:
    """The loudest quantum the ear reported from `since` of the context clock on."""
    return max(peaks_between(seen, since, math.inf), default=0.0)


def audible_after_edge(seen: dict[str, object], name: str) -> float:
    """The seconds the output was audible after the edge of the first switch `name`."""
    [at, *_] = action_times(seen, name)
    return audible_seconds(seen, at + EDGE_SECONDS, math.inf)


def peaks_wholly_between(
    seen: dict[str, object], since: float, until: float
) -> list[float]:
    """The peaks of the quanta that lie wholly inside [`since`, `until`] of the clock.

    A quantum that straddles a bound mixes what lies on either side of it, so it
    belongs to neither window.
    """
    quantum = seen["renderQuantumFrames"]
    first, last = frames_of(seen, since), frames_of(seen, until)
    return [
        peak
        for at, peak in seen["heardPeaks"]
        if first <= frames_of(seen, at) and frames_of(seen, at) + quantum <= last
    ]


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


def onset_after_start_frames(seen: dict[str, object], sound: int = 0) -> int:
    """The frames from sound number `sound`'s announced start to its first heard one."""
    return frames_of(seen, heard_start_of(seen, sound)) - frames_of(
        seen, start_of(seen, sound)
    )


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


def reversal_peaks(seen: dict[str, object], first: str, second: str) -> list[float]:
    """The peaks from the last throw of `first` to the end of `second`'s edge.

    A `second` thrown while `first`'s ramp still runs turns the volume at a peak
    between silent and full. The ear heard that turn on the context's clock, whatever
    the page's own thread did meanwhile.
    """
    [*_, began] = action_times(seen, first)
    [*_, turned] = action_times(seen, second)
    return peaks_between(seen, began - quantum_seconds(seen), turned + EDGE_SECONDS)


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


def markers(seen: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    """What the recording shows of the four ways a cell's timing goes wrong.

    The ear's quantum reports with a gap or a repeat, each sound's heard start
    against its announced one, and each pause's time to the `suspended` state.
    They describe the recording and decide nothing.
    """
    return {
        "quantumReports": quantum_report_faults(seen),
        "heardStarts": heard_starts(seen),
        "pauses": pause_suspends(seen),
    }


def quantum_report_faults(seen: dict[str, object]) -> list[dict[str, object]]:
    """Where the ear's reports skip quanta or repeat one, with the clock times."""
    quantum = quantum_seconds(seen)
    faults: list[dict[str, object]] = []
    for (before, _), (after, _) in pairwise(seen["heardPeaks"]):
        steps = round((after - before) / quantum)
        if steps == 0:
            faults.append({"kind": "duplicate", "at": after})
        elif steps > 1:
            faults.append(
                {"kind": "gap", "after": before, "before": after, "missing": steps - 1}
            )
    return faults


def heard_starts(seen: dict[str, object]) -> list[dict[str, object]]:
    """Each started sound's announced start and the first onset heard from it on."""
    quantum = quantum_seconds(seen)
    starts = []
    for sound in seen["sounds"]:
        if sound["start"] is None:
            continue
        announced = sound["start"]["contextTime"]
        heard = next(
            (onset for onset in seen["heardOnsets"] if onset >= announced - quantum),
            None,
        )
        late = None if heard is None else heard - announced
        starts.append(
            {
                "sound": sound["id"],
                "announced": announced,
                "heard": heard,
                "lateSeconds": late,
            }
        )
    return starts


def pause_suspends(seen: dict[str, object]) -> list[dict[str, object]]:
    """For each pause, its context time and the first `suspended` state after it."""
    suspended = [
        change["at"]
        for change in seen["stateChanges"]
        if change["state"] == "suspended"
    ]
    pauses = []
    for paused_at in action_times(seen, "pause"):
        at = next((when for when in suspended if when >= paused_at), None)
        seconds = None if at is None else at - paused_at
        pauses.append(
            {"pausedAt": paused_at, "suspendedAt": at, "suspendSeconds": seconds}
        )
    return pauses


@asynccontextmanager
async def _staged[StageT: Stage](
    tmp_path: Path, stage_type: type[StageT]
) -> AsyncIterator[StageT]:
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
            directory = os.environ.get(FAILURE_DIRECTORY_VARIABLE)
            recording = Recording(
                Path(directory) if directory else tmp_path.parent / "browser-failures"
            )
            _running[:] = [recording]
            stage = stage_type(side, page, base_url, recording)
            try:
                yield stage
            except BaseException:
                await recording.take_if_none(page)
                raise


async def _listening(stage: Stage) -> None:
    """Load the app's page, click once and wait until the page's ear listens."""
    await stage.open()
    await stage.activate()
    await stage.page.evaluate("window.playerPage.listening()")


@asynccontextmanager
async def unactivated(tmp_path: Path) -> AsyncIterator[Stage]:
    async with _staged(tmp_path, Stage) as stage:
        yield stage


@asynccontextmanager
async def activated(tmp_path: Path) -> AsyncIterator[Stage]:
    async with unactivated(tmp_path) as stage:
        await _listening(stage)
        yield stage


async def stepped_through(
    tmp_path: Path, *steps: Callable[[SteppedStage], Awaitable[object]]
) -> dict[str, object]:
    """Each of the `steps` taken in turn on a stepped stage, then what the page saw."""
    async with stepped(tmp_path) as stage:
        for step in steps:
            await step(stage)
        return await stage.snapshot()


@asynccontextmanager
async def stepped(tmp_path: Path) -> AsyncIterator[SteppedStage]:
    """An activated stage on the stepped context, its ear listening."""
    async with _staged(tmp_path, SteppedStage) as stage:
        await _listening(stage)
        yield stage
