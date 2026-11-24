"""An app's browser pauses, resumes and stops its sounds with a click-free edge.

The flows run in Chromium on the stand-in app's page; see the shared stage.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import TYPE_CHECKING

import pytest

from tests.support.browser_player_support import (
    AUDIBLE_RMS,
    CLICK_STEP,
    CLOSED,
    EDGE_QUANTA,
    EDGE_SECONDS,
    FULL_PEAK,
    HEADED,
    LONG,
    LOST_AT_EDGES_SECONDS,
    MARKER,
    MARKER_SECONDS,
    MARKER_STREAMED,
    MIN_QUANTA_CHECKED,
    ONSET_BOUND_FRAMES,
    OTHER,
    RESUMES_FOR_START_AND_RESUME,
    SHORT,
    SILENT_PEAK,
    SOUND_SECONDS,
    STOP_BOUND_MS,
    STREAMED,
    TONE_PEAK,
    action_times,
    activated,
    audible_seconds,
    dropped_seconds,
    edge_peaks,
    ends,
    fed,
    heard_start_of,
    largest_step,
    peaks_between,
    peaks_inside,
    peaks_outside_edges,
    quantum_seconds,
    ramp_end,
    reversal_peaks,
    start_of,
    starts,
    tone_quanta,
    unactivated,
    url,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from tests.support.browser_player_support import Stage

pytestmark = pytest.mark.browser


AUDIBLE_CONDITION = f"s.sounds[0].peakRms > {AUDIBLE_RMS}"
SUSPENDED = "s.contextStates[0] === 'suspended'"
# The streamed sound's first bytes have played out and its body has not come.
OUTPUT_SILENT = (
    f"s.heardPeaks.some(([, peak]) => peak > {SILENT_PEAK})"
    " && s.heardPeaks.at(-1)[1] === 0"
)
HEARD_AGAIN = "s.heardOnsets.length > 1"
# A stop ends a sound at once, with no ramp to prove, when none of its chunks is
# scheduled any more: a first chunk this long outlasts the stalls of the page
# that loaded runs (load 40 to 59) showed, the longest above 0.37 s.
LEAD_SECONDS = 3.0


async def audible_handed(stage: Stage, sound: Mapping[str, str | int]) -> None:
    await stage.hand_over(sound)
    await stage.until(AUDIBLE_CONDITION)


async def audible(stage: Stage, sound: str) -> None:
    await audible_handed(stage, url(sound))


async def short_sound_played(stage: Stage) -> dict[str, object]:
    await stage.hand_over(url(SHORT))
    return await stage.finish()


async def paused(stage: Stage) -> None:
    await stage.act("pause")
    await stage.until(SUSPENDED)


@pytest.mark.parametrize("sound", [MARKER, MARKER_STREAMED], ids=["whole", "streamed"])
def test_pause_ramps_the_output_down_without_a_click_and_holds_the_sound(
    tmp_path: Path, sound: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await audible(stage, sound)
            await paused(stage)
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    [paused_at] = action_times(seen, "pause")
    edge = edge_peaks(seen, "pause")
    assert largest_step(edge) < CLICK_STEP
    ramping = [peak for peak in edge if SILENT_PEAK < peak < FULL_PEAK]
    assert len(ramping) >= EDGE_QUANTA
    assert edge[-1] < SILENT_PEAK
    assert audible_seconds(seen, paused_at + EDGE_SECONDS, math.inf) == 0
    assert seen["contextStates"] == ["suspended"]
    assert ends(seen) == [None]


def test_the_record_of_a_failed_cell_marks_gaps_repeats_late_starts_and_suspends(
    tmp_path: Path,
) -> None:
    gap_after, gap_before, last = 9, 14, 29

    async def scenario() -> tuple[dict[str, object], Path]:
        async with activated(tmp_path) as stage:
            await audible(stage, MARKER)
            await paused(stage)
            seen = await stage.snapshot()
            step = quantum_seconds(seen)
            reports = [
                [index * step, 0.0]
                for index in range(last + 1)
                if not gap_after < index < gap_before
            ]
            seen["heardPeaks"] = [*reports, reports[-1]]
            stage.recording.seen = seen
            stage.recording.directory = tmp_path / "records"
            return seen, stage.recording.write("tests/test_x.py::test_y[case]")

    seen, path = asyncio.run(scenario())
    record = json.loads(path.read_text())
    step = quantum_seconds(seen)

    assert path.name == "tests_test_x.py_test_y_case_.json"
    assert record["cell"] == "tests/test_x.py::test_y[case]"
    assert record["snapshot"]["sampleRate"] == seen["sampleRate"]
    assert record["markers"]["quantumReports"] == [
        {
            "kind": "gap",
            "after": gap_after * step,
            "before": gap_before * step,
            "missing": gap_before - gap_after - 1,
        },
        {"kind": "duplicate", "at": last * step},
    ]
    [started] = record["markers"]["heardStarts"]
    assert started["announced"] == start_of(seen)
    assert started["heard"] == heard_start_of(seen)
    [pause] = record["markers"]["pauses"]
    assert pause["pausedAt"] == action_times(seen, "pause")[0]
    assert pause["suspendSeconds"] > 0


@pytest.mark.parametrize("sound", [MARKER, MARKER_STREAMED], ids=["whole", "streamed"])
def test_resume_plays_on_from_where_it_paused_though_its_body_arrived_meanwhile(
    tmp_path: Path, sound: str
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await audible(stage, sound)
            await paused(stage)
            stage.side.held_release.set()
            await stage.settle()
            while_paused = await stage.snapshot()
            await stage.act("resume")
            return while_paused, await stage.finish()

    while_paused, seen = asyncio.run(scenario())

    tone, gap, tail = MARKER_SECONDS
    start = heard_start_of(seen)
    [paused_at] = action_times(seen, "pause")
    assert audible_seconds(while_paused, paused_at + EDGE_SECONDS, math.inf) == 0
    assert ends(while_paused) == [None]
    assert ends(seen) == ["played"]
    # The tone loses only the edges' slack; the gap and the tail keep their place.
    assert audible_seconds(seen, start, start + tone) == pytest.approx(
        tone, abs=LOST_AT_EDGES_SECONDS
    )
    assert audible_seconds(seen, start + tone + gap, math.inf) == pytest.approx(
        tail, abs=quantum_seconds(seen)
    )
    tail_onset = seen["heardOnsets"][-1]
    # The output's render thread may skip quanta of the context's clock; the sound
    # waits for them, so the tail starts later by what the ear never rendered.
    assert tail_onset == pytest.approx(
        start + tone + gap + dropped_seconds(seen, start, tail_onset),
        abs=ONSET_BOUND_FRAMES / seen["sampleRate"],
    )
    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(
        tone + gap + tail, abs=seen["clockStep"]
    )


def test_resume_whose_promise_comes_late_plays_the_sound_on_at_full_volume_meanwhile(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await audible(stage, MARKER)
            await paused(stage)
            await stage.act("hold", "resume")
            await stage.act("resume")
            # The context renders again while the page still waits for the promise.
            await stage.until("s.contextStates[0] === 'running'")
            await stage.settle()
            await stage.settle()
            await stage.act("release")
            return await stage.finish()

    seen = asyncio.run(scenario())

    tone, _, _ = MARKER_SECONDS
    start = start_of(seen)
    assert ends(seen) == ["played"]
    # The wait for the promise is longer than the edges' slack, so a volume that ramps
    # up only when the promise comes leaves the tone short by the wait.
    assert audible_seconds(seen, start, start + tone) == pytest.approx(
        tone, abs=LOST_AT_EDGES_SECONDS
    )


def test_a_sound_handed_over_while_paused_waits_and_plays_after_resume(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.act("pause")
            await stage.hand_over(url(SHORT))
            await stage.settle()
            while_paused = await stage.snapshot()
            await stage.act("resume")
            return while_paused, await stage.finish()

    while_paused, seen = asyncio.run(scenario())

    assert starts(while_paused) == [None]
    assert ends(while_paused) == [None]
    assert while_paused["sourcesStarted"] == 0
    assert ends(seen) == ["played"]
    assert seen["sounds"][0]["peakRms"] > AUDIBLE_RMS


@pytest.mark.parametrize("then", ["resume", "stop"])
def test_a_streamed_sound_whose_first_samples_arrive_while_paused_starts_only_by_resume(
    tmp_path: Path, then: str
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(HEADED))
            await stage.settle()
            await paused(stage)
            stage.side.held_release.set()
            await stage.until(f"s.sourcesStarted > 0 && {SUSPENDED}")
            await stage.settle()
            while_paused = await stage.snapshot()
            await stage.act(then)
            await stage.until("s.sounds[0].endedValue !== null")
            return while_paused, await stage.snapshot()

    while_paused, seen = asyncio.run(scenario())

    assert starts(while_paused) == [None]
    assert ends(while_paused) == [None]
    if then == "resume":
        assert (starts(seen), ends(seen)) == (["playing"], ["played"])
    else:
        assert (starts(seen), ends(seen)) == (["ended"], ["stopped"])


@pytest.mark.parametrize("then", ["stop", "mute", "close"])
def test_a_streamed_sound_first_heard_in_the_pause_ramp_reports_its_start_and_playing(
    tmp_path: Path, then: str
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            # The sound outlasts the context's run on to the suspend.
            await stage.prepare_bytes("long", LONG)
            await stage.hand_over(fed("long"))
            await stage.settle()
            await stage.act("freezeClock")
            await stage.act("feedWhile", "long", "pause")
            await stage.until("s.sounds[0].startedValue !== null")
            await stage.act("thawClock")
            await stage.until(SUSPENDED)
            heard = await stage.snapshot()
            if then == "mute":
                await stage.set_muted(muted=True)
            else:
                await stage.act(then)
            await stage.until("s.sounds[0].endedValue !== null")
            return heard, await stage.snapshot()

    heard, seen = asyncio.run(scenario())

    [paused_at] = action_times(seen, "pause+feed")
    # The context clock stands still from before the pause until the start is
    # reported, so the chunk is scheduled inside the pause's ramp whatever the
    # page's timing.
    first_heard = heard["sounds"][0]["start"]
    assert first_heard is not None
    assert paused_at <= first_heard["contextTime"] < paused_at + EDGE_SECONDS
    assert starts(seen) == ["playing"]
    assert ends(seen) == ["muted" if then == "mute" else "stopped"]
    assert seen["sounds"][0]["playedSeconds"] > 0


def test_pause_and_resume_on_an_idle_player_report_and_start_nothing(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with unactivated(tmp_path) as stage:
            await stage.open()
            await stage.act("pause")
            await stage.settle()
            await stage.act("resume")
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert seen["log"] == []
    assert seen["sourcesStarted"] == 0


@pytest.mark.parametrize(
    ("sound", "body_released", "starved"),
    [
        (url(LONG), False, False),
        (fed("long", lead_seconds=LEAD_SECONDS), False, False),
        (url(STREAMED), True, False),
        (url(STREAMED), True, True),
    ],
    ids=[
        "whole",
        "streamed",
        "streamed-body-scheduled",
        "streamed-body-scheduled-after-a-gap",
    ],
)
def test_stop_ramps_down_without_a_click_then_ends_and_the_next_sound_is_at_full_volume(
    tmp_path: Path,
    sound: Mapping[str, str | int],
    *,
    body_released: bool,
    starved: bool,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            if "fed" in sound:
                await stage.prepare_bytes("long", LONG)
            await audible_handed(stage, sound)
            if starved:
                await stage.until(OUTPUT_SILENT)
            if body_released:
                stage.side.held_release.set()
                await (stage.until(HEARD_AGAIN) if starved else stage.settle())
            await stage.act("stop")
            at_stop = await stage.snapshot()
            return at_stop, await short_sound_played(stage)

    at_stop, seen = asyncio.run(scenario())

    assert at_stop["endedAtAction"] == [None]
    assert ends(seen) == ["stopped", "played"]
    # Not the played seconds: a sound silent for want of its body played fewer.
    edge_end = ramp_end(seen, 0)
    next_heard = heard_start_of(seen, -1)
    margin = 3 * quantum_seconds(seen)
    edge = edge_peaks(seen, "stop", until=edge_end)
    # The next sound may start where the edge ends, with no quantum of silence between.
    silence = peaks_between(seen, edge_end, next_heard - margin)
    assert seen["sounds"][0]["endedAt"] - seen["lastActionAt"] < STOP_BOUND_MS
    assert largest_step(edge) < CLICK_STEP
    assert edge[-1] < SILENT_PEAK
    assert max(silence, default=0) < SILENT_PEAK
    heard = peaks_between(
        seen, next_heard + margin, next_heard + SOUND_SECONDS["short.wav"] - margin
    )
    assert heard
    assert min(heard) >= FULL_PEAK


def test_a_sound_handed_over_as_the_stopped_one_ends_early_starts_at_full_volume(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.act("stopOnNextSourceEnd", url(SHORT))
            await stage.hand_over(url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert seen["instrumentStops"] == 1
    assert ends(seen) == ["stopped", "played"]
    heard = peaks_inside(seen, 1, SOUND_SECONDS["short.wav"])
    assert heard
    assert min(heard) >= FULL_PEAK


def test_stop_while_paused_ends_every_sound_at_once_and_a_later_one_waits_for_resume(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG), url(SHORT))
            await stage.until(AUDIBLE_CONDITION)
            await paused(stage)
            await stage.act("stop")
            at_stop = (await stage.snapshot())["endedAtAction"]
            await stage.hand_over(url(OTHER))
            await stage.settle()
            while_paused = await stage.snapshot()
            await stage.act("resume")
            return {**while_paused, "atStop": at_stop}, await stage.finish()

    while_paused, seen = asyncio.run(scenario())

    assert while_paused["atStop"] == ["stopped", "stopped"]
    assert starts(while_paused)[2] is None
    assert ends(seen) == ["stopped", "stopped", "played"]


def test_stop_while_the_pause_still_ramps_ends_the_sound_after_that_edge_and_holds(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await audible(stage, LONG)
            await stage.act("inOneTask", "pause", "stop")
            await stage.until("s.sounds[0].endedValue !== null")
            await stage.until(SUSPENDED)
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    edge = edge_peaks(seen, "pause+stop")
    assert ends(seen) == ["stopped"]
    assert largest_step(edge) < CLICK_STEP
    assert edge[-1] < SILENT_PEAK
    [at] = action_times(seen, "pause+stop")
    assert audible_seconds(seen, at + EDGE_SECONDS, math.inf) == 0


def test_resume_while_the_pause_is_still_ramping_plays_on_without_a_click(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await audible(stage, MARKER)
            await stage.act("reverse", "pause", "resume")
            return await stage.finish()

    seen = asyncio.run(scenario())

    start = heard_start_of(seen)
    # The pause set a ramp to silence and the resume came before that ramp's end, on the
    # context's clock; the ear heard no silence between them.
    [turned] = action_times(seen, "resume")
    assert turned < ramp_end(seen, 0)
    assert min(reversal_peaks(seen, "pause", "resume")) > SILENT_PEAK
    assert seen["contextStates"] == ["running"]
    assert ends(seen) == ["played"]
    assert largest_step([peak for _, peak in tone_quanta(seen, start)]) < CLICK_STEP
    assert min(peaks_outside_edges(seen, start)) >= FULL_PEAK


def test_pause_while_the_resume_is_still_ramping_up_holds_the_sound_without_a_click(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await audible(stage, MARKER)
            await paused(stage)
            await stage.act("reverse", "resume", "pause")
            await stage.until(SUSPENDED)
            await stage.settle()
            held = await stage.snapshot()
            await stage.act("resume")
            return held, await stage.finish()

    held, seen = asyncio.run(scenario())

    start = heard_start_of(seen)
    [*_, last_pause] = action_times(held, "pause")
    assert SILENT_PEAK < max(reversal_peaks(held, "resume", "pause")) < FULL_PEAK
    assert ends(held) == [None]
    assert audible_seconds(held, last_pause + EDGE_SECONDS, math.inf) == 0
    assert ends(seen) == ["played"]
    assert largest_step([peak for _, peak in tone_quanta(seen, start)]) < CLICK_STEP
    assert min(peaks_outside_edges(seen, start)) >= FULL_PEAK


def test_resume_after_stop_changes_nothing_and_the_next_sound_plays_at_full_volume(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await audible(stage, LONG)
            await stage.act("stop")
            await stage.act("resume")
            return await short_sound_played(stage)

    seen = asyncio.run(scenario())

    assert ends(seen) == ["stopped", "played"]
    assert min(peaks_inside(seen, 1, SOUND_SECONDS["short.wav"])) >= FULL_PEAK


def test_pause_twice_and_resume_twice_act_once(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await audible(stage, MARKER)
            await stage.act("pause")
            await paused(stage)
            await stage.act("resume")
            await stage.act("resume")
            return await stage.finish()

    seen = asyncio.run(scenario())

    start = start_of(seen)
    assert ends(seen) == ["played"]
    assert seen["resumesCalled"] == RESUMES_FOR_START_AND_RESUME
    assert min(peaks_outside_edges(seen, start)) >= FULL_PEAK


def test_muting_while_paused_ends_the_sound_at_once_and_clears_the_pause(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await audible(stage, LONG)
            await paused(stage)
            await stage.set_muted(muted=True)
            at_mute = await stage.snapshot()
            await stage.set_muted(muted=False)
            return at_mute, await short_sound_played(stage)

    at_mute, seen = asyncio.run(scenario())

    assert at_mute["endedAtAction"] == ["muted"]
    assert ends(seen) == ["muted", "played"]
    assert seen["gainValue"] == 1
    assert seen["sounds"][1]["peakRms"] > AUDIBLE_RMS


def test_close_while_paused_ends_the_sound_at_once_and_closes_the_context(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await audible(stage, LONG)
            await paused(stage)
            await stage.act("close")
            at_close = (await stage.snapshot())["endedAtAction"]
            await stage.until(f"s.contextStates.every(state => state === '{CLOSED}')")
            await stage.settle()
            return {**await stage.snapshot(), "atClose": at_close}

    seen = asyncio.run(scenario())

    assert seen["atClose"] == ["stopped"]
    assert ends(seen) == ["stopped"]
    assert seen["contextStates"] == [CLOSED]


def test_the_volume_changes_in_no_way_but_at_the_edges_of_a_pause_and_a_resume(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await audible(stage, MARKER)
            await paused(stage)
            await stage.act("resume")
            return await stage.finish()

    seen = asyncio.run(scenario())

    outside = peaks_outside_edges(seen, start_of(seen))
    assert len(outside) > MIN_QUANTA_CHECKED
    assert min(outside) >= FULL_PEAK
    assert max(outside) <= 1.05 * TONE_PEAK
