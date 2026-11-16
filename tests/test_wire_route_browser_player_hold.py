"""An app's browser pauses, resumes and stops its sounds with a click-free edge.

The flows run in Chromium on the stand-in app's page; see the shared stage.
"""

from __future__ import annotations

import asyncio
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
    action_gains,
    action_times,
    activated,
    audible_seconds,
    edge_peaks,
    ends,
    fed,
    largest_step,
    peaks_inside,
    peaks_outside_edges,
    quantum_seconds,
    start_of,
    starts,
    tone_quanta,
    unactivated,
    url,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.browser


AUDIBLE_CONDITION = f"s.sounds[0].peakRms > {AUDIBLE_RMS}"
SUSPENDED = "s.contextStates[0] === 'suspended'"


@pytest.mark.parametrize("sound", [MARKER, MARKER_STREAMED], ids=["whole", "streamed"])
def test_pause_ramps_the_output_down_without_a_click_and_holds_the_sound(
    tmp_path: Path, sound: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(sound))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.until(SUSPENDED)
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


@pytest.mark.parametrize("sound", [MARKER, MARKER_STREAMED], ids=["whole", "streamed"])
def test_resume_plays_on_from_where_it_paused_though_its_body_arrived_meanwhile(
    tmp_path: Path, sound: str
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(sound))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.until(SUSPENDED)
            stage.side.held_release.set()
            await stage.settle()
            while_paused = await stage.snapshot()
            await stage.act("resume")
            return while_paused, await stage.finish()

    while_paused, seen = asyncio.run(scenario())

    tone, gap, tail = MARKER_SECONDS
    start = start_of(seen)
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
    assert seen["heardOnsets"][-1] == pytest.approx(
        start + tone + gap, abs=ONSET_BOUND_FRAMES / seen["sampleRate"]
    )
    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(
        tone + gap + tail, abs=seen["clockStep"]
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
            await stage.act("pause")
            await stage.until(SUSPENDED)
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
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.prepare_bytes("short", SHORT)
            await stage.hand_over(fed("short"))
            await stage.settle()
            await stage.act("feedWhile", "short", "pause")
            await stage.until(SUSPENDED)
            if then == "mute":
                await stage.set_muted(muted=True)
            else:
                await stage.act(then)
            await stage.until("s.sounds[0].endedValue !== null")
            return await stage.snapshot()

    seen = asyncio.run(scenario())

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
    ("sound", "body_released"),
    [(LONG, False), (STREAMED, False), (STREAMED, True)],
    ids=["whole", "streamed", "streamed-body-scheduled"],
)
def test_stop_ramps_down_without_a_click_then_ends_and_the_next_sound_is_at_full_volume(
    tmp_path: Path, sound: str, *, body_released: bool
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(sound))
            await stage.until(AUDIBLE_CONDITION)
            if body_released:
                stage.side.held_release.set()
                await stage.settle()
            await stage.act("stop")
            at_stop = await stage.snapshot()
            await stage.hand_over(url(SHORT))
            return at_stop, await stage.finish()

    at_stop, seen = asyncio.run(scenario())

    assert at_stop["endedAtAction"] == [None]
    assert ends(seen) == ["stopped", "played"]
    # The next start is known to within a quantum; its first quanta are no edge.
    next_start = start_of(seen, 1) - 3 * quantum_seconds(seen)
    edge = edge_peaks(seen, "stop", until=next_start)
    assert seen["sounds"][0]["endedAt"] - seen["lastActionAt"] < STOP_BOUND_MS
    assert largest_step(edge) < CLICK_STEP
    assert edge[-1] < SILENT_PEAK
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
            await stage.act("pause")
            await stage.until(SUSPENDED)
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
            await stage.hand_over(url(LONG))
            await stage.until(AUDIBLE_CONDITION)
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
            await stage.hand_over(url(MARKER))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("reverse", "pause", "resume")
            return await stage.finish()

    seen = asyncio.run(scenario())

    start = start_of(seen)
    [gain_at_resume] = action_gains(seen, "resume")
    assert 0 < gain_at_resume < 1
    assert seen["contextStates"] == ["running"]
    assert ends(seen) == ["played"]
    assert largest_step([peak for _, peak in tone_quanta(seen, start)]) < CLICK_STEP
    assert min(peaks_outside_edges(seen, start)) >= FULL_PEAK


def test_pause_while_the_resume_is_still_ramping_up_holds_the_sound_without_a_click(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(MARKER))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.until(SUSPENDED)
            await stage.act("reverse", "resume", "pause")
            await stage.until(SUSPENDED)
            await stage.settle()
            held = await stage.snapshot()
            await stage.act("resume")
            return held, await stage.finish()

    held, seen = asyncio.run(scenario())

    start = start_of(seen)
    [*_, last_pause] = action_times(held, "pause")
    [*_, gain_at_pause] = action_gains(held, "pause")
    assert 0 < gain_at_pause < 1
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
            await stage.hand_over(url(LONG))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("stop")
            await stage.act("resume")
            await stage.hand_over(url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["stopped", "played"]
    assert min(peaks_inside(seen, 1, SOUND_SECONDS["short.wav"])) >= FULL_PEAK


def test_pause_twice_and_resume_twice_act_once(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(MARKER))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.act("pause")
            await stage.until(SUSPENDED)
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
            await stage.hand_over(url(LONG))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.until(SUSPENDED)
            await stage.set_muted(muted=True)
            at_mute = await stage.snapshot()
            await stage.set_muted(muted=False)
            await stage.hand_over(url(SHORT))
            return at_mute, await stage.finish()

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
            await stage.hand_over(url(LONG))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.until(SUSPENDED)
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
            await stage.hand_over(url(MARKER))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.until(SUSPENDED)
            await stage.act("resume")
            return await stage.finish()

    seen = asyncio.run(scenario())

    outside = peaks_outside_edges(seen, start_of(seen))
    assert len(outside) > MIN_QUANTA_CHECKED
    assert min(outside) >= FULL_PEAK
    assert max(outside) <= 1.05 * TONE_PEAK
