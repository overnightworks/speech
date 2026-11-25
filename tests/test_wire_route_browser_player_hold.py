"""An app's browser pauses, resumes and stops its sounds with a click-free edge.

The flows run in Chromium on the stand-in app's page, on a context the test steps;
see the shared stage.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import TYPE_CHECKING

import pytest

from tests.support.browser_player_support import (
    CLICK_STEP,
    CLOSED,
    EDGE_QUANTA,
    EDGE_SECONDS,
    FULL_PEAK,
    HEADED,
    LONG,
    MARKER,
    MARKER_SECONDS,
    MARKER_STREAMED,
    MIN_QUANTA_CHECKED,
    OTHER,
    RESUMES_FOR_START_AND_RESUME,
    SHORT,
    SILENT_PEAK,
    SOUND_SECONDS,
    STARTED,
    STREAMED,
    STRETCH_SECONDS,
    SUSPENDED,
    TONE_PEAK,
    SteppedStage,
    action_times,
    audible_seconds,
    bytes_of,
    edge_peaks,
    ends,
    fed,
    frames_of,
    heard_start_of,
    largest_step,
    peaks_between,
    peaks_inside,
    peaks_outside_edges,
    peaks_wholly_between,
    quantum_seconds,
    ramp_end,
    reversal_peaks,
    start_of,
    starts,
    stepped,
    tone_quanta,
    url,
)
from tests.support.fakes_support import SPEAK_SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from pathlib import Path

pytestmark = pytest.mark.browser


def scheduled_seconds(seconds: float) -> str:
    """A condition: a sound `seconds` long has its whole body with the output."""
    return (
        "s.scheduledChunks.reduce((total, chunk) => total + chunk.seconds, 0)"
        f" > {seconds - 1 / SPEAK_SAMPLE_RATE}"
    )


async def heard_then(
    tmp_path: Path,
    sound: Mapping[str, str | int],
    *steps: Callable[[SteppedStage], Awaitable[object]],
    report: Callable[[SteppedStage], Awaitable[dict[str, object]]],
) -> dict[str, object]:
    """The sound heard, each of the `steps` taken in turn, then what `report` gives."""
    async with stepped(tmp_path) as stage:
        await stage.hear(sound)
        for step in steps:
            await step(stage)
        return await report(stage)


def switch(
    action: str, *arguments: object
) -> Callable[[SteppedStage], Awaitable[None]]:
    """A step: the app's switch `action`."""
    return lambda stage: stage.act(action, *arguments)


# What of the tone the output plays before the pause, which lies well inside it.
PLAYED_BEFORE_PAUSE_SECONDS = 0.5
# A switch thrown this many quanta after the one before turns the volume mid-edge.
INTO_THE_EDGE_QUANTA = 4
# The streamed sound's first bytes have played out and its body has not come.
OUTPUT_SILENT = (
    f"s.heardPeaks.some(([, peak]) => peak > {SILENT_PEAK})"
    " && s.heardPeaks.at(-1)[1] === 0"
)
HEARD_AGAIN = "s.heardOnsets.length > 1"
SOUND_ENDED = "s.sounds[0].endedValue !== null"
RUNNING = "s.contextStates[0] === 'running'"
# What a streamed sound's first chunk plays out within, and a body's next one is heard.
STREAMED_WITHIN_SECONDS = 1.0
# A first chunk this long outlasts what is played before a stop and its edge.
LEAD_SECONDS = 1.0


@pytest.mark.parametrize("sound", [MARKER, MARKER_STREAMED], ids=["whole", "streamed"])
def test_pause_ramps_the_output_down_without_a_click_and_holds_the_sound(
    tmp_path: Path, sound: str
) -> None:
    seen = asyncio.run(
        heard_then(
            tmp_path,
            url(sound),
            lambda stage: stage.paused(),
            lambda stage: stage.stand_still_for(STRETCH_SECONDS),
            report=SteppedStage.snapshot,
        )
    )

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
        async with stepped(tmp_path) as stage:
            await stage.hear(url(MARKER))
            await stage.paused()
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
    assert started["announced"] + started["lateSeconds"] == pytest.approx(
        started["heard"]
    )
    assert started["lateSeconds"] >= -step
    [pause] = record["markers"]["pauses"]
    assert pause["pausedAt"] == action_times(seen, "pause")[0]
    assert pause["suspendSeconds"] > 0
    assert pause["pausedAt"] + pause["suspendSeconds"] == pytest.approx(
        pause["suspendedAt"]
    )


@pytest.mark.parametrize("sound", [MARKER, MARKER_STREAMED], ids=["whole", "streamed"])
def test_resume_plays_on_from_where_it_paused_though_its_body_arrived_meanwhile(
    tmp_path: Path, sound: str
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(sound), for_seconds=PLAYED_BEFORE_PAUSE_SECONDS)
            await stage.paused()
            stage.side.held_release.set()
            await stage.until(scheduled_seconds(sum(MARKER_SECONDS)))
            while_paused = await stage.snapshot()
            await stage.act("resume")
            return while_paused, await stage.finish()

    while_paused, seen = asyncio.run(scenario())

    tone, gap, tail = MARKER_SECONDS
    start = heard_start_of(seen)
    [paused_at] = action_times(seen, "pause")
    assert start < paused_at < start + tone
    assert audible_seconds(while_paused, paused_at + EDGE_SECONDS, math.inf) == 0
    assert ends(while_paused) == [None]
    assert ends(seen) == ["played"]
    # The tone loses only the edges; the gap and the tail keep their place.
    assert min(peaks_outside_edges(seen, start)) >= FULL_PEAK
    assert audible_seconds(seen, start + tone + gap, math.inf) == pytest.approx(
        tail, abs=quantum_seconds(seen)
    )
    # The context's clock stood still while paused, so the sound plays on from
    # the content it had played: the tail begins where the content puts it.
    assert frames_of(seen, seen["heardOnsets"][-1]) == frames_of(
        seen, start
    ) + frames_of(seen, tone + gap)
    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(tone + gap + tail)


@pytest.mark.parametrize(
    ("ending", "state"), [("suspend", "suspended"), ("close", "closed")]
)
def test_a_stepped_context_stays_as_the_call_after_its_pending_resume_left_it(
    tmp_path: Path, ending: str, state: str
) -> None:
    async def scenario() -> str:
        async with stepped(tmp_path) as stage:
            return await stage.page.evaluate(
                """async ending => {
                    const context = new AudioContext();
                    const resumed = context.resume();
                    const ended = context[ending]();
                    await Promise.all([resumed, ended]);
                    return context.state;
                }""",
                ending,
            )

    assert asyncio.run(scenario()) == state


def test_resume_whose_promise_comes_late_plays_the_sound_on_at_full_volume_meanwhile(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(MARKER))
            await stage.paused()
            await stage.act("hold", "resume")
            await stage.act("resume")
            # The context renders again while the page still waits for the promise.
            await stage.until(RUNNING)
            await stage.step_for(STRETCH_SECONDS)
            await stage.act("release")
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["played"]
    # The wait for the promise is longer than an edge, so a volume that ramps up
    # only when the promise comes leaves the tone silent beyond its edges.
    assert min(peaks_outside_edges(seen, heard_start_of(seen))) >= FULL_PEAK


def test_a_sound_handed_over_while_paused_waits_and_plays_after_resume(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.act("pause")
            await stage.hand_over_decoded(SHORT)
            await stage.stand_still_for(STRETCH_SECONDS)
            while_paused = await stage.snapshot()
            await stage.act("resume")
            return while_paused, await stage.finish()

    while_paused, seen = asyncio.run(scenario())

    assert starts(while_paused) == [None]
    assert ends(while_paused) == [None]
    assert while_paused["sourcesStarted"] == 0
    assert ends(seen) == ["played"]
    assert max(peaks_between(seen, 0, math.inf)) >= FULL_PEAK


@pytest.mark.parametrize("then", ["resume", "stop"])
def test_a_streamed_sound_whose_first_samples_arrive_while_paused_starts_only_by_resume(
    tmp_path: Path, then: str
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.hand_over(url(HEADED))
            # The sound waits for its first samples on a running context.
            await stage.until(RUNNING)
            await stage.paused()
            stage.side.held_release.set()
            await stage.until(f"s.sourcesStarted > 0 && {SUSPENDED}")
            await stage.stand_still_for(STRETCH_SECONDS)
            while_paused = await stage.snapshot()
            await stage.act(then)
            return while_paused, await stage.finish()

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
        async with stepped(tmp_path) as stage:
            # The sound outlasts the context's run on to the suspend.
            await stage.prepare_bytes("long", LONG)
            await stage.hand_over(fed("long"))
            await stage.until(RUNNING)
            await stage.act("feedWhile", "long", "pause")
            await stage.until("s.sounds[0].startedValue !== null")
            await stage.step_until(SUSPENDED, within_seconds=EDGE_SECONDS)
            heard = await stage.snapshot()
            if then == "mute":
                await stage.set_muted(muted=True)
            else:
                await stage.act(then)
            return heard, await stage.finish()

    heard, seen = asyncio.run(scenario())

    [paused_at] = action_times(seen, "pause+feed")
    first_heard = heard["sounds"][0]["start"]
    assert first_heard is not None
    assert paused_at <= first_heard["contextTime"] < ramp_end(seen, 0)
    assert starts(seen) == ["playing"]
    assert ends(seen) == ["muted" if then == "mute" else "stopped"]
    assert seen["sounds"][0]["playedSeconds"] > 0


def test_pause_and_resume_on_an_idle_player_report_and_start_nothing(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.act("pause")
            await stage.step_for(STRETCH_SECONDS)
            await stage.act("resume")
            await stage.step_for(STRETCH_SECONDS)
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
        async with stepped(tmp_path) as stage:
            if "fed" in sound:
                await stage.prepare_bytes("long", LONG)
            await stage.hear(sound)
            if starved:
                await stage.step_until(
                    OUTPUT_SILENT, within_seconds=STREAMED_WITHIN_SECONDS
                )
            if body_released:
                stage.side.held_release.set()
                await stage.until(scheduled_seconds(SOUND_SECONDS["long.wav"]))
                if starved:
                    await stage.step_until(
                        HEARD_AGAIN, within_seconds=STREAMED_WITHIN_SECONDS
                    )
            await stage.act("stop")
            at_stop = await stage.snapshot()
            await stage.hand_over_decoded(SHORT)
            await stage.step_until(SOUND_ENDED, within_seconds=EDGE_SECONDS)
            return at_stop, await stage.finish()

    at_stop, seen = asyncio.run(scenario())

    assert at_stop["endedAtAction"] == [None]
    assert ends(seen) == ["stopped", "played"]
    edge_end = ramp_end(seen, 0)
    next_heard = heard_start_of(seen, -1)
    edge = edge_peaks(seen, "stop", until=edge_end)
    # The next sound may start where the edge ends, with no quantum of silence between.
    silence = peaks_wholly_between(seen, edge_end, next_heard)
    assert largest_step(edge) < CLICK_STEP
    assert edge[-1] < SILENT_PEAK
    assert max(silence, default=0) < SILENT_PEAK
    heard = peaks_wholly_between(
        seen, next_heard, next_heard + SOUND_SECONDS["short.wav"]
    )
    assert heard
    assert min(heard) >= FULL_PEAK


def test_a_sound_handed_over_as_the_stopped_one_ends_early_starts_at_full_volume(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.prepare_bytes("next", SHORT)
            await stage.act("stopOnNextSourceEnd", bytes_of("next"))
            await stage.hand_over(url(SHORT))
            await stage.until(STARTED)
            # The sound's end hands the next one over; the page decodes it without time.
            await stage.step_until(
                "s.sounds.length > 1",
                within_seconds=SOUND_SECONDS["short.wav"] + EDGE_SECONDS,
            )
            await stage.until("s.sounds[1].startedValue !== null")
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
        async with stepped(tmp_path) as stage:
            await stage.hear(url(LONG), url(SHORT))
            await stage.paused()
            await stage.act("stop")
            at_stop = (await stage.snapshot())["endedAtAction"]
            await stage.hand_over_decoded(OTHER)
            await stage.stand_still_for(STRETCH_SECONDS)
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
        async with stepped(tmp_path) as stage:
            await stage.hear(url(LONG))
            await stage.act("inOneTask", "pause", "stop")
            await stage.step_until(
                f"{SOUND_ENDED} && {SUSPENDED}", within_seconds=EDGE_SECONDS
            )
            await stage.stand_still_for(STRETCH_SECONDS)
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
    seen = asyncio.run(
        heard_then(
            tmp_path,
            url(MARKER),
            switch("pause"),
            lambda stage: stage.step(INTO_THE_EDGE_QUANTA),
            switch("resume"),
            report=SteppedStage.finish,
        )
    )

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
        async with stepped(tmp_path) as stage:
            await stage.hear(url(MARKER))
            await stage.paused()
            await stage.act("resume")
            await stage.step(INTO_THE_EDGE_QUANTA)
            await stage.paused()
            await stage.stand_still_for(STRETCH_SECONDS)
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
    seen = asyncio.run(
        heard_then(
            tmp_path,
            url(LONG),
            switch("stop"),
            switch("resume"),
            lambda stage: stage.hand_over_decoded(SHORT),
            report=SteppedStage.finish,
        )
    )

    assert ends(seen) == ["stopped", "played"]
    assert min(peaks_inside(seen, 1, SOUND_SECONDS["short.wav"])) >= FULL_PEAK


def test_pause_twice_and_resume_twice_act_once(tmp_path: Path) -> None:
    seen = asyncio.run(
        heard_then(
            tmp_path,
            url(MARKER),
            switch("pause"),
            lambda stage: stage.paused(),
            switch("resume"),
            switch("resume"),
            report=SteppedStage.finish,
        )
    )

    start = start_of(seen)
    assert ends(seen) == ["played"]
    assert seen["resumesCalled"] == RESUMES_FOR_START_AND_RESUME
    assert min(peaks_outside_edges(seen, start)) >= FULL_PEAK


def test_muting_while_paused_ends_the_sound_at_once_and_clears_the_pause(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(LONG))
            await stage.paused()
            await stage.set_muted(muted=True)
            at_mute = await stage.snapshot()
            await stage.set_muted(muted=False)
            await stage.hand_over_decoded(SHORT)
            return at_mute, await stage.finish()

    at_mute, seen = asyncio.run(scenario())

    assert at_mute["endedAtAction"] == ["muted"]
    assert ends(seen) == ["muted", "played"]
    assert seen["gainValue"] == 1
    assert min(peaks_inside(seen, 1, SOUND_SECONDS["short.wav"])) >= FULL_PEAK


def test_close_while_paused_ends_the_sound_at_once_and_closes_the_context(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(LONG))
            await stage.paused()
            await stage.act("close")
            at_close = (await stage.snapshot())["endedAtAction"]
            await stage.until(f"s.contextStates.every(state => state === '{CLOSED}')")
            await stage.stand_still_for(STRETCH_SECONDS)
            return {**await stage.snapshot(), "atClose": at_close}

    seen = asyncio.run(scenario())

    assert seen["atClose"] == ["stopped"]
    assert ends(seen) == ["stopped"]
    assert seen["contextStates"] == [CLOSED]


def test_the_volume_changes_in_no_way_but_at_the_edges_of_a_pause_and_a_resume(
    tmp_path: Path,
) -> None:
    seen = asyncio.run(
        heard_then(
            tmp_path,
            url(MARKER),
            lambda stage: stage.paused(),
            switch("resume"),
            report=SteppedStage.finish,
        )
    )

    outside = peaks_outside_edges(seen, start_of(seen))
    assert len(outside) > MIN_QUANTA_CHECKED
    assert min(outside) >= FULL_PEAK
    assert max(outside) <= 1.05 * TONE_PEAK
