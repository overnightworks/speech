"""An app's browser holds a sound once its played position reaches one it gives.

The flows run in Chromium on the stand-in app's page, on a context the test steps: each
cell steps to the state it needs and asserts in frames from what it arranged (the
sound's announced start, the chunks the page scheduled, the volume ramps it recorded).
A cell that shows nothing more happens steps a stated stretch.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

import pytest

from tests.support.browser_player_support import (
    ABSENT,
    CLICK_STEP,
    CUT_LATE,
    EDGE_SECONDS,
    LONG,
    OTHER,
    SHORT,
    SILENT_PEAK,
    SOUND_SECONDS,
    STRETCH_SECONDS,
    audible_seconds,
    audible_throughout,
    bytes_of,
    edge_peaks,
    ends,
    fed,
    hold_point_of,
    largest_step,
    last_ramp_end,
    peaks_between,
    quantum_seconds,
    result_of,
    start_of,
    starts,
    stepped,
    url,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.support.browser_player_support import SteppedStage

pytestmark = pytest.mark.browser

LONG_SECONDS = SOUND_SECONDS["long.wav"]
# A position well inside the long sound and past the first chunk of its streamed form.
HOLD_AT = 2.0
# The seconds the streamed sound's first chunk holds before the rest arrives.
LEAD_SECONDS = 1.0
# A distance from a position that is under any edge: a few render quanta.
WITHIN_AN_EDGE = 0.005
# How long the stepped page is late between the fade's end and the suspend.
LATE_PAGE_SECONDS = 0.3
HELD = {"kind": "held"}
SOUNDS_OF = {"whole": False, "streamed": True}
WHOLE_OR_STREAMED = pytest.mark.parametrize(
    "streamed", [False, True], ids=["whole", "streamed"]
)


async def hear_long(stage: SteppedStage, *, streamed: bool, then: str = SHORT) -> None:
    """Hear the long sound (streamed: its rest arrived), the sound `then` queued."""
    if streamed:
        await stage.prepare_bytes("long", LONG)
        await stage.hear(fed("long", lead_seconds=LEAD_SECONDS), url(then))
        await stage.act("feed", "long")
        await stage.until("s.scheduledChunks.length >= 2")
    else:
        await stage.hear_decoded(LONG, then)


def held_at(position: float) -> dict[str, object]:
    return {"kind": "held", "playedSeconds": position}


def silent_from(seen: dict[str, object], at: float) -> bool:
    """Whether every quantum that starts from `at` of the clock was silent."""
    return audible_seconds(seen, at, math.inf) == 0


@WHOLE_OR_STREAMED
def test_a_position_well_ahead_silences_the_sound_from_it_on_the_audio_clock(
    tmp_path: Path, *, streamed: bool
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await hear_long(stage, streamed=streamed)
            await stage.pause_at(0, HOLD_AT)
            await stage.until_resolved()
            await stage.stand_still_for(STRETCH_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    fade_end = last_ramp_end(seen, 0)
    assert result_of(seen) == held_at(HOLD_AT)
    assert fade_end == pytest.approx(start_of(seen) + HOLD_AT)
    assert audible_throughout(seen, start_of(seen) + 0.1, fade_end - EDGE_SECONDS)
    assert (
        largest_step(peaks_between(seen, fade_end - EDGE_SECONDS, fade_end))
        < CLICK_STEP
    )
    assert silent_from(seen, fade_end)
    assert seen["playedNow"][0] == HOLD_AT


@pytest.mark.parametrize("late", [False, True], ids=["chunk_in_time", "late_chunk"])
def test_a_position_in_a_chunk_not_yet_arrived_waits_for_it_and_holds_at_its_content(
    tmp_path: Path, *, late: bool
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.prepare_bytes("long", LONG)
            await stage.hear(fed("long", lead_seconds=LEAD_SECONDS))
            await stage.pause_at(0, HOLD_AT)
            if late:
                await stage.step_until(
                    f"s.playedNow[0] >= {LEAD_SECONDS}", within_seconds=LEAD_SECONDS
                )
            await stage.step_for(STRETCH_SECONDS)
            waiting = await stage.snapshot()
            await stage.act("feed", "long")
            await stage.until("s.scheduledChunks.length >= 2")
            await stage.until_resolved()
            return waiting, await stage.snapshot()

    waiting, seen = asyncio.run(scenario())

    first, second = seen["scheduledChunks"][:2]
    assert waiting["pauseAts"][0]["result"] is None
    assert waiting["contextStates"][0] == "running"
    assert result_of(seen) == held_at(HOLD_AT)
    # The position lies in the second chunk: that chunk's start plus the rest of it.
    assert last_ramp_end(seen, 0) == pytest.approx(
        second["start"] + HOLD_AT - first["seconds"]
    )
    assert silent_from(seen, last_ramp_end(seen, 0))
    if late:
        assert second["start"] > first["start"] + first["seconds"]


AT_ONCE: dict[str, Callable[[float], float]] = {
    "passed": lambda _read: 0.01,
    "within_an_edge": lambda read: read + WITHIN_AN_EDGE,
    "negative": lambda _read: -1.0,
    "nan": lambda _read: math.nan,
}


@pytest.mark.parametrize("position", AT_ONCE.values(), ids=AT_ONCE.keys())
def test_a_position_that_has_passed_holds_the_sound_at_once(
    tmp_path: Path, position: Callable[[float], float]
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(LONG)
            read = (await stage.snapshot())["playedNow"][0]
            await stage.pause_at(0, position(read))
            await stage.until_resolved()
            await stage.stand_still_for(STRETCH_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    [thrown] = (
        action["at"] for action in seen["actions"] if action["name"] == "pauseAt"
    )
    assert result_of(seen) == held_at(pytest.approx(hold_point_of(seen)))
    assert thrown < last_ramp_end(seen, 0) <= thrown + EDGE_SECONDS
    assert silent_from(seen, last_ramp_end(seen, 0))
    assert seen["playedNow"][0] == result_of(seen)["playedSeconds"]


ENDS: dict[str, Callable[[], float]] = {
    "the_length": lambda: LONG_SECONDS,
    "past_the_length": lambda: LONG_SECONDS + 5,
    "infinity": lambda: math.inf,
    "within_an_edge_of_the_end": lambda: LONG_SECONDS - WITHIN_AN_EDGE,
}


@WHOLE_OR_STREAMED
@pytest.mark.parametrize("position", ENDS.values(), ids=ENDS.keys())
def test_a_position_at_the_end_lets_the_sound_end_played_and_holds_before_the_next(
    tmp_path: Path, position: Callable[[], float], *, streamed: bool
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await hear_long(stage, streamed=streamed)
            await stage.pause_at(0, position())
            await stage.until_resolved()
            await stage.stand_still_for(STRETCH_SECONDS)
            held = await stage.snapshot()
            await stage.act("resume")
            await stage.finish()
            return held, await stage.snapshot()

    held, after = asyncio.run(scenario())

    assert result_of(held) == {"kind": "held_after_end"}
    assert ends(held) == ["played", None]
    assert held["sounds"][0]["playedSeconds"] == pytest.approx(LONG_SECONDS)
    assert starts(held) == ["playing", None]
    assert held["contextStates"][0] == "suspended"
    assert ends(after) == ["played", "played"]
    assert after["sounds"][1]["playedSeconds"] == pytest.approx(
        SOUND_SECONDS["short.wav"]
    )


def test_a_streamed_sound_that_ends_before_the_position_holds_before_the_next_sound(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.prepare_bytes("long", LONG)
            await stage.hear(fed("long", lead_seconds=LEAD_SECONDS), url(SHORT))
            await stage.pause_at(0, LONG_SECONDS + 5)
            await stage.step_for(STRETCH_SECONDS)
            await stage.act("feed", "long")
            await stage.until_resolved()
            await stage.stand_still_for(STRETCH_SECONDS)
            held = await stage.snapshot()
            await stage.act("resume")
            await stage.finish()
            return held, await stage.snapshot()

    held, after = asyncio.run(scenario())

    assert result_of(held) == {"kind": "held_after_end"}
    assert (ends(held), starts(held)) == (["played", None], ["playing", None])
    assert ends(after) == ["played", "played"]


def test_a_streamed_sound_that_fails_before_the_position_holds_before_the_next_sound(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(CUT_LATE), url(SHORT))
            await stage.pause_at(0, LONG_SECONDS)
            await stage.step_for(STRETCH_SECONDS)
            stage.side.held_release.set()
            await stage.until("s.sounds[0].endedValue === 'failed'")
            await stage.until_resolved()
            await stage.stand_still_for(STRETCH_SECONDS)
            held = await stage.snapshot()
            await stage.act("resume")
            await stage.finish()
            return held, await stage.snapshot()

    held, after = asyncio.run(scenario())

    assert result_of(held) == {"kind": "held_after_end"}
    assert (ends(held), starts(held)) == (["failed", None], ["playing", None])
    assert ends(after) == ["failed", "played"]


def test_resume_after_a_hold_plays_the_rest_faded_in_then_the_queue(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await hear_long(stage, streamed=False)
            await stage.pause_at(0, HOLD_AT)
            await stage.until_resolved()
            await stage.act("resume")
            await stage.finish()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["played", "played"]
    assert largest_step(edge_peaks(seen, "resume")) < CLICK_STEP
    assert (
        max(peaks_between(seen, last_ramp_end(seen, 0) + EDGE_SECONDS, math.inf))
        > SILENT_PEAK
    )
    assert seen["playedSeries"] == sorted(seen["playedSeries"])
    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(LONG_SECONDS)


@pytest.mark.parametrize("hold", ["held", "held_after_end"])
def test_stop_after_a_hold_drops_the_waiting_sounds_and_the_pause_stays(
    tmp_path: Path, hold: str
) -> None:
    position = HOLD_AT if hold == "held" else LONG_SECONDS

    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await hear_long(stage, streamed=False)
            await stage.pause_at(0, position)
            await stage.until_resolved()
            await stage.act("stop")
            await stage.stand_still_for(STRETCH_SECONDS)
            stopped = await stage.snapshot()
            await stage.hand_over(url(OTHER))
            await stage.stand_still_for(STRETCH_SECONDS)
            waiting = await stage.snapshot()
            await stage.act("resume")
            await stage.finish()
            return stopped, waiting

    stopped, waiting = asyncio.run(scenario())

    assert result_of(stopped)["kind"] == hold
    assert ends(stopped)[1] == "stopped"
    if hold == "held":
        assert ends(stopped)[0] == "stopped"
        assert stopped["sounds"][0]["playedSeconds"] == pytest.approx(HOLD_AT)
    assert stopped["contextStates"][0] == "suspended"
    assert starts(waiting)[2] is None
    assert ends(waiting)[2] is None


async def throw_hold(stage: SteppedStage, hold: str, position: float) -> None:
    """Throw the app's hold: `pauseAt` at `position`, or `pause`."""
    if hold == "pauseAt":
        await stage.pause_at(0, position)
    else:
        await stage.act("pause")


async def step_past_the_fade(stage: SteppedStage, seconds: float) -> dict[str, object]:
    """Render in one step to `seconds` past the last fade to silence; the page after."""
    thrown = await stage.snapshot()
    await stage.step_for(last_ramp_end(thrown, 0) + seconds - thrown["contextNow"])
    return await stage.snapshot()


def hold_point(seen: dict[str, object], hold: str, position: float) -> float:
    """Where the hold put the sound: the `pauseAt` position, else the fade's end."""
    return position if hold == "pauseAt" else hold_point_of(seen)


HOLDS = pytest.mark.parametrize("hold", ["pauseAt", "pause"])


@HOLDS
def test_a_late_page_leaves_the_hold_at_its_point_and_resume_loses_at_most_the_lag(
    tmp_path: Path, hold: str
) -> None:
    async def scenario() -> tuple[
        dict[str, object], dict[str, object], dict[str, object]
    ]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(LONG)
            await throw_hold(stage, hold, HOLD_AT)
            late = await step_past_the_fade(stage, LATE_PAGE_SECONDS)
            await stage.until("s.contextStates[0] === 'suspended'")
            held = await stage.snapshot()
            await stage.act("resume")
            await stage.step_for(STRETCH_SECONDS)
            return late, held, await stage.snapshot()

    late, held, resumed = asyncio.run(scenario())

    point = hold_point(held, hold, HOLD_AT)
    fade_end = last_ramp_end(held, 0)
    assert held["contextNow"] >= fade_end + LATE_PAGE_SECONDS
    assert silent_from(held, fade_end)
    assert held["playedNow"][0] == pytest.approx(point)
    assert late["playedNow"][0] == pytest.approx(point)
    ran_on = held["contextNow"] - fade_end
    elapsed = resumed["contextNow"] - held["contextNow"]
    lost = resumed["playedNow"][0] - point - elapsed
    assert 0 <= lost <= ran_on + quantum_seconds(held)
    assert resumed["playedSeries"] == sorted(resumed["playedSeries"])


@pytest.mark.parametrize("then", ["resume", "stop"])
@HOLDS
def test_a_held_sound_that_runs_out_in_a_late_page_does_not_end_played(
    tmp_path: Path, hold: str, then: str
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(SHORT, OTHER)
            await throw_hold(stage, hold, 0.2)
            await step_past_the_fade(
                stage, SOUND_SECONDS["short.wav"] + STRETCH_SECONDS
            )
            await stage.until("s.contextStates[0] === 'suspended'")
            held = await stage.snapshot()
            await stage.act(then)
            await stage.step_for(STRETCH_SECONDS)
            await stage.finish()
            return held, await stage.snapshot()

    held, after = asyncio.run(scenario())

    assert ends(held) == [None, None]
    assert starts(held) == ["playing", None]
    assert held["playedNow"][0] == pytest.approx(hold_point(held, hold, 0.2))
    if then == "resume":
        assert ends(after) == ["played", "played"]
        assert after["sounds"][0]["playedSeconds"] == pytest.approx(
            SOUND_SECONDS["short.wav"]
        )
    else:
        assert ends(after) == ["stopped", "stopped"]
        assert after["sounds"][0]["playedSeconds"] == pytest.approx(
            hold_point(held, hold, 0.2)
        )
        assert starts(after) == ["playing", "ended"]


def test_the_read_while_paused_is_where_the_pause_fade_ended(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(LONG)
            await stage.paused()
            await stage.stand_still_for(STRETCH_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert seen["playedNow"][0] == pytest.approx(hold_point_of(seen))


def test_the_read_during_a_stops_edge_never_passes_the_edges_end(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(LONG)
            await stage.act("readAfterStopPast", STRETCH_SECONDS)
            await stage.finish()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert seen["readPastStop"] == pytest.approx(hold_point_of(seen))
    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(hold_point_of(seen))


async def mute(stage: SteppedStage) -> None:
    await stage.set_muted(muted=True)


async def mute_during_the_fade(stage: SteppedStage) -> None:
    thrown = await stage.snapshot()
    two_quanta = 2 * quantum_seconds(thrown)
    await stage.step_for(last_ramp_end(thrown, 0) - two_quanta - thrown["contextNow"])
    await mute(stage)


# The switch the app throws while a call waits, and how the sound it ends reads.
SWITCHES: dict[str, tuple[Callable[[SteppedStage], object], str]] = {
    "stop": (lambda stage: stage.act("stop"), "stopped"),
    "mute": (mute, "muted"),
    "mute_during_the_fade": (mute_during_the_fade, "muted"),
    "close": (lambda stage: stage.act("close"), "stopped"),
}


@pytest.mark.parametrize("switch", SWITCHES.items(), ids=SWITCHES.keys())
def test_a_switch_that_ends_the_sound_overtakes_the_waiting_call(
    tmp_path: Path, switch: tuple[str, tuple[Callable[[SteppedStage], object], str]]
) -> None:
    _, (throw, end_kind) = switch

    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(LONG)
            await stage.pause_at(0, HOLD_AT)
            await throw(stage)
            await stage.until("s.pauseAts[0].result !== null")
            await stage.finish()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert result_of(seen) == {"kind": "overtaken"}
    assert ends(seen) == [end_kind]
    assert seen["sounds"][0]["playedSeconds"] <= HOLD_AT


def test_pause_while_a_call_waits_holds_now_and_the_call_gives_where_it_held(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(LONG)
            await stage.pause_at(0, HOLD_AT)
            await stage.act("pause")
            await stage.until_resolved()
            await stage.stand_still_for(STRETCH_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    [pause_at] = (
        action["at"] for action in seen["actions"] if action["name"] == "pause"
    )
    assert result_of(seen) == held_at(pytest.approx(hold_point_of(seen)))
    assert result_of(seen)["playedSeconds"] < HOLD_AT / 2
    assert pause_at < last_ramp_end(seen, 0) <= pause_at + EDGE_SECONDS
    assert silent_from(seen, last_ramp_end(seen, 0))


def test_resume_while_a_call_waits_cancels_the_fade_and_the_sound_plays_on(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(LONG)
            await stage.pause_at(0, HOLD_AT)
            await stage.act("resume")
            await stage.until("s.pauseAts[0].result !== null")
            await stage.finish()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    start = start_of(seen)
    assert result_of(seen) == {"kind": "overtaken"}
    assert audible_throughout(
        seen, start + HOLD_AT - EDGE_SECONDS, start + HOLD_AT + STRETCH_SECONDS
    )
    assert ends(seen) == ["played"]
    assert seen["contextStates"][0] == "running"


def test_a_newer_call_replaces_the_waiting_one(tmp_path: Path) -> None:
    later = HOLD_AT + 1.0

    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(LONG)
            await stage.pause_at(0, HOLD_AT)
            await stage.pause_at(0, later)
            await stage.until_resolved(1)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    start = start_of(seen)
    assert result_of(seen, 0) == {"kind": "overtaken"}
    assert result_of(seen, 1) == held_at(later)
    assert last_ramp_end(seen, 0) == pytest.approx(start + later)
    assert audible_throughout(
        seen, start + HOLD_AT - EDGE_SECONDS, start + later - EDGE_SECONDS
    )
    assert silent_from(seen, last_ramp_end(seen, 0))


@pytest.mark.parametrize("paused", [True, False], ids=["paused", "pausing"])
def test_a_call_for_the_held_sound_while_paused_or_pausing_gives_the_hold_points_read(
    tmp_path: Path, *, paused: bool
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(LONG)
            if paused:
                await stage.paused()
            else:
                await stage.act("pause")
            await stage.pause_at(0, HOLD_AT)
            if paused:
                await stage.until("s.pauseAts[0].result !== null")
            else:
                await stage.until_resolved()
            await stage.stand_still_for(STRETCH_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert result_of(seen) == held_at(pytest.approx(hold_point_of(seen)))
    assert seen["playedNow"][0] == pytest.approx(hold_point_of(seen))
    assert result_of(seen)["playedSeconds"] < HOLD_AT / 2


async def hear_second(stage: SteppedStage, *sounds: dict[str, str | int]) -> None:
    """Hand the sounds over and step until the second is heard."""
    await stage.hand_over(*sounds)
    await stage.until("s.sounds[1].startedValue === 'playing'")
    await stage.step_until("s.heardOnsets.length > 0", within_seconds=EDGE_SECONDS)
    await stage.step_for(0.1)


async def ended_played(stage: SteppedStage) -> None:
    await stage.prepare_bytes(SHORT, SHORT)
    await stage.prepare_bytes(LONG, LONG)
    await stage.hand_over(bytes_of(SHORT), bytes_of(LONG))
    await stage.until("s.sounds[0].startedValue === 'playing'")
    await stage.step_until(
        "s.sounds[0].endedValue !== null",
        within_seconds=SOUND_SECONDS["short.wav"] + EDGE_SECONDS,
    )
    await stage.until("s.sounds[1].startedValue === 'playing'")
    await stage.step_for(0.1)


async def ended_failed(stage: SteppedStage) -> None:
    await hear_second(stage, url(ABSENT), url(LONG))


async def ended_refused(stage: SteppedStage) -> None:
    await stage.act("rejectNextResume")
    await hear_second(stage, url(SHORT), url(LONG))


ENDED_ON_ITS_OWN = {
    "played": ended_played,
    "failed": ended_failed,
    "refused": ended_refused,
}


@pytest.mark.parametrize(
    "arrange", ENDED_ON_ITS_OWN.values(), ids=ENDED_ON_ITS_OWN.keys()
)
def test_a_target_that_ended_on_its_own_holds_the_sound_that_plays_now(
    tmp_path: Path, arrange: Callable[[SteppedStage], object]
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await arrange(stage)
            await stage.pause_at(0, HOLD_AT)
            await stage.until_resolved()
            await stage.stand_still_for(STRETCH_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert result_of(seen) == {"kind": "held_after_end"}
    assert ends(seen)[0] is not None
    assert ends(seen)[1] is None
    assert seen["playedNow"][1] == pytest.approx(hold_point_of(seen, 1))
    assert silent_from(seen, last_ramp_end(seen, 0))
    assert seen["contextStates"][0] == "suspended"


def test_a_queued_target_is_held_at_the_position_once_it_plays(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(OTHER, LONG)
            await stage.pause_at(1, HOLD_AT)
            await stage.until_resolved()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert result_of(seen) == held_at(HOLD_AT)
    assert ends(seen) == ["played", None]
    assert last_ramp_end(seen, 0) == pytest.approx(start_of(seen, 1) + HOLD_AT)
    assert silent_from(seen, last_ramp_end(seen, 0))
    assert seen["playedNow"][1] == HOLD_AT


def test_a_queued_target_while_paused_is_held_after_the_resume(
    tmp_path: Path,
) -> None:
    position = 0.2

    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(OTHER)
            await stage.paused()
            await stage.hand_over_decoded(SHORT)
            await stage.pause_at(1, position)
            await stage.stand_still_for(STRETCH_SECONDS)
            await stage.act("resume")
            waiting = await stage.snapshot()
            await stage.until_resolved()
            return waiting, await stage.snapshot()

    waiting, seen = asyncio.run(scenario())

    assert waiting["pauseAts"][0]["result"] is None
    assert result_of(seen) == held_at(position)
    assert ends(seen) == ["played", None]
    assert last_ramp_end(seen, 0) == pytest.approx(start_of(seen, 1) + position)
    assert silent_from(seen, last_ramp_end(seen, 0))


def test_a_playback_of_another_player_is_overtaken_and_the_sound_plays_on(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(LONG)
            await stage.pause_at("foreign", 1.0)
            await stage.until("s.pauseAts[0].result !== null")
            await stage.finish()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert result_of(seen) == {"kind": "overtaken"}
    assert ends(seen) == ["played"]
    assert seen["contextStates"][0] == "running"


def test_a_call_while_a_stops_edge_is_in_flight_is_overtaken(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(LONG)
            await stage.act("stop")
            await stage.pause_at(0, HOLD_AT)
            await stage.until("s.pauseAts[0].result !== null")
            await stage.finish()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert result_of(seen) == {"kind": "overtaken"}
    assert ends(seen) == ["stopped"]
