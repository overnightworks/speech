"""A hold at a position that a resume or a late page meets.

The flows run in Chromium on the stand-in app's page, on a context the test steps: the
resume's fade in, the hold a late page leaves and the app's calls thrown while a
resume is in flight.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from tests.support.browser_pause_at_support import (
    HOLD_AT,
    SWITCHES,
    held_at,
    silent_from,
)
from tests.support.browser_player_support import (
    CLICK_STEP,
    LONG,
    OTHER,
    SHORT,
    SOUND_SECONDS,
    STRETCH_SECONDS,
    action_times,
    ends,
    largest_step,
    last_ramp_end,
    peaks_between,
    quantum_seconds,
    result_of,
    start_of,
    stepped,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support.browser_player_support import SteppedStage

pytestmark = pytest.mark.browser

LONG_SECONDS = SOUND_SECONDS["long.wav"]
# A position inside the short sound that the late page's context renders past.
LATE_HOLD_AT = 0.2


def test_resume_in_the_edge_of_a_hold_after_the_end_overtakes_the_call_and_plays_on(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(LONG, SHORT)
            await stage.pause_at(0, LONG_SECONDS)
            await stage.step_until(
                "s.sounds[0].endedValue === 'played'",
                within_seconds=LONG_SECONDS,
                quanta=4,
            )
            await stage.act("resume")
            await stage.until_resolved()
            await stage.finish()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert result_of(seen) == {"kind": "overtaken"}
    assert ends(seen) == ["played", "played"]
    assert seen["contextStates"][0] == "running"


def test_a_call_thrown_as_the_resume_fades_in_fades_out_from_where_the_volume_is(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear_decoded(LONG)
            await stage.paused()
            await stage.stand_still_for(STRETCH_SECONDS)
            await stage.act("resume")
            resumed = await stage.snapshot()
            # One and a half of the resume's edges ahead: the fade starts inside it.
            edge = last_ramp_end(resumed, 1) - resumed["contextNow"]
            await stage.pause_at(
                0, resumed["contextNow"] - start_of(resumed) + 1.5 * edge
            )
            await stage.until_resolved()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    [resumed_at] = action_times(seen, "resume")
    assert result_of(seen)["kind"] == "held"
    assert (
        largest_step(
            peaks_between(
                seen, resumed_at - quantum_seconds(seen), last_ramp_end(seen, 0)
            )
        )
        < CLICK_STEP
    )
    assert silent_from(seen, last_ramp_end(seen, 0))


async def into_the_late_window(stage: SteppedStage) -> None:
    """Hold the short sound at `LATE_HOLD_AT`, then render the context past its end.

    The page is late: its timers have not run, so the player still waits for the fade
    that has ended on the audio clock, and the sound's sources have run out held.
    """
    await stage.hear_decoded(SHORT, OTHER)
    await stage.pause_at(0, LATE_HOLD_AT)
    thrown = await stage.snapshot()
    sound_end = start_of(thrown) + SOUND_SECONDS["short.wav"]
    await stage.render(
        stage.quanta_for(sound_end + STRETCH_SECONDS - thrown["contextNow"])
    )
    assert ends(await stage.snapshot()) == [None, None]


@pytest.mark.parametrize("switch", ["stop", "mute", "close"])
def test_a_switch_in_the_late_window_ends_the_held_sound_at_its_hold_point(
    tmp_path: Path, switch: str
) -> None:
    throw, end_kind = SWITCHES[switch]

    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await into_the_late_window(stage)
            await throw(stage)
            await stage.until("s.pauseAts[0].result !== null")
            await stage.finish()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert result_of(seen) == {"kind": "overtaken"}
    assert ends(seen)[0] == end_kind
    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(LATE_HOLD_AT)


def test_resume_in_the_late_window_ends_the_held_sound_played_and_starts_the_next(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await into_the_late_window(stage)
            await stage.act("resume")
            await stage.until_resolved()
            await stage.finish()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert result_of(seen) == {"kind": "overtaken"}
    assert ends(seen) == ["played", "played"]
    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(
        SOUND_SECONDS["short.wav"]
    )


def test_pause_in_the_late_window_keeps_the_hold_point_of_the_call(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await into_the_late_window(stage)
            await stage.act("pause")
            await stage.until_resolved()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert result_of(seen) == held_at(LATE_HOLD_AT)
    assert ends(seen) == [None, None]


async def resume_and_pause_at_in_one_task(stage: SteppedStage, position: float) -> None:
    """Hear the long sound and the other, pause, then resume and hold at `position`.

    Both are thrown in one page task, so the call comes while the resume is in flight.
    """
    await stage.hear_decoded(LONG, OTHER)
    await stage.paused()
    await stage.act("resumeThenPauseAt", 0, position)


def test_a_call_thrown_while_the_resume_is_in_flight_holds_at_the_position(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await resume_and_pause_at_in_one_task(stage, HOLD_AT)
            await stage.until_resolved()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert result_of(seen) == held_at(HOLD_AT)
    assert ends(seen) == [None, None]


def test_a_call_thrown_while_the_resume_is_in_flight_holds_a_sound_that_runs_out_late(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await resume_and_pause_at_in_one_task(stage, HOLD_AT)
            thrown = await stage.snapshot()
            sound_end = start_of(thrown) + LONG_SECONDS
            await stage.render(
                stage.quanta_for(sound_end + STRETCH_SECONDS - thrown["contextNow"])
            )
            late = await stage.snapshot()
            await stage.until_resolved()
            return late, await stage.snapshot()

    late, seen = asyncio.run(scenario())

    assert ends(late) == [None, None]
    assert result_of(seen) == held_at(HOLD_AT)
    assert ends(seen) == [None, None]
