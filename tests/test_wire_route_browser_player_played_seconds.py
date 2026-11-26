"""An app's browser reads how many seconds each of its sounds has played.

The flows run in Chromium on the stand-in app's page, on a context the test steps; see
the shared stage. The two that block the page around the player's start call run on
a real context: a stall in which the render thread runs on exists only there.
"""

from __future__ import annotations

import asyncio
import itertools
from typing import TYPE_CHECKING

import pytest

from tests.support.browser_player_support import (
    ALL_ENDED,
    CUT_LATE,
    EDGE_SECONDS,
    FIRST_SAMPLE_SILENT_FRAMES,
    HEARD,
    LONG,
    SHORT,
    SOUND_SECONDS,
    STARTED,
    STREAMED,
    STRETCH_SECONDS,
    activated,
    bytes_of,
    fed,
    heard_start_of,
    onset_after_start_frames,
    quantum_seconds,
    ramp_end,
    start_of,
    stepped,
    url,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.browser

# The late-chunk sound's first chunk; how far its read must grow after the next.
LEAD_SECONDS = 1.0
GROWTH_SECONDS = 0.3
# How far past a stop's edge the clock reads when the cut comes.
PAST_THE_EDGE_SECONDS = 0.5
# Where the page's task is blocked around the player's start call.
AROUND_THE_START_CALL = pytest.mark.parametrize(
    "stall", ["before", "after"], ids=["before_the_call", "after_the_call"]
)


def test_a_sound_played_through_reports_its_full_length_as_played_seconds(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hand_over(url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(
        SOUND_SECONDS["short.wav"]
    )


@pytest.mark.parametrize("cut", ["stop", "mute"])
@pytest.mark.parametrize("sound", [LONG, STREAMED], ids=["whole", "streamed"])
def test_a_cut_sound_reports_the_seconds_the_context_clock_passed_since_it_started(
    tmp_path: Path, cut: str, sound: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(sound), url(SHORT))
            if cut == "stop":
                await stage.act("stop")
            else:
                await stage.set_muted(muted=True)
            return await stage.finish()

    seen = asyncio.run(scenario())

    cut_sound, waiting = seen["sounds"]
    start = cut_sound["start"]["contextTime"]
    if cut == "mute":
        assert cut_sound["playedSeconds"] == pytest.approx(
            seen["contextAtAction"] - start
        )
    else:
        # A stopped sound plays on through its edge, up to the end of the ramp.
        assert cut_sound["playedSeconds"] == pytest.approx(ramp_end(seen, 0) - start)
    assert waiting["playedSeconds"] == 0


def test_a_sound_that_never_started_ends_with_zero_played_seconds(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.set_muted(muted=True)
            await stage.hand_over(url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert seen["sounds"][0]["playedSeconds"] == 0


@pytest.mark.parametrize("sound", [SHORT, STREAMED], ids=["whole", "streamed"])
def test_the_start_a_sound_hands_over_is_the_context_time_its_first_sample_is_heard(
    tmp_path: Path, sound: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hand_over(url(sound))
            await stage.until(STARTED)
            await stage.step_until(HEARD, within_seconds=EDGE_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert seen["sounds"][0]["start"]["onThePlayersContext"]
    assert onset_after_start_frames(seen) == FIRST_SAMPLE_SILENT_FRAMES


@AROUND_THE_START_CALL
def test_a_sound_is_heard_at_its_start_or_later_by_the_stall_around_the_start_call(
    tmp_path: Path, stall: str
) -> None:
    """The sound is heard from its context time, or later by up to the page's stall.

    The page's task is blocked around the player's start call for longer than the
    output's lead. The first sample is then heard no earlier than the start less one
    render quantum and no later than the start plus the blocked time plus one
    quantum, and the played seconds read ahead of the seconds heard by no more than
    the blocked time plus one quantum, both read at one instant of the clock.
    """

    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.act("stallNextStart", stall)
            await stage.hand_over(url(LONG))
            await stage.until("s.heardOnsets.length > 0")
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    (stalled,) = seen["stalls"]
    blocked = stalled["to"] - stalled["from"]
    quantum = quantum_seconds(seen)
    start = start_of(seen)
    heard = heard_start_of(seen)
    assert start - quantum <= heard <= start + blocked + quantum
    played_ahead = seen["playedNow"][0] - (seen["contextNow"] - heard)
    assert played_ahead <= blocked + quantum


@AROUND_THE_START_CALL
def test_a_streamed_sounds_next_chunk_is_chained_from_its_first_chunks_asked_start(
    tmp_path: Path, stall: str
) -> None:
    """The next chunk is chained from the start the first chunk was asked for.

    Under a stall past the output's lead, before or after the start call, the next
    chunk starts at the first chunk's asked start plus its length, not at where the
    first chunk was heard. How far the first chunk is heard late is the sibling cell's
    bound.
    """

    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.prepare_bytes("long", LONG)
            await stage.act("stallNextStart", stall)
            await stage.hand_over(fed("long", lead_seconds=LEAD_SECONDS))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.act("feed", "long")
            await stage.until("s.scheduledChunks.length >= 2")
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert len(seen["stalls"]) == 1
    first, second = seen["scheduledChunks"][:2]
    assert second["start"] == pytest.approx(first["start"] + first["seconds"])


@pytest.mark.parametrize("sound", [SHORT, STREAMED], ids=["whole", "streamed"])
def test_the_start_of_a_sound_scheduled_right_after_another_ended_is_where_it_is_heard(
    tmp_path: Path, sound: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hand_over(url(SHORT), url(sound))
            await stage.until(STARTED)
            await stage.step_until(
                "s.sounds[0].endedValue !== null",
                within_seconds=SOUND_SECONDS["short.wav"] + EDGE_SECONDS,
            )
            await stage.until("s.sounds[1].startedValue === 'playing'")
            await stage.step_until(
                "s.heardOnsets.length === 2", within_seconds=EDGE_SECONDS
            )
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    # The second sound's time has passed when it is scheduled; the player still hands
    # over a start the render thread has not passed.
    assert onset_after_start_frames(seen, 1) == FIRST_SAMPLE_SILENT_FRAMES


def test_a_streamed_sound_reads_still_across_a_late_chunks_gap_and_grows_after_it(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], ...]:
        async with stepped(tmp_path) as stage:
            await stage.prepare_bytes("long", LONG)
            await stage.hear(fed("long", lead_seconds=LEAD_SECONDS), for_seconds=0)
            await stage.step_until(
                f"s.playedNow[0] >= {LEAD_SECONDS}", within_seconds=LEAD_SECONDS
            )
            gap_begins = await stage.snapshot()
            await stage.step_for(STRETCH_SECONDS)
            in_gap = await stage.snapshot()
            await stage.act("feed", "long")
            await stage.step_until(
                f"s.playedNow[0] > {LEAD_SECONDS + GROWTH_SECONDS}",
                within_seconds=GROWTH_SECONDS + EDGE_SECONDS,
            )
            return gap_begins, in_gap, await stage.snapshot()

    gap_begins, in_gap, after = asyncio.run(scenario())

    assert in_gap["contextNow"] > gap_begins["contextNow"]
    assert in_gap["playedNow"][0] == pytest.approx(LEAD_SECONDS)
    assert in_gap["playedNow"][0] == gap_begins["playedNow"][0]
    assert after["playedNow"][0] > in_gap["playedNow"][0]


@pytest.mark.parametrize("sound", [LONG, STREAMED], ids=["whole", "streamed"])
def test_a_pause_holds_the_seconds_a_sound_has_played(
    tmp_path: Path, sound: str
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(sound), for_seconds=0)
            await stage.step_until(
                f"s.playedNow[0] > {GROWTH_SECONDS}", within_seconds=GROWTH_SECONDS
            )
            await stage.paused()
            held = await stage.snapshot()
            await stage.stand_still_for(STRETCH_SECONDS)
            return held, await stage.snapshot()

    held, later = asyncio.run(scenario())

    assert held["playedNow"][0] > GROWTH_SECONDS
    assert later["playedNow"][0] == held["playedNow"][0]


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
def test_the_seconds_a_sound_has_played_equal_those_its_end_reports(
    tmp_path: Path, *, streamed: bool
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.prepare_bytes("long", LONG)
            if streamed:
                await stage.hand_over(fed("long", lead_seconds=LEAD_SECONDS))
                await stage.act("feed", "long")
            else:
                await stage.hand_over(bytes_of("long"))
            await stage.finish()
            await stage.step_for(STRETCH_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(
        SOUND_SECONDS["long.wav"]
    )
    assert seen["playedNow"][0] == seen["sounds"][0]["playedSeconds"]


@pytest.mark.parametrize("cut", ["stop", "mute"])
def test_a_cut_sounds_read_stays_at_the_seconds_its_end_reports(
    tmp_path: Path, cut: str
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.prepare_bytes("long", LONG)
            await stage.hear(fed("long", lead_seconds=LEAD_SECONDS))
            await stage.act("feed", "long")
            if cut == "stop":
                await stage.act("stop")
            else:
                await stage.set_muted(muted=True)
            await stage.finish()
            after_end = await stage.snapshot()
            await stage.step_for(STRETCH_SECONDS)
            return after_end, await stage.snapshot()

    after_end, later = asyncio.run(scenario())

    reported = later["sounds"][0]["playedSeconds"]
    assert after_end["playedNow"][0] == reported
    assert later["playedNow"][0] == reported


@pytest.mark.parametrize("cut", ["mute", "close"])
def test_a_stopped_sound_cut_after_its_edge_reports_no_more_than_the_edge(
    tmp_path: Path, cut: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.prepare_bytes("long", LONG)
            await stage.hear(fed("long", lead_seconds=LEAD_SECONDS))
            await stage.act("feed", "long")
            await stage.act("stopThenCutPastItsEdge", cut, PAST_THE_EDGE_SECONDS)
            await stage.finish()
            await stage.step_for(STRETCH_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    reported = seen["sounds"][0]["playedSeconds"]
    assert reported == pytest.approx(ramp_end(seen, 0) - start_of(seen))
    assert seen["playedNow"][0] == reported
    series = seen["playedSeries"]
    assert all(later >= earlier for earlier, later in itertools.pairwise(series))


def test_a_streamed_sound_that_breaks_off_after_playing_keeps_what_it_played(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(CUT_LATE))
            before_break = await stage.snapshot()
            stage.side.held_release.set()
            await stage.until(ALL_ENDED)
            await stage.step_for(STRETCH_SECONDS)
            return before_break, await stage.snapshot()

    before_break, after_break = asyncio.run(scenario())

    assert after_break["sounds"][0]["endedValue"] == "failed"
    assert before_break["playedNow"][0] > 0
    assert after_break["playedNow"][0] >= before_break["playedNow"][0]


def test_the_read_never_steps_back_across_a_stops_fade_and_its_end(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.prepare_bytes("long", LONG)
            await stage.hear(fed("long", lead_seconds=LEAD_SECONDS))
            await stage.act("feed", "long")
            await stage.act("stop")
            await stage.step_until(ALL_ENDED, within_seconds=EDGE_SECONDS)
            await stage.step_for(STRETCH_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    series = seen["playedSeries"]
    assert len(series) > 1
    assert all(later >= earlier for earlier, later in itertools.pairwise(series))


def test_a_sound_muted_at_hand_over_reads_zero_played_seconds(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.set_muted(muted=True)
            await stage.hand_over(url(SHORT))
            await stage.finish()
            await stage.step_for(STRETCH_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert seen["playedNow"][0] == 0
