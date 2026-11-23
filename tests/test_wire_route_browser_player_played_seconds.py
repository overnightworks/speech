"""An app's browser reads how many seconds each of its sounds has played.

The flows run in Chromium on the stand-in app's page; see the shared stage.
"""

from __future__ import annotations

import asyncio
import itertools
from typing import TYPE_CHECKING

import pytest

from tests.support.browser_player_support import (
    AUDIBLE_RMS,
    CUT_LATE,
    LONG,
    ONSET_BOUND_FRAMES,
    SHORT,
    SOUND_SECONDS,
    STREAMED,
    activated,
    bytes_of,
    fed,
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


def test_a_sound_played_through_reports_its_full_length_as_played_seconds(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(
        SOUND_SECONDS["short.wav"], abs=seen["clockStep"]
    )


@pytest.mark.parametrize("cut", ["stop", "mute"])
@pytest.mark.parametrize("sound", [LONG, STREAMED], ids=["whole", "streamed"])
def test_a_cut_sound_reports_the_seconds_the_context_clock_passed_since_it_started(
    tmp_path: Path, cut: str, sound: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(sound), url(SHORT))
            # The sound plays on whatever the page's timing: a streamed body held back
            # would leave the output silent once its first bytes have played.
            stage.side.held_release.set()
            await stage.until(f"s.sounds[0].peakRms > {AUDIBLE_RMS}")
            await stage.settle()
            if cut == "stop":
                await stage.act("stop")
            else:
                await stage.set_muted(muted=True)
            return await stage.finish()

    seen = asyncio.run(scenario())

    cut_sound, waiting = seen["sounds"]
    start = cut_sound["start"]["contextTime"]
    elapsed = cut_sound["contextAtEnd"] - start
    assert elapsed > 0
    if cut == "mute":
        assert cut_sound["playedSeconds"] == pytest.approx(
            elapsed, abs=seen["clockStep"]
        )
    else:
        # A stopped sound plays on through its edge, up to the end it reports.
        step = seen["clockStep"]
        assert cut_sound["playedSeconds"] >= seen["contextAtAction"] - start - step
        assert cut_sound["playedSeconds"] <= elapsed + step
    assert waiting["playedSeconds"] == 0


def test_a_sound_that_never_started_ends_with_zero_played_seconds(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
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
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(sound))
            await stage.until("s.heardOnsets.length > 0")
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    start = seen["sounds"][0]["start"]
    assert start["onThePlayersContext"]
    # The start is never earlier than the render thread's clock, so the sound is heard
    # within one render quantum of it. The page hears the onset to one sample, and the
    # tone's first sample is silent: the onset heard is a sample later.
    assert seen["heardOnsets"][0] == pytest.approx(
        start["contextTime"], abs=ONSET_BOUND_FRAMES / seen["sampleRate"]
    )


@pytest.mark.parametrize(
    "stall",
    ["before", "after", "beforeAndAfterRead"],
    ids=[
        "before_the_call",
        "after_the_call",
        "before_the_call_and_after_the_clock_read",
    ],
)
def test_the_start_is_where_it_is_heard_when_the_page_stalls_around_the_start_call(
    tmp_path: Path, stall: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.act("stallNextStart", stall)
            await stage.hand_over(url(SHORT))
            await stage.until("s.heardOnsets.length > 0")
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    # Before the call the render thread passes the start asked for and plays the sound
    # from the quantum it is on; after it the sound plays at the start asked for. A
    # stall after the clock read leaves the start the clock read.
    assert seen["heardOnsets"][0] == pytest.approx(
        seen["sounds"][0]["start"]["contextTime"],
        abs=ONSET_BOUND_FRAMES / seen["sampleRate"],
    )


@pytest.mark.parametrize("sound", [SHORT, STREAMED], ids=["whole", "streamed"])
def test_the_start_of_a_sound_scheduled_right_after_another_ended_is_where_it_is_heard(
    tmp_path: Path, sound: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(SHORT), url(sound))
            await stage.until("s.heardOnsets.length === 2")
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    start = seen["sounds"][1]["start"]
    # The second sound's time has passed when it is scheduled; the player still hands
    # over a start the render thread has not passed.
    assert seen["heardOnsets"][1] == pytest.approx(
        start["contextTime"], abs=ONSET_BOUND_FRAMES / seen["sampleRate"]
    )


def test_a_streamed_sound_reads_still_across_a_late_chunks_gap_and_grows_after_it(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], ...]:
        async with activated(tmp_path) as stage:
            await stage.prepare_bytes("long", LONG)
            await stage.hand_over(fed("long", lead_seconds=LEAD_SECONDS))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.until(f"s.playedNow[0] >= {LEAD_SECONDS}")
            gap_begins = await stage.snapshot()
            await stage.settle()
            in_gap = await stage.snapshot()
            await stage.act("feed", "long")
            await stage.until(f"s.playedNow[0] > {LEAD_SECONDS + GROWTH_SECONDS}")
            return gap_begins, in_gap, await stage.snapshot()

    gap_begins, in_gap, after = asyncio.run(scenario())

    step = in_gap["clockStep"]
    assert in_gap["contextNow"] > gap_begins["contextNow"]
    assert in_gap["playedNow"][0] == pytest.approx(LEAD_SECONDS, abs=step)
    assert in_gap["playedNow"][0] == pytest.approx(gap_begins["playedNow"][0], abs=step)
    assert after["playedNow"][0] > in_gap["playedNow"][0]


@pytest.mark.parametrize("sound", [LONG, STREAMED], ids=["whole", "streamed"])
def test_a_pause_holds_the_seconds_a_sound_has_played(
    tmp_path: Path, sound: str
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(sound))
            await stage.until(f"s.playedNow[0] > {GROWTH_SECONDS}")
            await stage.act("pause")
            await stage.until("s.contextStates[0] === 'suspended'")
            held = await stage.snapshot()
            await stage.settle()
            return held, await stage.snapshot()

    held, later = asyncio.run(scenario())

    assert held["playedNow"][0] > GROWTH_SECONDS
    assert later["playedNow"][0] == held["playedNow"][0]


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
def test_the_seconds_a_sound_has_played_equal_those_its_end_reports(
    tmp_path: Path, *, streamed: bool
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.prepare_bytes("long", LONG)
            if streamed:
                await stage.hand_over(fed("long", lead_seconds=LEAD_SECONDS))
                await stage.act("feed", "long")
            else:
                await stage.hand_over(bytes_of("long"))
            await stage.finish()
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert seen["sounds"][0]["playedSeconds"] > 0
    assert seen["playedNow"][0] == seen["sounds"][0]["playedSeconds"]


@pytest.mark.parametrize("cut", ["stop", "mute"])
def test_a_cut_sounds_read_stays_at_the_seconds_its_end_reports(
    tmp_path: Path, cut: str
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.prepare_bytes("long", LONG)
            await stage.hand_over(fed("long", lead_seconds=LEAD_SECONDS))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.act("feed", "long")
            if cut == "stop":
                await stage.act("stop")
            else:
                await stage.set_muted(muted=True)
            await stage.finish()
            after_end = await stage.snapshot()
            await stage.settle()
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
        async with activated(tmp_path) as stage:
            await stage.prepare_bytes("long", LONG)
            await stage.hand_over(fed("long", lead_seconds=LEAD_SECONDS))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.act("feed", "long")
            await stage.act("stopThenCutPastItsEdge", cut, PAST_THE_EDGE_SECONDS)
            await stage.finish()
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    reported = seen["sounds"][0]["playedSeconds"]
    assert reported < seen["playedBeforeStop"] + PAST_THE_EDGE_SECONDS / 2
    assert seen["playedNow"][0] == reported
    series = seen["playedSeries"]
    assert all(later >= earlier for earlier, later in itertools.pairwise(series))


def test_a_streamed_sound_that_breaks_off_after_playing_keeps_what_it_played(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(CUT_LATE))
            await stage.until(f"s.sounds[0].peakRms > {AUDIBLE_RMS}")
            await stage.settle()
            before_break = await stage.snapshot()
            stage.side.held_release.set()
            await stage.until("s.sounds[0].endedValue !== null")
            after_break = await stage.snapshot()
            return before_break, after_break

    before_break, after_break = asyncio.run(scenario())

    assert after_break["sounds"][0]["endedValue"] == "failed"
    assert before_break["playedNow"][0] > 0
    assert after_break["playedNow"][0] >= before_break["playedNow"][0]


def test_the_read_never_steps_back_across_a_stops_fade_and_its_end(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.prepare_bytes("long", LONG)
            await stage.hand_over(fed("long", lead_seconds=LEAD_SECONDS))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.act("feed", "long")
            await stage.act("stop")
            await stage.finish()
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    series = seen["playedSeries"]
    assert len(series) > 1
    assert all(later >= earlier for earlier, later in itertools.pairwise(series))


def test_a_sound_muted_at_hand_over_reads_zero_played_seconds(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.set_muted(muted=True)
            await stage.hand_over(url(SHORT))
            await stage.finish()
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert seen["playedNow"][0] == 0
