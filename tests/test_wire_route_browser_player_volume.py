"""An app turns the player down or mutes it and reads each sound's played seconds.

The flows run in Chromium on the stand-in app's page; see the shared stage.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from tests.support.browser_player_support import (
    AUDIBLE_RMS,
    DUCKED,
    LONG,
    SHORT,
    SOUND_SECONDS,
    STREAMED,
    activated,
    ends,
    url,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.browser


def test_a_sound_turned_down_keeps_playing_and_comes_back_at_full_volume(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG))
            await stage.until(f"s.sounds[0].peakRms > {AUDIBLE_RMS}")
            await stage.set_volume(DUCKED)
            await stage.until(f"s.gainValue < {DUCKED} + 0.01")
            await stage.settle()
            ducked = await stage.snapshot()
            await stage.set_volume(1)
            await stage.until("s.gainValue > 0.99")
            return ducked, await stage.finish()

    ducked, seen = asyncio.run(scenario())

    assert ends(ducked) == [None]
    assert ducked["gainValue"] == pytest.approx(DUCKED, abs=0.01)
    assert ends(seen) == ["played"]


def test_volume_zero_ends_no_sound_and_muting_over_it_ends_the_sound(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG))
            await stage.until(f"s.sounds[0].peakRms > {AUDIBLE_RMS}")
            await stage.set_volume(0)
            await stage.settle()
            silent = await stage.snapshot()
            await stage.set_muted(muted=True)
            return silent, await stage.finish()

    silent, seen = asyncio.run(scenario())

    assert ends(silent) == [None]
    assert ends(seen) == ["muted"]


def test_mute_wins_over_the_volume_and_unmuting_brings_back_the_last_volume(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[float, float]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.set_muted(muted=True)
            await stage.set_volume(DUCKED)
            muted = (await stage.snapshot())["gainValue"]
            await stage.set_muted(muted=False)
            await stage.hand_over(url(LONG))
            await stage.until("s.sounds[1].startedValue === 'playing'")
            await stage.until(f"Math.abs(s.gainValue - {DUCKED}) < 0.01")
            return muted, (await stage.snapshot())["gainValue"]

    muted, back = asyncio.run(scenario())

    assert muted == 0
    assert back == pytest.approx(DUCKED, abs=0.01)


@pytest.mark.parametrize("level", [-0.1, 1.1, "NaN"])
def test_a_volume_outside_zero_to_one_throws_a_range_error_and_changes_nothing(
    tmp_path: Path, level: float | str
) -> None:
    async def scenario() -> tuple[object, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.set_volume(DUCKED)
            await stage.until(f"s.gainValue < {DUCKED} + 0.01")
            thrown = await stage.page.evaluate(
                "level => window.playerPage.volumeError(level === 'NaN' ? NaN : level)",
                level,
            )
            await stage.settle()
            return thrown, (await stage.snapshot())["gainValue"]

    thrown, gain = asyncio.run(scenario())

    assert thrown == "RangeError"
    assert gain == pytest.approx(DUCKED, abs=0.01)


def test_a_sound_played_through_reports_its_full_length_as_played_seconds(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(
        SOUND_SECONDS["short.wav"], abs=seen["renderQuantum"]
    )


@pytest.mark.parametrize("cut", ["stop", "mute"])
@pytest.mark.parametrize("sound", [LONG, STREAMED], ids=["whole", "streamed"])
def test_a_cut_sound_reports_the_seconds_the_context_clock_passed_since_it_started(
    tmp_path: Path, cut: str, sound: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(sound), url(SHORT))
            await stage.until(f"s.sounds[0].peakRms > {AUDIBLE_RMS}")
            await stage.settle()
            if cut == "stop":
                await stage.act("stop")
            else:
                await stage.set_muted(muted=True)
            return await stage.finish()

    seen = asyncio.run(scenario())

    cut_sound, waiting = seen["sounds"]
    elapsed = cut_sound["contextAtEnd"] - cut_sound["contextAtStart"]
    assert elapsed > 0
    assert cut_sound["playedSeconds"] == pytest.approx(
        elapsed, abs=seen["renderQuantum"]
    )
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
