"""A live session that asks for sustained speech, and the audio to feed it."""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

from tests.support.clock_support import FakeClock
from tests.support.detector_support import FOUR_TENTHS_MS
from tests.support.hear_support import FakeHearing, loaded
from tests.support.live_support import (
    SPEECH_LEVEL,
    FakeSocket,
    started,
    update,
    windows,
)

if TYPE_CHECKING:
    from speech_service.hearing import Hearing
    from speech_service.live_session import LiveSession
    from speech_service.sound_events import SoundTaggers

FOUR_TENTHS = {"duration_ms": FOUR_TENTHS_MS}
LEVELS = {"v": SPEECH_LEVEL, ".": 0}


def voice(pattern: str) -> str:
    """The audio of `pattern`, one window a character: `v` voiced, `.` silent.

    A silent window follows: the resampler holds back the last samples it is fed.
    """
    return "".join(
        windows(len(list(run)), LEVELS[character])
        for character, run in groupby(pattern + ".")
    )


async def asking(
    ask: object = FOUR_TENTHS,
    engine: FakeHearing | None = None,
    *,
    sounds: SoundTaggers | None = None,
    sound_detection: dict[str, str] | None = None,
) -> tuple[LiveSession, FakeSocket, Hearing, FakeClock]:
    """A session over `engine` that asked for sustained speech with `ask`."""
    clock = FakeClock()
    hearing = loaded(engine or FakeHearing(), clock, sounds=sounds)
    session, socket = await started(hearing)
    await session.on_message(
        update(sound_detection=sound_detection, sustained_speech=ask)
    )
    return session, socket, hearing, clock
