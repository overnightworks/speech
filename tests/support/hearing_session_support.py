"""Hearing-session scenario helpers: a clock and windows of loud or quiet audio."""

from __future__ import annotations

import struct

from speech_service.config import DEFAULT_LIVE_SILENCE_SECONDS
from speech_service.hearing_session import HearingSession, SpeechStart, TurnEnd
from speech_service.pcm import HEARING_SAMPLE_RATE as SAMPLE_RATE
from speech_service.voice_activity import WINDOW_SAMPLES, WINDOW_SECONDS
from tests.support.detector_support import LOUD_LEVEL, LevelDetector

SILENT_WINDOWS = round(DEFAULT_LIVE_SILENCE_SECONDS / WINDOW_SECONDS + 0.5)


class Clock:
    """A clock the test sets by assigning `now`."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """The time now."""
        return self.now


def windows(count: int, level: int) -> bytes:
    """`count` windows of PCM at one level."""
    return struct.pack("<h", level) * (count * WINDOW_SAMPLES)


def seconds(duration: float, level: int) -> bytes:
    """`duration` seconds of PCM at one level."""
    return struct.pack("<h", level) * int(duration * SAMPLE_RATE)


def speech(count: int) -> bytes:
    """`count` windows loud enough for the level detector to call speech."""
    return windows(count, LOUD_LEVEL)


def quiet(count: int) -> bytes:
    """`count` windows of silence."""
    return windows(count, 0)


def level_session() -> tuple[HearingSession, Clock]:
    """A session on the level detector with the default silence, and its clock."""
    clock = Clock()
    return HearingSession(clock, LevelDetector(), DEFAULT_LIVE_SILENCE_SECONDS), clock


def ends(events: list[SpeechStart | TurnEnd]) -> list[TurnEnd]:
    """The turn ends among `events`."""
    return [event for event in events if isinstance(event, TurnEnd)]
