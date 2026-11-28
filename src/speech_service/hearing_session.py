"""C2's scheduling of a live session's audio: turns and partials.

Pure and synchronous: audio goes in, events and snapshots come out, the detector
says which windows of the audio are speech, and the clock is injected. Decoding,
transport and the item lifecycle belong to the caller.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from speech_service.pcm import BYTES_PER_SAMPLE, HEARING_SAMPLE_RATE
from speech_service.voice_activity import (
    PRE_ROLL_SECONDS,
    START_THRESHOLD,
    START_WINDOWS,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    is_voiced,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from speech_service.voice_activity import SpeechDetector

PARTIAL_PERIOD_SECONDS: Final = 0.4
MIN_AUDIO_SECONDS: Final = 0.4
POLL_WAIT_SECONDS: Final = 0.3
MAX_TURN_SECONDS: Final = 30.0
DEFAULT_SUSTAINED_SPEECH_MS: Final = 400
SUSTAINED_PAUSE_SECONDS: Final = 0.15
_SAMPLES_PER_MILLISECOND: Final = HEARING_SAMPLE_RATE // 1000
_MAX_TURN_SAMPLES: Final = int(MAX_TURN_SECONDS * HEARING_SAMPLE_RATE)
_MIN_PARTIAL_SAMPLES: Final = int(MIN_AUDIO_SECONDS * HEARING_SAMPLE_RATE)
_PRE_ROLL_SAMPLES: Final = int(PRE_ROLL_SECONDS * HEARING_SAMPLE_RATE)


@dataclass(frozen=True, slots=True)
class SpeechStart:
    """C2 first heard speech in the open turn, `audio_ms` into the session."""

    audio_ms: int


@dataclass(frozen=True, slots=True)
class SpeechSustain:
    """The open turn's voice has lasted the duration, `audio_ms` into the session.

    `audio_ms` is the end of the window that completed the count.
    """

    audio_ms: int


@dataclass(frozen=True, slots=True)
class SpeechSustainFrom:
    """The voice at the moment `from_ms` has lasted the duration from there on.

    `audio_ms` is the end of the window that completed the count.
    """

    from_ms: int
    audio_ms: int


class TurnEndedBy(StrEnum):
    """What ended a turn: a client commit, a drain cut, the length cap or silence."""

    CLIENT = "client"
    DRAIN = "drain"
    CAP = "cap"
    SILENCE = "silence"


@dataclass(frozen=True, slots=True)
class TurnEnd:
    """A turn's 16 kHz audio, where it lay in the session and how much of it was speech.

    The audio starts a pre-roll before the speech; the milliseconds count the
    session's audio, the dropped audio between turns included. `silence_ms` is
    the silence that ended the turn, `0` for a turn another end cut.
    """

    pcm: bytes
    start_ms: int
    end_ms: int
    voiced_seconds: float
    ended_by: TurnEndedBy
    silence_ms: int = 0


type SchedulerEvent = SpeechStart | SpeechSustain | SpeechSustainFrom | TurnEnd


@dataclass(slots=True)
class _Moment:
    """A moment the voice is counted from.

    `base` is the item's count when the moment's item was decided, `None` before;
    `spent` is set once the moment can yield nothing more.
    """

    from_ms: int
    base: int | None = None
    spent: bool = False

    @property
    def sample(self) -> int:
        return self.from_ms * _SAMPLES_PER_MILLISECOND


@dataclass(frozen=True, slots=True)
class OpenTurn:
    """The open turn's audio so far and how much of it was speech."""

    pcm: bytes
    voiced_seconds: float


def _pause_windows(pause_seconds: float) -> int:
    """The run of unvoiced windows that restarts the count: a pause past the bound."""
    return int(pause_seconds * HEARING_SAMPLE_RATE) // WINDOW_SAMPLES + 1


def _voiced_since_last_pause(
    starts: deque[int], pause_windows: int, *, from_sample: int = 0
) -> int:
    """How many voiced windows, by their starts, follow the last long pause.

    Only windows that begin at or after `from_sample` count.
    """
    count = 0
    newer: int | None = None
    for start in reversed(starts):
        if start < from_sample:
            break
        if newer is not None and (newer - start) // WINDOW_SAMPLES - 1 >= pause_windows:
            break
        count += 1
        newer = start
    return count


class HearingSession:
    """One session's open turn, and the schedule of its partials.

    A turn opens once `START_WINDOWS` windows in a row are speech and ends after
    `silence_seconds` of windows that are not, or at the cap. Between turns only the
    last pre-roll of audio is held; its position stays counted.
    """

    def __init__(
        self,
        clock: Callable[[], float],
        detector: SpeechDetector,
        silence_seconds: float,
        *,
        pause_seconds: float = SUSTAINED_PAUSE_SECONDS,
    ) -> None:
        """Count time with `clock`, in seconds; read the speech from `detector`.

        A turn ends after `silence_seconds` without speech. A pause longer than
        `pause_seconds` restarts the sustained count.
        """
        self._clock = clock
        self._detector = detector
        self._pause_windows = _pause_windows(pause_seconds)
        self._silent_windows_to_end = math.ceil(
            silence_seconds * HEARING_SAMPLE_RATE / WINDOW_SAMPLES
        )
        self._pcm = bytearray()
        self._held_from = 0
        self._windows = 0
        self._speaking = False
        self._speech_windows = 0
        self._run_start = 0
        self._silent_windows = 0
        self._voiced_windows = 0
        self._sustained_windows_to_send: int | None = None
        self._sustained_windows = 0
        self._sustained_sent = False
        self._moment: _Moment | None = None
        self._waiting_voiced: deque[int] = deque()
        self._last_partial_at: float | None = None
        self._last_partial_samples = 0
        self._last_turn = False

    def sustain_after(self, duration_ms: int) -> None:
        """Raise `SpeechSustain` once a turn's voice has lasted `duration_ms`.

        The voice that counts is the voiced windows since the turn's last pause
        longer than the session's pause bound, its pre-roll's included. Call it before
        the first `add`.
        """
        self._sustained_windows_to_send = math.ceil(
            duration_ms * _SAMPLES_PER_MILLISECOND / WINDOW_SAMPLES
        )

    @property
    def sustains(self) -> bool:
        """Whether `sustain_after` asked for sustained speech."""
        return self._sustained_windows_to_send is not None

    def sustain_from(self, from_ms: int) -> None:
        """Raise `SpeechSustainFrom` once the item at `from_ms` has lasted the duration.

        The item is the one that began at or before `from_ms` and had not ended before
        it; only the windows that begin at or after `from_ms` count, under the rule
        of `sustain_after`. A moment replaces the standing one.
        """
        self._moment = _Moment(from_ms)

    def add(self, pcm: bytes) -> list[SchedulerEvent]:
        """Take 16 kHz mono 16-bit audio; a turn that reaches the cap ends there."""
        self._pcm.extend(pcm)
        events: list[SchedulerEvent] = []
        for probability in self._detector.probabilities(pcm):
            events += self._take_window(probability)
        return events

    def commit(self) -> TurnEnd | None:
        """End the open turn at the client's word; `None` when it holds no audio.

        After `stop_opening_turns` only a turn that is open can end: audio kept
        since the last turn is not a turn.
        """
        if self._last_turn and not self.turn_open:
            return None
        ended = self._end(self._held_end, TurnEndedBy.CLIENT)
        return ended[0] if ended else None

    @property
    def turn_open(self) -> bool:
        """Whether a turn is open or its first speech windows are being confirmed."""
        return self._speaking or self._speech_windows > 0

    def stop_opening_turns(self) -> None:
        """Open no turn after this call; a turn already open or confirming runs on."""
        self._last_turn = True

    def cut(self) -> TurnEnd | None:
        """End the open turn for a drain; `None` when it was still being confirmed."""
        if not self._speaking:
            self._speech_windows = 0
            return None
        return self._end(self._held_end, TurnEndedBy.DRAIN)[0]

    @property
    def silent(self) -> bool:
        """Whether the open turn's latest windows are silence, the wait for its end."""
        return self._speaking and self._silent_windows > 0

    def partial_due(self) -> OpenTurn | None:
        """The open turn's audio when a partial is due, else `None`."""
        samples = len(self._pcm) // BYTES_PER_SAMPLE
        if not self._speaking or samples < _MIN_PARTIAL_SAMPLES:
            return None
        if samples == self._last_partial_samples:
            return None
        last = self._last_partial_at
        if last is not None and self._clock() - last < PARTIAL_PERIOD_SECONDS:
            return None
        return OpenTurn(bytes(self._pcm), self._voiced_windows * WINDOW_SECONDS)

    def partial_started(self) -> None:
        """Note that the caller decoded the audio `partial_due` gave."""
        self._last_partial_at = self._clock()
        self._last_partial_samples = len(self._pcm) // BYTES_PER_SAMPLE

    @property
    def _turn_start(self) -> int:
        """Where the turn being confirmed or open starts, in samples."""
        return max(self._run_start, self._held_from)

    @property
    def _held_end(self) -> int:
        return self._held_from + len(self._pcm) // BYTES_PER_SAMPLE

    def _take_window(self, probability: float) -> list[SchedulerEvent]:
        start = self._windows * WINDOW_SAMPLES
        self._windows += 1
        if self._speaking:
            self._reach_moment(start)
            return self._continue_turn(probability, start + WINDOW_SAMPLES)
        if is_voiced(probability):
            self._waiting_voiced.append(start)
        return self._wait_for_speech(probability, start)

    def _wait_for_speech(self, probability: float, start: int) -> list[SchedulerEvent]:
        if probability < START_THRESHOLD or (
            self._last_turn and not self._speech_windows
        ):
            self._speech_windows = 0
        else:
            if not self._speech_windows:
                self._run_start = start
            self._speech_windows += 1
            if self._speech_windows >= START_WINDOWS:
                return self._open_turn(start + WINDOW_SAMPLES)
        self._pass_moment_without_item(start)
        newest = self._run_start if self._speech_windows else start + WINDOW_SAMPLES
        self._drop_before(newest - _PRE_ROLL_SAMPLES)
        return []

    def _open_turn(self, end: int) -> list[SchedulerEvent]:
        self._drop_before(self._run_start - _PRE_ROLL_SAMPLES)
        self._speaking = True
        self._voiced_windows = len(self._waiting_voiced)
        self._sustained_windows = _voiced_since_last_pause(
            self._waiting_voiced, self._pause_windows
        )
        self._decide_moment_of_opening_turn(end - WINDOW_SAMPLES)
        self._waiting_voiced.clear()
        self._silent_windows = 0
        self._speech_windows = 0
        return [
            SpeechStart(audio_ms=self._turn_start // _SAMPLES_PER_MILLISECOND),
            *self._sustained_when_reached(end),
            *self._moment_when_reached(end),
        ]

    def _continue_turn(self, probability: float, end: int) -> list[SchedulerEvent]:
        if is_voiced(probability):
            if self._silent_windows >= self._pause_windows:
                self._sustained_windows = 0
                if self._moment is not None and self._moment.base is not None:
                    self._moment.base = 0
            self._sustained_windows += 1
            self._voiced_windows += 1
            self._silent_windows = 0
        else:
            self._silent_windows += 1
        if end - self._held_from >= _MAX_TURN_SAMPLES:
            return self._end(self._held_from + _MAX_TURN_SAMPLES, TurnEndedBy.CAP)
        if self._silent_windows >= self._silent_windows_to_end:
            silence_ms = (
                self._silent_windows * WINDOW_SAMPLES // _SAMPLES_PER_MILLISECOND
            )
            return self._end(end, TurnEndedBy.SILENCE, silence_ms=silence_ms)
        return [*self._sustained_when_reached(end), *self._moment_when_reached(end)]

    def _sustained_when_reached(self, end: int) -> list[SpeechSustain]:
        """The turn's one event, once the window ending at `end` completes the count."""
        if (
            self._sustained_windows_to_send is None
            or self._sustained_sent
            or self._sustained_windows < self._sustained_windows_to_send
        ):
            return []
        self._sustained_sent = True
        return [SpeechSustain(audio_ms=end // _SAMPLES_PER_MILLISECOND)]

    def _undecided_moment(self, start: int) -> _Moment | None:
        """The moment when it is undecided and the window at `start` reaches it."""
        moment = self._moment
        if moment is None or moment.spent or moment.base is not None:
            return None
        return moment if start >= moment.sample else None

    def _reach_moment(self, start: int) -> None:
        """The audio reaches the moment while an item is open: its item is that one."""
        if (moment := self._undecided_moment(start)) is not None:
            moment.base = self._sustained_windows

    def _decide_moment_of_opening_turn(self, start: int) -> None:
        """The audio is at the moment as a turn opens: count what begins at or after it.

        A turn that opens before the moment is decided when the audio reaches it.
        """
        if (moment := self._undecided_moment(start)) is not None:
            moment.base = self._sustained_windows - _voiced_since_last_pause(
                self._waiting_voiced, self._pause_windows, from_sample=moment.sample
            )

    def _pass_moment_without_item(self, start: int) -> None:
        """The audio passes the moment with no item that began at or before it."""
        moment = self._undecided_moment(start)
        if moment is not None and not (
            self._speech_windows
            and self._turn_start // _SAMPLES_PER_MILLISECOND <= moment.from_ms
        ):
            moment.spent = True

    def _moment_when_reached(self, end: int) -> list[SpeechSustainFrom]:
        """The moment's one event, once the window ending at `end` completes it."""
        moment = self._moment
        if (
            moment is None
            or moment.spent
            or moment.base is None
            or self._sustained_windows_to_send is None
            or self._sustained_windows - moment.base < self._sustained_windows_to_send
        ):
            return []
        moment.spent = True
        return [SpeechSustainFrom(moment.from_ms, end // _SAMPLES_PER_MILLISECOND)]

    def _drop_before(self, sample: int) -> None:
        dropped = min(sample, self._held_end) - self._held_from
        if dropped > 0:
            del self._pcm[: dropped * BYTES_PER_SAMPLE]
            self._held_from += dropped
        while self._waiting_voiced and self._waiting_voiced[0] < self._held_from:
            self._waiting_voiced.popleft()

    def _end(
        self,
        end: int,
        ended_by: TurnEndedBy,
        *,
        silence_ms: int = 0,
    ) -> list[TurnEnd]:
        start = self._held_from
        size = (end - start) * BYTES_PER_SAMPLE
        pcm = bytes(self._pcm[:size])
        voiced_seconds = (
            self._voiced_windows + len(self._waiting_voiced)
        ) * WINDOW_SECONDS
        del self._pcm[:size]
        self._held_from = end
        self._speaking = False
        self._speech_windows = 0
        self._silent_windows = 0
        self._voiced_windows = 0
        self._sustained_sent = False
        if self._moment is not None and self._moment.base is not None:
            self._moment.spent = True
        self._waiting_voiced.clear()
        self._last_partial_at = None
        self._last_partial_samples = 0
        if not pcm:
            return []
        return [
            TurnEnd(
                pcm,
                start // _SAMPLES_PER_MILLISECOND,
                end // _SAMPLES_PER_MILLISECOND,
                voiced_seconds,
                ended_by,
                silence_ms,
            )
        ]
