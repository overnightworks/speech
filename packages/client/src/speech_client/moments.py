"""The moments an app names on a hearing's audio timeline, and how they travel.

`SustainMoments` is the handle an app passes to `hear_utterances`; `MomentLedger` is
what a `HearingSession` keeps of the newest moment: where it lies in the audio queued,
whether a wire has carried it, and which events of it are delivered (WIRE-176 to
WIRE-178).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from speech_client.realtime_wire import (
    ServerEventError,
    SpeechSustainedFrom,
    sustain_from,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class SustainMoments:
    """Names the moment sustained speech is counted from, for one hearing at a time."""

    def __init__(self) -> None:
        """Start unbound: no hearing runs."""
        self._name: Callable[[], int | None] | None = None

    def now(self) -> int | None:
        """Name a moment at the audio handed to the hearing so far.

        Returns its position on the hearing's audio timeline, in whole milliseconds,
        rounded down; the event of the voice that lasts from it carries no position of
        its own, since the library delivers only the newest moment's. Returns `None`
        and names nothing while no hearing runs.
        """
        return None if self._name is None else self._name()

    @contextlib.contextmanager
    def running(self, name: Callable[[], int | None]) -> Iterator[None]:
        """Name moments through `name` while the hearing runs."""
        self._name = name
        try:
            yield
        finally:
            self._name = None


@dataclass(frozen=True, slots=True)
class _Moment:
    position_ms: int
    at_bytes: int


class MomentLedger:
    """The newest moment of a hearing: queued, written on the wire, and its events."""

    def __init__(self) -> None:
        """Start with no moment."""
        self._newest: _Moment | None = None
        self._written_on_wire = False
        self._ever_written = False

    def name(self, at_bytes: int, position_ms: int) -> None:
        """Make a moment at `position_ms`, after `at_bytes` of queued audio, the newest.

        A moment at the newest one's position is neither queued nor written again.
        """
        if self._newest is None or self._newest.position_ms != position_ms:
            self._newest = _Moment(position_ms, at_bytes)
            self._written_on_wire = False

    def new_wire(self) -> None:
        """A wire session begins: the newest moment is not yet written on it."""
        self._written_on_wire = False

    def take_due(self, written_bytes: int, offset_ms: int) -> str | None:
        """The message to write now on the wire that starts at `offset_ms`, or `None`.

        The newest moment is due once the audio before it is written and it is not
        below the wire's start: a moment below it lies before every item of that wire.
        """
        moment = self._newest
        if (
            moment is None
            or self._written_on_wire
            or moment.at_bytes > written_bytes
            or moment.position_ms < offset_ms
        ):
            return None
        self._written_on_wire = self._ever_written = True
        return sustain_from(moment.position_ms - offset_ms)

    def on_timeline(
        self, event: SpeechSustainedFrom, offset_ms: int
    ) -> SpeechSustainedFrom | None:
        """The event on the hearing's timeline, or `None` for one of an older moment.

        Raises `ServerEventError` when no moment was ever written.
        """
        if self._newest is None or not self._ever_written:
            raise ServerEventError
        position = event.from_ms + offset_ms
        if position != self._newest.position_ms:
            return None
        return event.model_copy(
            update={"from_ms": position, "audio_end_ms": event.audio_end_ms + offset_ms}
        )
