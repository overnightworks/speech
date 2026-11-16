"""The live hearing session: `WS /v1/realtime`, bounded in time and memory.

`HearingSession` is an async context manager. It starts opening the connection
on entry, takes every input from its call site, and never logs, and keeps its
socket's logger above debug; no outcome and no repr carries a token or heard text.

- The tokens come from a `TokenSource`, an async callable with no arguments that
  returns `SessionTokens` or `None`. The session awaits it before every opening,
  reconnects included, within that opening's bound: the service checks tokens
  only at the upgrade (WIRE-140), so each opening presents the source's current
  ones. `None` ends the session with `NoToken`. An `Exception` the source
  raises ends the session at once, nothing more is opened and `outcome` stays
  `None`; leaving the context raises it. A bound that expires while the source
  runs ends the session as that bound says.
- `traceparent` (default: the trace running when the session is built) goes on
  every upgrade as a header and parents only the service's own lines of the
  session, never a turn: a turn's trace comes back in its `transcript.completed`.
- `send(pcm)` queues PCM16 audio at 24 kHz. Audio queued before a session is
  open, or faster than it is written, is held up to `held_limit_bytes`; one byte
  over ends the session with `OverCap`, at any moment. A chunk leaves the held
  audio when its write begins, and audio is written only after the session's
  `session.created`. `NEEDED_HOLD_BYTES` is the limit that covers the opening
  bound and a whole reconnect. There is no commit: the service ends turns
  itself (WIRE-65), and a commit with no audio is refused (WIRE-64).
- `events()` yields the typed server events until the session ends, and the
  client-side `Reconnected` between two wire sessions.
- `outcome` is `None` while the session runs and one `HearingOutcome` after
  `events()` ran out, a closed union a caller matches with `assert_never`:
  - `Closed`: the connection ended in order, by the service or by leaving the
    context.
  - `Refused`: the upgrade was answered with an HTTP refusal.
  - `Busy`: the refusal is `busy`, with the wait the service asks for, as for
    `speak`; a `busy` without a whole-second `Retry-After` is `Malformed`.
  - `OverCap`: the held audio would pass `held_limit_bytes`.
  - `Malformed`: a server message is not an event the wire names, the first
    event of a session is not `session.created` (WIRE-68), or the upgrade was
    answered off the wire, a redirect included: none is followed, so no token
    travels to another origin.
  - `NoToken`: the token source returned `None`.
  - `ReconnectFailed`: the reconnect bound ran out.
  - `Cut`, `PastDeadline` and `Unreachable`, as for `speak`
    (`speech_client.outcomes`); `deadline_seconds` bounds the whole
    hearing, reopenings included. `opening_seconds` (default `OPENING_SECONDS`)
    bounds each opening: the token source, the handshake and the wait for the
    first event, which the service sends first. A service that does not speak by
    then ends the opening with `Unreachable`; a strictly smaller deadline wins.

Once the hearing has had one successful opening, a session that closes with
`DRAIN_CLOSE_CODE` or is cut after its `session.created`, having sent no `error`
and no `failed` item, is reopened at the same origin with the same
`session.update`: `Reconnected` reports why and which stretch of audio went
unheard, and the new session's items start afresh. Any other end, and any end
before the first success, ends the hearing as it did. `RECONNECT_BOUND` bounds
the reopening from the end of the open session; a session that answered resets
it. A session answers with an event about the audio or a transcript after its
`session.created`, `session.updated` aside. A refusal `unavailable` or `busy`
(its `Retry-After` is not waited for), `Unreachable`, a cut before
`session.created`, an expired opening bound and a session that ended without
answering each count as one failed attempt, and the bound runs on from the end
of the first session that ended unanswered; any other refusal or `Malformed`
ends the hearing with that outcome.

The positions `events()` yields (`audio_start_ms`, `audio_end_ms`) lie on one
audio timeline across reconnects: the service's position plus the audio of every
chunk whose write began on an earlier session, in whole milliseconds, rounded
down. A chunk whose write began on a session that ended is lost, never written
again, and still counted.

A failure outside these outcomes ends the session at once; `outcome` stays
`None` and leaving the context raises the failure.

A caller reads `events()` while it sends: the session waits for the caller to
take an event before it reads the next one.

Leaving the context in order closes the connection with the normal code, within
the socket's close timeout, at the deadline or, before the first event, at the
opening bound, which then keeps the end's outcome. Leaving under cancellation, or a
cancel while leaving, aborts the connection at once, without a close handshake.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from http import HTTPStatus
from typing import TYPE_CHECKING, Final, Self

from pydantic import ValidationError
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import (
    ConnectionClosed,
    InvalidHandshake,
    InvalidMessage,
    InvalidStatus,
)

from speech_client._traceparent import traceparent_headers
from speech_client._transport import base_origin
from speech_client.outcomes import (
    Busy,
    Cut,
    Malformed,
    OverCap,
    PastDeadline,
    RetryAfterError,
    Unreachable,
    busy_wait,
)
from speech_client.realtime_wire import (
    AudioAppend,
    ErrorEvent,
    PcmFormat,
    ServerEventError,
    SessionAudio,
    SessionCreated,
    SessionInput,
    SessionUpdate,
    SessionUpdated,
    SpeechStarted,
    SpeechStopped,
    TranscriptCompleted,
    TranscriptFailed,
    TranscriptionSession,
    TranscriptionSetting,
    parse_server_event,
)
from speech_client.wire import (
    DRAIN_CLOSE_CODE,
    REALTIME_INTENT,
    REALTIME_INTENT_PARAM,
    REALTIME_PATH,
    Language,
    Refusal,
    RefusalCode,
    token_headers,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from types import TracebackType

    from speech_client.realtime_wire import ServerEvent


@dataclass(frozen=True, slots=True)
class Closed:
    """The connection ended in order."""


@dataclass(frozen=True, slots=True)
class Refused:
    """The upgrade was refused over HTTP; `code` is `None` off the contract."""

    status: int
    code: RefusalCode | None


@dataclass(frozen=True, slots=True)
class NoToken:
    """The token source returned `None`; nothing more was opened."""


@dataclass(frozen=True, slots=True)
class ReconnectFailed:
    """The reconnect bound ran out before a session opened."""


type HearingOutcome = (
    Closed
    | Refused
    | Busy
    | OverCap
    | Malformed
    | NoToken
    | ReconnectFailed
    | PastDeadline
    | Cut
    | Unreachable
)


@dataclass(frozen=True, slots=True)
class SessionTokens:
    """The two tokens of one opening; neither is shown in a repr."""

    workload_token: str = field(repr=False)
    person_token: str = field(repr=False)


type TokenSource = Callable[[], Awaitable[SessionTokens | None]]


@dataclass(frozen=True, slots=True)
class ReconnectBound:
    """How long and how often the session tries to reopen."""

    attempts: int
    pause_seconds: float
    within_seconds: float


class ReconnectCause(StrEnum):
    """Why a session was reopened."""

    DRAIN = "drain"
    CUT = "cut"


@dataclass(frozen=True, slots=True)
class Reconnected:
    """A new session replaced one that ended; client side, not a wire event.

    The audio from `unheard_start_ms` to `unheard_end_ms`, on the session's
    audio timeline, may not have been heard: it follows the last turn the old
    session delivered and precedes the new session's first position.
    """

    cause: ReconnectCause
    unheard_start_ms: int
    unheard_end_ms: int


OPENING_SECONDS: Final = 5.0
RECONNECT_BOUND: Final = ReconnectBound(
    attempts=5, pause_seconds=0.5, within_seconds=10
)
_BYTES_PER_SECOND: Final = (
    SessionInput().format.rate * PcmFormat.sample_bytes * PcmFormat.channels
)
NEEDED_HOLD_BYTES: Final = int(
    max(OPENING_SECONDS, RECONNECT_BOUND.within_seconds) * _BYTES_PER_SECOND
)

# The socket logs request headers and frame texts at debug; this logger never
# passes them on unless an app raises it on purpose.
_SOCKET_LOGGER = logging.getLogger(f"{__name__}.socket")
_SOCKET_LOGGER.setLevel(logging.WARNING)


class _Connect(connect):
    """A connection that follows no redirect."""

    def process_redirect(self, exc: Exception) -> Exception:
        return exc


class _SourceFailedError(Exception):
    """Carries an exception of the token source past the outcome mapping."""

    def __init__(self, original: Exception) -> None:
        super().__init__()
        self.original = original


@dataclass(slots=True)
class _Interrupted:
    """The reconnect under way: why, from when, and how many attempts failed."""

    cause: ReconnectCause
    unheard_start_ms: int
    at: float
    failed_attempts: int = 0


@dataclass(slots=True)
class _Wire:
    """One wire session: where its audio starts on the timeline and how it went."""

    offset_ms: int
    last_turn_end_ms: int
    opened: asyncio.Event = field(default_factory=asyncio.Event)
    close_code: int | None = None
    failed_sent: bool = False
    stopped: dict[str, int] = field(default_factory=dict)
    ended: HearingOutcome | None = None
    opening: asyncio.Timeout | None = None
    carried: _Interrupted | None = None
    answered: bool = False


def realtime_url(base_url: str) -> str:
    """The Realtime transcription WebSocket of the service at `base_url`."""
    return (
        f"ws{base_url.removeprefix('http')}{REALTIME_PATH}"
        f"?{REALTIME_INTENT_PARAM}={REALTIME_INTENT}"
    )


def session_update(languages: tuple[Language, ...]) -> str:
    """The one `session.update` that sets the languages of a session."""
    update = SessionUpdate(
        type="session.update",
        session=TranscriptionSession(
            type="transcription",
            audio=SessionAudio(
                input=SessionInput(
                    transcription=TranscriptionSetting(languages=languages)
                )
            ),
        ),
    )
    return update.model_dump_json(exclude_none=True)


def _refusal_code(body: bytes) -> RefusalCode | None:
    try:
        return Refusal.model_validate_json(body).error.code
    except ValidationError:
        return None


# In order: a timeout is an `OSError` too. A dropped handshake is the one
# `InvalidMessage` that is `Cut`; `_wire_session` handles it first.
_FAILURE_OUTCOMES: tuple[tuple[type[Exception], HearingOutcome], ...] = (
    (TimeoutError, PastDeadline()),
    (InvalidHandshake, Malformed()),
    (ConnectionClosed, Cut()),
    (ServerEventError, Malformed()),
    (OSError, Unreachable()),
)
_FAILURES = tuple(kind for kind, _ in _FAILURE_OUTCOMES)


def _failure_outcome(failure: BaseException) -> HearingOutcome:
    return next(
        outcome for kind, outcome in _FAILURE_OUTCOMES if isinstance(failure, kind)
    )


def _timeout_outcome(
    timed_out: TimeoutError, ended: HearingOutcome | None, opening: asyncio.Timeout
) -> HearingOutcome:
    # A bound that cuts the polite close of an end keeps that end.
    if ended is not None:
        return ended
    return Unreachable() if opening.expired() else _failure_outcome(timed_out)


def accept_first_event(first_event: ServerEvent, *, bound_expired: bool) -> None:
    """Accept the first event of an attempt as the opening of its session.

    Raises `ServerEventError` for any first event but `session.created`, and
    `TimeoutError` once the opening bound has expired: that attempt has failed even
    with its `session.created` in hand, since the cancel the bound sent ends it.
    """
    if not isinstance(first_event, SessionCreated):
        raise ServerEventError
    if bound_expired:
        raise TimeoutError


def _left_under_cancellation(exception: BaseException | None) -> bool:
    # A cancel the caller's own code turned into another exception still counts.
    task = asyncio.current_task()
    return isinstance(exception, asyncio.CancelledError) or (
        task is not None and task.cancelling() > 0
    )


async def _settled(*tasks: asyncio.Task[object]) -> None:
    """Cancel `tasks` and wait until each is done, retrieving every result.

    A failure outside the outcome mapping is raised, even over a cancel that
    landed meanwhile, as in a task group; otherwise that cancel is raised again.
    The cancel count is left as it is, for `asyncio.timeout` and the caller.
    """
    for task in tasks:
        task.cancel()
    try:
        await asyncio.wait(tasks)
    except asyncio.CancelledError:
        await _settled(*tasks)
        raise
    failures = [task.exception() for task in tasks if not task.cancelled()]
    for failure in failures:
        if failure is not None and not isinstance(failure, _FAILURES):
            raise failure


async def _abandoned(opening: asyncio.Task[ClientConnection | NoToken]) -> None:
    # A bound or an end that won the race beats what the token source raised.
    try:
        with contextlib.suppress(_SourceFailedError):
            await _settled(opening)
    finally:
        if not opening.cancelled() and opening.exception() is None:
            opened = opening.result()
            if isinstance(opened, ClientConnection):
                opened.transport.abort()


def _refusal_outcome(refused: InvalidStatus) -> HearingOutcome:
    answer = refused.response
    if answer.status_code < HTTPStatus.BAD_REQUEST:
        return Malformed()
    try:
        wait = busy_wait(answer.status_code, answer.headers)
    except RetryAfterError:
        return Malformed()
    if wait is not None:
        return Busy(wait)
    return Refused(answer.status_code, _refusal_code(answer.body))


def _is_failed_attempt(outcome: HearingOutcome) -> bool:
    match outcome:
        case Busy() | Unreachable() | Cut():
            return True
        case Refused(code=RefusalCode.UNAVAILABLE):
            return True
        case _:
            return False


def _milliseconds(audio_bytes: int) -> int:
    return audio_bytes * 1000 // _BYTES_PER_SECOND


class HearingSession:
    """A live hearing session; see the module for its calls and outcomes."""

    def __init__(  # noqa: PLR0913 - one keyword per call-site input
        self,
        base_url: str,
        *,
        token_source: TokenSource,
        languages: tuple[Language, ...],
        deadline_seconds: float,
        held_limit_bytes: int,
        opening_seconds: float = OPENING_SECONDS,
        traceparent: str | None = None,
    ) -> None:
        """Take every input; nothing is opened before `__aenter__`."""
        base_origin(base_url)
        if opening_seconds <= 0:
            message = "opening_seconds must be above 0"
            raise ValueError(message)
        self._url = realtime_url(base_url)
        self._traceparent_headers = traceparent_headers(traceparent)
        self._token_source = token_source
        self._update = session_update(languages)
        self._deadline_seconds = deadline_seconds
        self._opening_seconds = opening_seconds
        self._held_limit_bytes = held_limit_bytes
        self._bound = RECONNECT_BOUND
        self._held: deque[bytes] = deque()
        self._held_bytes = 0
        self._held_changed = asyncio.Event()
        self._written_bytes = 0
        self._wire: _Wire | None = None
        self._interrupted: _Interrupted | None = None
        self._end: asyncio.Future[HearingOutcome] | None = None
        self._received: asyncio.Queue[ServerEvent | Reconnected | None] = asyncio.Queue(
            maxsize=1
        )
        self._runner: asyncio.Task[None] | None = None
        self.outcome: HearingOutcome | None = None

    async def __aenter__(self) -> Self:
        """Start opening the connection."""
        self._end = asyncio.get_running_loop().create_future()
        # Eager, so a cancel cannot end the runner before its first await.
        self._runner = asyncio.create_task(self._run(self._end), eager_start=True)
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the connection and wait until the session is gone."""
        assert self._runner is not None  # noqa: S101 - entered
        assert self._end is not None  # noqa: S101 - entered
        if _left_under_cancellation(exception):
            self._runner.cancel()
        elif not self._end.done():
            self._end.set_result(Closed())
        # A cancel landing while the runner runs is passed to it and raised here; once
        # the runner has finished, asyncio raises it here or at the caller's next
        # await. Nothing absorbs it.
        await self._runner

    def send(self, pcm: bytes) -> None:
        """Queue PCM16 audio; past the held limit the session ends with `OverCap`."""
        assert self._end is not None  # noqa: S101 - entered
        if self._end.done():
            return
        if self._held_bytes + len(pcm) > self._held_limit_bytes:
            self._end.set_result(OverCap())
            return
        self._held.append(pcm)
        self._held_bytes += len(pcm)
        self._held_changed.set()

    async def events(self) -> AsyncIterator[ServerEvent | Reconnected]:
        """The server events, in order, until the session ends."""
        while True:
            if (
                self._runner is not None
                and self._runner.done()
                and self._received.empty()
            ):
                return
            event = await self._received.get()
            if event is None:
                return
            yield event

    async def _run(self, end: asyncio.Future[HearingOutcome]) -> None:
        source_failure: Exception | None = None
        try:
            self.outcome = await self._hear(end)
        except asyncio.CancelledError:
            self.outcome = Closed()
            raise
        except _SourceFailedError as failed:
            source_failure = failed.original
        finally:
            # Wakes a caller waiting in `events()`; when an event is still queued,
            # that caller sees the session ended after taking it.
            with contextlib.suppress(asyncio.QueueFull):
                self._received.put_nowait(None)
        if source_failure is not None:
            raise source_failure

    async def _hear(self, end: asyncio.Future[HearingOutcome]) -> HearingOutcome:
        started = asyncio.get_running_loop().time()
        deadline_at = started + self._deadline_seconds
        try:
            async with asyncio.timeout_at(deadline_at):
                return await self._reopening(end, started)
        except TimeoutError:
            return self._past_deadline(deadline_at)

    def _past_deadline(self, deadline_at: float) -> HearingOutcome:
        wire = self._wire
        if wire is None:
            # A pause ends at the reconnect bound; only a strictly smaller deadline wins
            # against it.
            bound_ran_out = self._opening_at(deadline_at) is None
            return ReconnectFailed() if bound_ran_out else PastDeadline()
        # A deadline that cuts the polite close of an end keeps that end.
        if wire.ended is not None:
            return wire.ended
        # Only a strictly smaller deadline wins against a pending opening bound.
        opening_at = wire.opening.when() if wire.opening is not None else None
        if opening_at is None or opening_at > deadline_at:
            return PastDeadline()
        if self._interrupted is None:
            return Unreachable()
        # A reopening counts the attempt the bound ended, as when it wins alone.
        now = asyncio.get_running_loop().time()
        ended = self._ended_by(Unreachable(), wire, now)
        if ended is None and self._opening_at(now) is None:
            return ReconnectFailed()
        return ended or PastDeadline()

    async def _reopening(
        self, end: asyncio.Future[HearingOutcome], started: float
    ) -> HearingOutcome:
        """Run one wire session after another until the hearing ends."""
        clock = asyncio.get_running_loop().time
        attempt_started = started
        while True:
            if end.done():
                return end.result()
            opening_at = self._opening_at(attempt_started)
            if opening_at is None:
                return ReconnectFailed()
            self._wire = _Wire(
                offset_ms=_milliseconds(self._written_bytes),
                last_turn_end_ms=_milliseconds(self._written_bytes),
            )
            outcome = await self._wire_session(end, opening_at)
            wire, self._wire = self._wire, None
            ended = self._ended_by(outcome, wire, clock())
            if ended is not None:
                return ended
            await self._paused(end, clock())
            attempt_started = clock()

    def _opening_at(self, attempt_started: float) -> float | None:
        """When an opening begun now must be over; `None` once the bound ran out."""
        opening_at = attempt_started + self._opening_seconds
        if self._interrupted is None:
            return opening_at
        bound_ends_at = self._interrupted.at + self._bound.within_seconds
        if attempt_started >= bound_ends_at:
            return None
        return min(opening_at, bound_ends_at)

    def _ended_by(
        self, outcome: HearingOutcome, wire: _Wire, now: float
    ) -> HearingOutcome | None:
        """The outcome that ends the hearing, or `None` when another opening follows."""
        if wire.opened.is_set():
            cause = self._reopening_cause(outcome, wire)
            if cause is None:
                return outcome
            interrupted = self._interruption_by(cause, wire, now)
            if interrupted.failed_attempts >= self._bound.attempts:
                return ReconnectFailed()
            self._interrupted = interrupted
            return None
        interrupted = self._interrupted
        if interrupted is None or not _is_failed_attempt(outcome):
            return outcome
        interrupted.failed_attempts += 1
        if interrupted.failed_attempts >= self._bound.attempts:
            return ReconnectFailed()
        return None

    @staticmethod
    def _interruption_by(
        cause: ReconnectCause, wire: _Wire, now: float
    ) -> _Interrupted:
        """The reconnect a session's end starts, or the one it keeps running.

        A session that answered starts a fresh one. One that did not costs an
        attempt on top of those the reconnect it was opened by had spent, from
        that reconnect's start, so a service that drops every session runs the
        bound out.
        """
        if wire.answered:
            return _Interrupted(cause, wire.last_turn_end_ms, now)
        carried = wire.carried or _Interrupted(cause, wire.last_turn_end_ms, now)
        return _Interrupted(
            cause, carried.unheard_start_ms, carried.at, carried.failed_attempts + 1
        )

    async def _paused(self, end: asyncio.Future[HearingOutcome], now: float) -> None:
        """Wait the pause after a failed attempt, or until the hearing ends."""
        interrupted = self._interrupted
        if interrupted is None or interrupted.failed_attempts == 0:
            return
        bound_ends_in = interrupted.at + self._bound.within_seconds - now
        pause = min(self._bound.pause_seconds, bound_ends_in)
        await asyncio.wait({end}, timeout=max(pause, 0))

    @staticmethod
    def _reopening_cause(outcome: HearingOutcome, wire: _Wire) -> ReconnectCause | None:
        if not isinstance(outcome, Cut) or wire.failed_sent:
            return None
        if wire.close_code == DRAIN_CLOSE_CODE:
            return ReconnectCause.DRAIN
        return ReconnectCause.CUT

    async def _wire_session(  # noqa: PLR0911 - one return per way a session ends
        self, end: asyncio.Future[HearingOutcome], opening_at: float
    ) -> HearingOutcome:
        wire = self._wire
        assert wire is not None  # noqa: S101 - set by the caller
        opening = wire.opening = asyncio.timeout_at(opening_at)
        try:
            async with opening:
                socket = await self._open(end)
                if socket is None:
                    return end.result()
                if isinstance(socket, NoToken):
                    return socket
                try:
                    await socket.send(self._update)
                    wire.ended = await self._converse(socket, end, opening, wire)
                    await socket.close()
                    return wire.ended
                finally:
                    # A stalled peer holds a polite close past the deadline or a
                    # cancel; aborting after a completed close is a no-op.
                    socket.transport.abort()
        except TimeoutError as timed_out:
            return _timeout_outcome(timed_out, wire.ended, opening)
        except InvalidStatus as refused:
            return _refusal_outcome(refused)
        except InvalidMessage as unparsed:
            return Cut() if isinstance(unparsed.__cause__, EOFError) else Malformed()
        except _FAILURES as failure:
            return _failure_outcome(failure)

    async def _open(
        self, end: asyncio.Future[HearingOutcome]
    ) -> ClientConnection | NoToken | None:
        opening = asyncio.create_task(self._connect())
        try:
            await asyncio.wait({opening, end}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            await _abandoned(opening)
            raise
        if opening.done():
            return opening.result()
        await _abandoned(opening)
        return None

    async def _connect(self) -> ClientConnection | NoToken:
        try:
            tokens = await self._token_source()
        except Exception as failure:  # noqa: BLE001 - re-raised unchanged by `_run`
            raise _SourceFailedError(failure) from None
        if tokens is None:
            return NoToken()
        return await _Connect(
            self._url,
            additional_headers={
                **token_headers(
                    workload_token=tokens.workload_token,
                    person_token=tokens.person_token,
                ),
                **self._traceparent_headers,
            },
            open_timeout=None,
            logger=_SOCKET_LOGGER,
        )

    async def _converse(
        self,
        socket: ClientConnection,
        end: asyncio.Future[HearingOutcome],
        opening: asyncio.Timeout,
        wire: _Wire,
    ) -> HearingOutcome:
        reader = asyncio.create_task(self._read(socket, opening, wire))
        writer = asyncio.create_task(self._write(socket, wire))
        pending = {reader, writer, end}
        try:
            while reader in pending and end in pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                if writer in done and writer.exception() is not None:
                    break
        finally:
            await _settled(reader, writer)
        for child in (reader, writer):
            if not child.cancelled() and (failure := child.exception()) is not None:
                if isinstance(failure, ConnectionClosed):
                    wire.close_code = socket.close_code
                return _failure_outcome(failure)
        return end.result() if reader.cancelled() else Closed()

    async def _read(
        self, socket: ClientConnection, opening: asyncio.Timeout, wire: _Wire
    ) -> None:
        async for message in socket:
            if not isinstance(message, str):
                raise ServerEventError
            event = parse_server_event(message)
            if wire.opened.is_set():
                wire.answered = wire.answered or not isinstance(event, SessionUpdated)
            else:
                accept_first_event(event, bound_expired=opening.expired())
                # The first event proves the session is open, before a slow caller
                # can hold the queue.
                opening.reschedule(None)
                wire.opened.set()
                await self._announce_reconnect(wire)
            await self._received.put(self._on_timeline(event, wire))

    async def _announce_reconnect(self, wire: _Wire) -> None:
        interrupted, self._interrupted = self._interrupted, None
        if interrupted is not None:
            wire.carried = interrupted
            await self._received.put(
                Reconnected(
                    interrupted.cause, interrupted.unheard_start_ms, wire.offset_ms
                )
            )

    @staticmethod
    def _on_timeline(event: ServerEvent, wire: _Wire) -> ServerEvent:
        """Put the service's positions on the hearing's audio timeline."""
        match event:
            case SpeechStarted():
                return event.model_copy(
                    update={"audio_start_ms": event.audio_start_ms + wire.offset_ms}
                )
            case SpeechStopped():
                moved = event.audio_end_ms + wire.offset_ms
                wire.stopped[event.item_id] = moved
                return event.model_copy(update={"audio_end_ms": moved})
            case TranscriptCompleted() if event.item_id in wire.stopped:
                wire.last_turn_end_ms = wire.stopped.pop(event.item_id)
            case TranscriptFailed() | ErrorEvent():
                wire.failed_sent = True
        return event

    async def _write(self, socket: ClientConnection, wire: _Wire) -> None:
        # The reader tells how the connection ended.
        with contextlib.suppress(ConnectionClosed):
            await wire.opened.wait()
            while True:
                while self._held:
                    pcm = self._held.popleft()
                    self._held_bytes -= len(pcm)
                    self._written_bytes += len(pcm)
                    await socket.send(
                        AudioAppend(
                            type="input_audio_buffer.append",
                            audio=base64.b64encode(pcm).decode(),
                        ).model_dump_json(exclude_none=True)
                    )
                self._held_changed.clear()
                await self._held_changed.wait()
