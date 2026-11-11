"""The live hearing session: one `WS /v1/realtime`, bounded in time and memory.

`HearingSession` is an async context manager. It starts opening the connection
on entry, takes every input from its call site, and never logs, and keeps its
socket's logger above debug; no outcome and no repr carries a token or heard text.

- `send(pcm)` queues PCM16 audio at 24 kHz. Audio queued before the connection
  is open, or faster than it is written, is held up to `held_limit_bytes`; one
  byte over ends the session with `OverCap`. There is no commit: the service
  ends turns itself (WIRE-65), and a commit with no audio is refused (WIRE-64).
- `events()` yields the typed server events until the session ends.
- `outcome` is `None` while the session runs and one `HearingOutcome` after
  `events()` ran out, a closed union a caller matches with `assert_never`:
  - `Closed`: the connection ended in order, by the service or by leaving the
    context.
  - `Refused`: the upgrade was answered with an HTTP refusal.
  - `Busy`: the refusal is `busy`, with the wait the service asks for, as for
    `speak`; a `busy` without a whole-second `Retry-After` is `Malformed`.
  - `OverCap`: the held audio would pass `held_limit_bytes`.
  - `Malformed`: a server message is not an event the wire names, or the
    upgrade was answered off the wire, a redirect included: none is followed, so
    no token travels to another origin.
  - `Cut`, `PastDeadline` and `Unreachable`, as for `speak`
    (`speech_client.outcomes`); `deadline_seconds` bounds the whole
    session, opening included. `opening_seconds` (default `OPENING_SECONDS`)
    bounds the opening: the handshake and the wait for the first event, which
    the service sends first (WIRE-68). A service that does not speak by then
    ends the session with `Unreachable`; a strictly smaller deadline wins.

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
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Final, Self

from pydantic import ValidationError
from websockets.asyncio.client import connect
from websockets.exceptions import (
    ConnectionClosed,
    InvalidHandshake,
    InvalidMessage,
    InvalidStatus,
)

from speech_client._transport import base_origin
from speech_client.outcomes import (
    Busy,
    Cut,
    Malformed,
    OverCap,
    PastDeadline,
    Unreachable,
)
from speech_client.wire import (
    REALTIME_INTENT,
    REALTIME_INTENT_PARAM,
    REALTIME_PATH,
    AudioAppend,
    HearingLanguage,
    Refusal,
    RefusalCode,
    RetryAfterError,
    ServerEventError,
    SessionAudio,
    SessionInput,
    SessionUpdate,
    TranscriptionSession,
    TranscriptionSetting,
    busy_wait,
    parse_server_event,
    token_headers,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    from websockets.asyncio.client import ClientConnection

    from speech_client.wire import ServerEvent


@dataclass(frozen=True, slots=True)
class Closed:
    """The connection ended in order."""


@dataclass(frozen=True, slots=True)
class Refused:
    """The upgrade was refused over HTTP; `code` is `None` off the contract."""

    status: int
    code: RefusalCode | None


type HearingOutcome = (
    Closed | Refused | Busy | OverCap | Malformed | PastDeadline | Cut | Unreachable
)

OPENING_SECONDS: Final = 5.0

# The socket logs request headers and frame texts at debug; this logger never
# passes them on unless an app raises it on purpose.
_SOCKET_LOGGER = logging.getLogger(f"{__name__}.socket")
_SOCKET_LOGGER.setLevel(logging.WARNING)


class _Connect(connect):
    """A connection that follows no redirect."""

    def process_redirect(self, exc: Exception) -> Exception:
        return exc


def realtime_url(base_url: str) -> str:
    """The Realtime transcription WebSocket of the service at `base_url`."""
    return (
        f"ws{base_url.removeprefix('http')}{REALTIME_PATH}"
        f"?{REALTIME_INTENT_PARAM}={REALTIME_INTENT}"
    )


def session_update(languages: tuple[HearingLanguage, ...]) -> str:
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
# `InvalidMessage` that is `Cut`; `_session` handles it first.
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


async def _abandoned(opening: asyncio.Task[ClientConnection]) -> None:
    try:
        await _settled(opening)
    finally:
        if not opening.cancelled() and opening.exception() is None:
            opening.result().transport.abort()


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


class HearingSession:
    """A live hearing session; see the module for its calls and outcomes."""

    def __init__(  # noqa: PLR0913 - one keyword per call-site input
        self,
        base_url: str,
        *,
        workload_token: str,
        person_token: str,
        languages: tuple[HearingLanguage, ...],
        deadline_seconds: float,
        held_limit_bytes: int,
        opening_seconds: float = OPENING_SECONDS,
    ) -> None:
        """Take every input; nothing is opened before `__aenter__`."""
        base_origin(base_url)
        if opening_seconds <= 0:
            message = "opening_seconds must be above 0"
            raise ValueError(message)
        self._url = realtime_url(base_url)
        self._headers = token_headers(
            workload_token=workload_token, person_token=person_token
        )
        self._update = session_update(languages)
        self._deadline_seconds = deadline_seconds
        self._opening_seconds = opening_seconds
        self._held_limit_bytes = held_limit_bytes
        self._held: deque[bytes] = deque()
        self._held_bytes = 0
        self._held_changed = asyncio.Event()
        self._end: asyncio.Future[HearingOutcome] | None = None
        self._received: asyncio.Queue[ServerEvent | None] = asyncio.Queue(maxsize=1)
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

    async def events(self) -> AsyncIterator[ServerEvent]:
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
        try:
            self.outcome = await self._session(end)
        except asyncio.CancelledError:
            self.outcome = Closed()
            raise
        finally:
            # Wakes a caller waiting in `events()`; when an event is still queued,
            # that caller sees the session ended after taking it.
            with contextlib.suppress(asyncio.QueueFull):
                self._received.put_nowait(None)

    async def _session(self, end: asyncio.Future[HearingOutcome]) -> HearingOutcome:
        ended: HearingOutcome | None = None
        opening = asyncio.timeout(self._opening_seconds)
        try:
            async with asyncio.timeout(self._deadline_seconds), opening:
                socket = await self._open(end)
                if socket is None:
                    return end.result()
                try:
                    await socket.send(self._update)
                    ended = await self._converse(socket, end, opening)
                    await socket.close()
                    return ended
                finally:
                    # A stalled peer holds a polite close past the deadline or a
                    # cancel; aborting after a completed close is a no-op.
                    socket.transport.abort()
        except TimeoutError as timed_out:
            return _timeout_outcome(timed_out, ended, opening)
        except InvalidStatus as refused:
            return _refusal_outcome(refused)
        except InvalidMessage as unparsed:
            return Cut() if isinstance(unparsed.__cause__, EOFError) else Malformed()
        except _FAILURES as failure:
            return _failure_outcome(failure)

    async def _open(
        self, end: asyncio.Future[HearingOutcome]
    ) -> ClientConnection | None:
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

    async def _connect(self) -> ClientConnection:
        return await _Connect(
            self._url,
            additional_headers=self._headers,
            open_timeout=None,
            logger=_SOCKET_LOGGER,
        )

    async def _converse(
        self,
        socket: ClientConnection,
        end: asyncio.Future[HearingOutcome],
        opening: asyncio.Timeout,
    ) -> HearingOutcome:
        reader = asyncio.create_task(self._read(socket, opening))
        writer = asyncio.create_task(self._write(socket))
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
                return _failure_outcome(failure)
        return end.result() if reader.cancelled() else Closed()

    async def _read(self, socket: ClientConnection, opening: asyncio.Timeout) -> None:
        async for message in socket:
            if not isinstance(message, str):
                raise ServerEventError
            event = parse_server_event(message)
            # The first event proves the session is open, before a slow caller can
            # hold the queue. A bound that expires in this very pass has won:
            # rescheduling then raises.
            if not opening.expired():
                opening.reschedule(None)
            await self._received.put(event)

    async def _write(self, socket: ClientConnection) -> None:
        # The reader tells how the connection ended.
        with contextlib.suppress(ConnectionClosed):
            while True:
                await self._held_changed.wait()
                self._held_changed.clear()
                while self._held:
                    pcm = self._held.popleft()
                    await socket.send(
                        AudioAppend(
                            type="input_audio_buffer.append",
                            audio=base64.b64encode(pcm).decode(),
                        ).model_dump_json(exclude_none=True)
                    )
                    self._held_bytes -= len(pcm)
