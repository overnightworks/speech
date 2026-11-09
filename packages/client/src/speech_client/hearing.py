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
    session, opening included.

A caller reads `events()` while it sends: the session waits for the caller to
take an event before it reads the next one.

Leaving the context under cancellation aborts the connection at once, without a
close handshake.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Self

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


def session_update(language: HearingLanguage) -> str:
    """The one `session.update` that sets the language of a session."""
    update = SessionUpdate(
        type="session.update",
        session=TranscriptionSession(
            type="transcription",
            audio=SessionAudio(
                input=SessionInput(
                    transcription=TranscriptionSetting(language=language)
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
        language: HearingLanguage,
        deadline_seconds: float,
        held_limit_bytes: int,
    ) -> None:
        """Take every input; nothing is opened before `__aenter__`."""
        base_origin(base_url)
        self._url = realtime_url(base_url)
        self._headers = token_headers(
            workload_token=workload_token, person_token=person_token
        )
        self._update = session_update(language)
        self._deadline_seconds = deadline_seconds
        self._held_limit_bytes = held_limit_bytes
        self._held: deque[bytes] = deque()
        self._held_bytes = 0
        self._held_changed = asyncio.Event()
        self._over_limit = False
        self._abandoned = False
        self._received: asyncio.Queue[ServerEvent | None] = asyncio.Queue(maxsize=1)
        self._runner: asyncio.Task[None] | None = None
        self.outcome: HearingOutcome | None = None

    async def __aenter__(self) -> Self:
        """Start opening the connection."""
        # Eager, so a cancel cannot end the runner before its first await.
        self._runner = asyncio.create_task(self._run(), eager_start=True)
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the connection and wait until the session is gone."""
        assert self._runner is not None  # noqa: S101 - entered
        task = asyncio.current_task()
        # A caller under cancellation has run out of time: no polite close.
        self._abandoned = task is not None and task.cancelling() > 0
        self._runner.cancel()
        try:
            await self._runner
        except asyncio.CancelledError:
            # Only the runner's own cancellation is ours to absorb.
            if task is not None and task.cancelling():
                raise

    def send(self, pcm: bytes) -> None:
        """Queue PCM16 audio; past the held limit the session ends with `OverCap`."""
        assert self._runner is not None  # noqa: S101 - entered
        if self._over_limit:
            return
        if self._held_bytes + len(pcm) > self._held_limit_bytes:
            self._over_limit = True
            self._runner.cancel()
            return
        self._held.append(pcm)
        self._held_bytes += len(pcm)
        self._held_changed.set()

    async def events(self) -> AsyncIterator[ServerEvent]:
        """The server events, in order, until the session ends."""
        while True:
            if self.outcome is not None and self._received.empty():
                return
            event = await self._received.get()
            if event is None:
                return
            yield event

    async def _run(self) -> None:
        try:
            self.outcome = await self._session()
        except asyncio.CancelledError:
            if not self._over_limit:
                self.outcome = Closed()
                raise
            self.outcome = OverCap()
        finally:
            # Wakes a caller waiting in `events()`; when an event is still queued,
            # that caller sees the outcome after taking it.
            with contextlib.suppress(asyncio.QueueFull):
                self._received.put_nowait(None)

    def _may_close_politely(self) -> bool:
        """Whether no caller has run out of time, whether before or during the exit."""
        runner = asyncio.current_task()
        # `__aexit__` cancels the runner once; a second cancel is the caller's own
        # and may merge into the first before the runner sees either.
        return not self._abandoned and (runner is None or runner.cancelling() <= 1)

    async def _session(self) -> HearingOutcome:
        try:
            async with (
                asyncio.timeout(self._deadline_seconds) as deadline,
                _Connect(
                    self._url,
                    additional_headers=self._headers,
                    open_timeout=None,
                    logger=_SOCKET_LOGGER,
                ) as socket,
            ):
                await socket.send(self._update)
                writer = asyncio.create_task(self._write(socket))
                try:
                    try:
                        await self._read(socket)
                    finally:
                        writer.cancel()
                        # `asyncio.wait` does not absorb a cancel of this task.
                        await asyncio.wait({writer})
                        if not writer.cancelled():
                            with contextlib.suppress(ConnectionClosed):
                                writer.result()
                        if not deadline.expired() and self._may_close_politely():
                            await socket.close()
                finally:
                    # A stalled peer holds a polite close past the deadline or a
                    # cancel; aborting after a completed close is a no-op.
                    socket.transport.abort()
        except InvalidStatus as refused:
            return _refusal_outcome(refused)
        except InvalidMessage as unparsed:
            return Cut() if isinstance(unparsed.__cause__, EOFError) else Malformed()
        except _FAILURES as failure:
            return next(
                outcome
                for kind, outcome in _FAILURE_OUTCOMES
                if isinstance(failure, kind)
            )
        return Closed()

    async def _read(self, socket: ClientConnection) -> None:
        async for message in socket:
            if not isinstance(message, str):
                raise ServerEventError
            await self._received.put(parse_server_event(message))

    async def _write(self, socket: ClientConnection) -> None:
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
