"""A hearing service that plays one script per connection, and what each saw."""

from __future__ import annotations

import base64
import contextlib
import json
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING

from websockets.asyncio.server import serve
from websockets.frames import CloseCode

from packages.client.tests.support.hearing_support import CREATED_EVENT, LOOPBACK
from speech_client.realtime_wire import (
    DurationUsage,
    SessionUpdated,
    SpeechStarted,
    SpeechStopped,
    SpeechSustainedFrom,
    TranscriptCompleted,
    TranscriptFailed,
    TranscriptionSession,
    TranscriptLanguage,
    error_event,
    item_failure,
)
from speech_client.wire import (
    DRAIN_CLOSE_CODE,
    REFUSAL_STATUS,
    RETRY_AFTER_HEADER,
    Language,
    RefusalCode,
    refusal,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from websockets.asyncio.server import ServerConnection
    from websockets.datastructures import Headers
    from websockets.http11 import Request, Response

CLOSE_ENDS_IN_ORDER = CloseCode.NORMAL_CLOSURE


@dataclass
class Visit:
    """What one connection of the service received."""

    headers: Headers
    update: str | None = None
    audio: list[bytes] = field(default_factory=list)
    frames: list[str] = field(default_factory=list)
    moments: list[tuple[int, int]] = field(default_factory=list)
    """Each `sustain_from`: how many chunks came before it, and its `from_ms`."""


@dataclass(frozen=True)
class Refusal:
    """An upgrade answered with an HTTP refusal instead of a session."""

    code: RefusalCode | None
    retry_after: str | None = None

    @property
    def status(self) -> int:
        """The HTTP status the refusal answers with."""
        return REFUSAL_STATUS[self.code] if self.code else HTTPStatus.BAD_GATEWAY


type Script = Refusal | Callable[[ServerConnection, Visit], Awaitable[None]]


@contextlib.asynccontextmanager
async def scripted(*scripts: Script) -> AsyncIterator[tuple[int, list[Visit]]]:
    """Serve connection n with script n (the last script serves the rest).

    Yield the port and the visits, one per attempt so far, refused ones included.
    """
    visits: list[Visit] = []
    visit_of: dict[int, Visit] = {}

    def script_of(visit: Visit) -> Script:
        return scripts[min(visits.index(visit) + 1, len(scripts)) - 1]

    def meets(connection: ServerConnection, request: Request) -> Response | None:
        visit = Visit(request.headers)
        visits.append(visit)
        visit_of[id(request)] = visit
        script = script_of(visit)
        if not isinstance(script, Refusal):
            return None
        body = refusal(script.code).model_dump_json().encode() if script.code else b""
        answer = connection.respond(HTTPStatus(script.status), body.decode())
        if script.retry_after is not None:
            answer.headers[RETRY_AFTER_HEADER] = script.retry_after
        return answer

    async def play(connection: ServerConnection) -> None:
        visit = visit_of[id(connection.request)]
        script = script_of(visit)
        assert not isinstance(script, Refusal)
        await script(connection, visit)

    async with serve(
        play, LOOPBACK, 0, close_timeout=0.2, process_request=meets
    ) as server:
        yield server.sockets[0].getsockname()[1], visits


async def opens(connection: ServerConnection, visit: Visit) -> None:
    """Take the `session.update`, then say `session.created`."""
    visit.update = str(await connection.recv())
    await connection.send(CREATED_EVENT)


async def takes_audio(
    connection: ServerConnection, visit: Visit, chunks: int, moments: int = 0
) -> None:
    """Take audio and moments until `chunks` chunks and `moments` moments arrived."""
    while len(visit.audio) < chunks or len(visit.moments) < moments:
        frame = str(await connection.recv())
        visit.frames.append(frame)
        message = json.loads(frame)
        if "audio" in message:
            visit.audio.append(base64.b64decode(message["audio"]))
        else:
            visit.moments.append((len(visit.audio), message["from_ms"]))


def says(*events: str, then_close: int | None = None) -> Script:
    """Open, say `events`, then close with `then_close` or wait for the client."""

    async def play(connection: ServerConnection, visit: Visit) -> None:
        await opens(connection, visit)
        for event in events:
            await connection.send(event)
        if then_close is None:
            await connection.wait_closed()
        else:
            await connection.close(then_close)

    return play


async def drains(connection: ServerConnection, visit: Visit) -> None:
    await opens(connection, visit)
    await connection.close(DRAIN_CLOSE_CODE)


async def ends_in_order(connection: ServerConnection, visit: Visit) -> None:
    await opens(connection, visit)
    await connection.close(CLOSE_ENDS_IN_ORDER)


async def drops_after_opening(connection: ServerConnection, visit: Visit) -> None:
    await opens(connection, visit)
    connection.transport.abort()


async def drains_before_opening(connection: ServerConnection, visit: Visit) -> None:
    visit.update = str(await connection.recv())
    await connection.close(DRAIN_CLOSE_CODE)


def updated() -> str:
    return SessionUpdated(
        session=TranscriptionSession(type="transcription")
    ).model_dump_json()


def speech_started(item_id: str, audio_start_ms: int) -> str:
    return SpeechStarted(
        item_id=item_id, audio_start_ms=audio_start_ms
    ).model_dump_json()


answers_then_drains = says(speech_started("i", 0), then_close=DRAIN_CLOSE_CODE)


def speech_stopped(item_id: str, audio_end_ms: int) -> str:
    return SpeechStopped(item_id=item_id, audio_end_ms=audio_end_ms).model_dump_json()


def sustained_from(item_id: str, from_ms: int, audio_end_ms: int) -> str:
    return SpeechSustainedFrom(
        item_id=item_id, from_ms=from_ms, audio_end_ms=audio_end_ms
    ).model_dump_json()


def completed(item_id: str, transcript: str = "words") -> str:
    return TranscriptCompleted(
        item_id=item_id,
        transcript=transcript,
        usage=DurationUsage(seconds=1),
        languages=(TranscriptLanguage(code=Language.DE),),
    ).model_dump_json()


def failed(item_id: str) -> str:
    return TranscriptFailed(
        item_id=item_id, error=item_failure(RefusalCode.UNAVAILABLE)
    ).model_dump_json()


def errored() -> str:
    return error_event(RefusalCode.UNAVAILABLE).model_dump_json()
