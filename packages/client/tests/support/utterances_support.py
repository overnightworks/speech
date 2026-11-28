"""The scripted browser, recorder and services the utterance tests share."""

from __future__ import annotations

import asyncio
import contextlib
import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from packages.client.tests.support.hearing_support import (
    CREATED_EVENT,
    LOOPBACK,
    fixed_tokens,
    silent_peer,
)
from speech_client.realtime_wire import (
    DurationUsage,
    SoundDetected,
    SoundName,
    SpeechStarted,
    SpeechStopped,
    SpeechSustained,
    TranscriptCompleted,
    TranscriptDelta,
    TranscriptLanguage,
)
from speech_client.relay import HOP_FRAME_BYTES
from speech_client.utterances import (
    HeardSound,
    HearingCallbacks,
    Partial,
    SpeechStart,
    SpeechStop,
    SustainedFrom,
    SustainedVoice,
    TurnEnd,
    Utterance,
    hear_utterances,
)
from speech_client.wire import Language

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
    from http import HTTPStatus

    from websockets.asyncio.server import ServerConnection
    from websockets.http11 import Request

    from speech_client.hearing import Reconnected, TokenSource
    from speech_client.moments import SustainMoments
    from speech_client.realtime_wire import ServerEvent
    from speech_client.utterances import UtteranceOutcome

type Peer = Callable[[], contextlib.AbstractAsyncContextManager[int]]

WAIT_SECONDS = 10
UNASKED_SESSION_UPDATE_DE = (
    '{"type":"session.update","session":{"type":"transcription","audio":{"input":'
    '{"format":{"type":"audio/pcm","rate":24000},"transcription":{"model":"speech",'
    '"languages":["de"]},"turn_detection":{"type":"server_vad"}}}}}'
)


def ended_already() -> asyncio.Event:
    ended = asyncio.Event()
    ended.set()
    return ended


def binary_message(size: int) -> dict[str, object]:
    return {"type": "websocket.receive", "bytes": b"\0" * size, "text": None}


def text_message() -> dict[str, object]:
    return {"type": "websocket.receive", "bytes": None, "text": "hello"}


def delta(item_id: str, text: str) -> TranscriptDelta:
    return TranscriptDelta(item_id=item_id, delta=text)


def started(item_id: str, at_ms: int) -> SpeechStarted:
    return SpeechStarted(item_id=item_id, audio_start_ms=at_ms)


def stopped(item_id: str, at_ms: int) -> SpeechStopped:
    return SpeechStopped(item_id=item_id, audio_end_ms=at_ms)


def sustained(item_id: str, at_ms: int) -> SpeechSustained:
    return SpeechSustained(item_id=item_id, audio_end_ms=at_ms)


def sound_detected(name: SoundName, start_ms: int, end_ms: int) -> SoundDetected:
    return SoundDetected(sound=name, audio_start_ms=start_ms, audio_end_ms=end_ms)


def completed(
    item_id: str,
    text: str,
    language: Language = Language.DE,
    traceparent: str | None = None,
) -> TranscriptCompleted:
    return TranscriptCompleted(
        item_id=item_id,
        transcript=text,
        usage=DurationUsage(seconds=1),
        languages=(TranscriptLanguage(code=language),),
        traceparent=traceparent,
    )


def utterance_of(  # noqa: PLR0913 - one keyword per field of the record
    text: str,
    language: Language = Language.DE,
    *,
    start: int | None = None,
    end: int | None = None,
    item_id: str = "a",
    traceparent: str | None = None,
) -> Utterance:
    return Utterance(
        text,
        language,
        audio_start_ms=start,
        audio_end_ms=end,
        item_id=item_id,
        traceparent=traceparent,
    )


class Browser:
    """A browser socket that plays its messages, then waits for its peer.

    It offers only what the relay calls: nothing can be written to it but a close.
    """

    def __init__(self, messages: tuple[Mapping[str, object], ...] = ()) -> None:
        """Hold the messages it plays."""
        self._messages = list(messages)
        self.closed_with: list[int] = []

    async def receive(self) -> Mapping[str, object]:
        """The next message, else wait forever."""
        if self._messages:
            return self._messages.pop(0)
        await asyncio.Event().wait()
        raise AssertionError

    async def close(self, code: int) -> None:
        """Note the close code."""
        self.closed_with.append(code)


class NamingBrowser(Browser):
    """A browser that names a moment once its messages are played."""

    def __init__(
        self, moments: SustainMoments, messages: tuple[Mapping[str, object], ...]
    ) -> None:
        """Hold the messages it plays, and the handle that names the moment."""
        super().__init__(messages)
        self._moments = moments
        self.position: int | None = None

    async def receive(self) -> Mapping[str, object]:
        """The next message; after the last, the moment is named at the audio so far."""
        if not self._messages:
            self.position = self._moments.now()
        return await super().receive()


@dataclass
class Recorder:
    """The app's side: what its callbacks got, in order, and how they behave."""

    ended: asyncio.Event = field(default_factory=asyncio.Event)
    ends_after_first: bool = False
    raises: Exception | None = None
    got: list[
        Utterance
        | Partial
        | HeardSound
        | SpeechStart
        | SpeechStop
        | SustainedVoice
        | SustainedFrom
    ] = field(default_factory=list)
    turn_ends: list[TurnEnd] = field(default_factory=list)
    reconnected: list[Reconnected] = field(default_factory=list)

    async def on_reconnected(self, reconnected: Reconnected) -> None:
        """Note a reconnection."""
        self.reconnected.append(reconnected)

    async def on_turn_ended(self, turn_end: TurnEnd) -> None:
        """Note a turn's end."""
        self.turn_ends.append(turn_end)

    async def on_utterance(self, utterance: Utterance) -> None:
        """Take a finished utterance."""
        await self._take(utterance)

    async def on_partial(self, partial: Partial) -> None:
        """Take the text so far."""
        await self._take(partial)

    async def on_sound(self, sound: HeardSound) -> None:
        """Take a heard sound."""
        await self._take(sound)

    async def on_speech_started(self, start: SpeechStart) -> None:
        """Take a speech start."""
        await self._take(start)

    async def on_speech_stopped(self, stop: SpeechStop) -> None:
        """Take a speech stop."""
        await self._take(stop)

    async def on_speech_sustained(self, voice: SustainedVoice) -> None:
        """Take a sustained voice."""
        await self._take(voice)

    async def on_speech_sustained_from(self, voice: SustainedFrom) -> None:
        """Take a voice sustained from the newest moment."""
        await self._take(voice)

    async def _take(
        self,
        heard: Utterance
        | Partial
        | HeardSound
        | SpeechStart
        | SpeechStop
        | SustainedVoice
        | SustainedFrom,
    ) -> None:
        self.got.append(heard)
        if self.ends_after_first:
            self.ended.set()
        if self.raises is not None:
            raise self.raises


@dataclass(frozen=True)
class Bounds:
    """The bounds a scenario runs under."""

    deadline_seconds: float = 5
    held_limit_bytes: int = 2 * HOP_FRAME_BYTES
    opening_seconds: float = 5


def handler_saying(
    *events: ServerEvent, stays_open: bool = False
) -> Callable[[ServerConnection], Awaitable[None]]:
    async def handler(connection: ServerConnection) -> None:
        await connection.recv()
        await connection.send(CREATED_EVENT)
        for event in events:
            await connection.send(event.model_dump_json())
        if stays_open:
            await connection.wait_closed()

    return handler


async def speaks_nonsense(connection: ServerConnection) -> None:
    await connection.send("not json")
    await connection.wait_closed()


async def drops_the_connection(connection: ServerConnection) -> None:
    connection.transport.abort()


@contextlib.asynccontextmanager
async def service_saying(
    handler: Callable[[ServerConnection], Awaitable[None]],
) -> AsyncIterator[int]:
    async with serve(handler, LOOPBACK, 0) as server:
        yield server.sockets[0].getsockname()[1]


@contextlib.asynccontextmanager
async def service_refusing(
    status: HTTPStatus, *headers: tuple[str, str]
) -> AsyncIterator[int]:
    def refuse(_: ServerConnection, __: Request) -> Response:
        return Response(status.value, status.phrase, Headers(headers), b"")

    async with serve(handler_saying(), LOOPBACK, 0, process_request=refuse) as server:
        yield server.sockets[0].getsockname()[1]


@contextlib.asynccontextmanager
async def service_not_answering() -> AsyncIterator[int]:
    async with await asyncio.start_server(silent_peer, LOOPBACK, 0) as server:
        yield server.sockets[0].getsockname()[1]


async def hear(  # noqa: PLR0913 - one keyword per thing a scenario varies
    peer: Peer,
    browser: Browser | None = None,
    recorder: Recorder | None = None,
    *,
    partials: bool = True,
    sounds: bool = False,
    speech_marks: bool = False,
    sustained: bool = False,
    sustained_speech_ms: int | None = None,
    moments: SustainMoments | None = None,
    reconnects: bool = True,
    languages: tuple[Language, ...] = (Language.DE,),
    bounds: Bounds = Bounds(),
    token_source: TokenSource = fixed_tokens,
    traceparent: str | None = None,
    on_text: Callable[[str], None] | None = None,
) -> UtteranceOutcome:
    recorder = recorder or Recorder()
    async with peer() as port:
        return await asyncio.wait_for(
            hear_utterances(
                browser or Browser(),
                f"http://{LOOPBACK}:{port}",
                token_source=token_source,
                traceparent=traceparent,
                sustained_speech_ms=sustained_speech_ms,
                moments=moments,
                languages=languages,
                deadline_seconds=bounds.deadline_seconds,
                held_limit_bytes=bounds.held_limit_bytes,
                opening_seconds=bounds.opening_seconds,
                ended=recorder.ended,
                callbacks=HearingCallbacks(
                    on_utterance=recorder.on_utterance,
                    on_partial=recorder.on_partial if partials else None,
                    on_turn_ended=recorder.on_turn_ended,
                    on_speech_started=recorder.on_speech_started
                    if speech_marks
                    else None,
                    on_speech_stopped=recorder.on_speech_stopped
                    if speech_marks
                    else None,
                    on_sound=recorder.on_sound if sounds else None,
                    on_speech_sustained=recorder.on_speech_sustained
                    if sustained
                    else None,
                    on_speech_sustained_from=recorder.on_speech_sustained_from
                    if moments is not None
                    else None,
                    on_reconnected=recorder.on_reconnected if reconnects else None,
                    on_text=on_text,
                ),
            ),
            WAIT_SECONDS,
        )


def hearing(*events: ServerEvent, stays_open: bool = False) -> Peer:
    return functools.partial(
        service_saying, handler_saying(*events, stays_open=stays_open)
    )
