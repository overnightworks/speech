"""A live session over a fake socket: what it sent, and audio to feed it."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from typing import TYPE_CHECKING, get_args

from speech_client.realtime_wire import (
    AudioAppend,
    AudioCommit,
    ErrorEvent,
    ServerEvent,
    SessionUpdate,
    SustainFrom,
    TranscriptFailed,
    parse_server_event,
)
from speech_service.config import DEFAULT_LIVE_SILENCE_SECONDS
from speech_service.live_session import LiveSession
from speech_service.pcm import HEARING_SAMPLE_RATE
from speech_service.voice_activity import WINDOW_SAMPLES

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel

    from speech_service.hearing import Hearing

SPEECH_LEVEL = 5000
INPUT_RATE = 24_000
WAIT_SECONDS = 5.0
ASK_FOR_SOUNDS = {"type": "audioset"}
NOT_ASKED = object()
_INPUT_SAMPLES_PER_WINDOW = WINDOW_SAMPLES * INPUT_RATE // HEARING_SAMPLE_RATE


def read_event(text: str) -> ServerEvent:
    """The server event in `text`; one the client cannot read in full fails loud."""
    event = parse_server_event(text)
    assert json.loads(event.model_dump_json()) == json.loads(text), (
        "the client reads every field the service sent"
    )
    return event


class FakeSocket:
    """The events a session sent, as the client reads them, and whether it closed."""

    def __init__(self) -> None:
        """Start empty and open."""
        self.events: list[ServerEvent] = []
        self.closed = False
        self.close_code: int | None = None
        self._changed = asyncio.Condition()

    async def send(self, text: str) -> None:
        """Record one event; one the client cannot read fails the session."""
        event = read_event(text)
        async with self._changed:
            self.events.append(event)
            self._changed.notify_all()

    async def close(self, code: int) -> None:
        """Record the close and its code."""
        async with self._changed:
            self.closed = True
            self.close_code = code
            self._changed.notify_all()

    async def until(self, ready: Callable[[], bool]) -> None:
        """Wait, without polling, until `ready()` holds."""
        async with asyncio.timeout(WAIT_SECONDS):
            async with self._changed:
                await self._changed.wait_for(ready)

    async def until_closed(self) -> None:
        """Wait until the session closed the socket."""
        await self.until(lambda: self.closed)

    def types(self) -> list[type[ServerEvent]]:
        """The class of every event sent, in order."""
        return [type(event) for event in self.events]

    def of_type[Event: ServerEvent](self, kind: type[Event]) -> list[Event]:
        """The events of one class, in order."""
        return [event for event in self.events if isinstance(event, kind)]


class StallingSocket(FakeSocket):
    """A socket whose send of one event of one class stalls until released."""

    def __init__(self, kind: type[ServerEvent], occurrence: int = 1) -> None:
        """Stall the `occurrence`th event of class `kind`."""
        super().__init__()
        self._kind = kind
        self._skipped = occurrence - 1
        self.stalled = asyncio.Event()
        self.proceed = asyncio.Event()

    async def send(self, text: str) -> None:
        """Record the event, after the stall if it is the stalled kind."""
        if not self.stalled.is_set() and isinstance(
            parse_server_event(text), self._kind
        ):
            if self._skipped:
                self._skipped -= 1
            else:
                self.stalled.set()
                await self.proceed.wait()
        await super().send(text)


def _type_of(event: type[BaseModel]) -> str:
    """The wire name a client event's `type` field is fixed to."""
    (name,) = get_args(event.model_fields["type"].annotation)
    return name


def samples(count: int, level: int = SPEECH_LEVEL) -> str:
    """The base64 PCM16 at 24 kHz of `count` samples of one level."""
    return base64.b64encode(struct.pack("<h", level) * count).decode()


def audio(seconds: float, level: int = SPEECH_LEVEL) -> str:
    """The base64 PCM16 at 24 kHz of a constant level `seconds` long."""
    return samples(int(seconds * INPUT_RATE), level)


def windows(count: int, level: int = SPEECH_LEVEL) -> str:
    """The base64 PCM16 at 24 kHz of `count` whole 32 ms windows of one level."""
    return samples(count * _INPUT_SAMPLES_PER_WINDOW, level)


def append(data: str, event_id: str | None = None) -> str:
    """An `input_audio_buffer.append` message."""
    event: dict[str, object] = {"type": _type_of(AudioAppend), "audio": data}
    if event_id:
        event["event_id"] = event_id
    return json.dumps(event)


def commit(event_id: str | None = None) -> str:
    """An `input_audio_buffer.commit` message."""
    event: dict[str, object] = {"type": _type_of(AudioCommit)}
    if event_id:
        event["event_id"] = event_id
    return json.dumps(event)


def sustain_from(**fields: object) -> str:
    """An `input_audio_buffer.sustain_from` message with exactly `fields`."""
    return json.dumps({"type": _type_of(SustainFrom), **fields})


def update(
    language: str | None = "de",
    model: str | None = None,
    languages: list[str] | None = None,
    sound_detection: dict[str, str] | None = None,
    sustained_speech: object = NOT_ASKED,
) -> str:
    """A `session.update` message for `language` or `languages`, `model` if given.

    It asks for sounds with `sound_detection`, and for a sustained speech with
    `sustained_speech`, whatever that is, when they are given.
    """
    transcription: dict[str, object] = {}
    if language is not None:
        transcription["language"] = language
    if languages is not None:
        transcription["languages"] = languages
    if model is not None:
        transcription["model"] = model
    audio_input: dict[str, object] = {"transcription": transcription}
    if sound_detection is not None:
        audio_input["sound_detection"] = sound_detection
    if sustained_speech is not NOT_ASKED:
        audio_input["sustained_speech"] = sustained_speech
    session = {"type": "transcription", "audio": {"input": audio_input}}
    return json.dumps({"type": _type_of(SessionUpdate), "session": session})


async def started(
    hearing: Hearing,
    *,
    final_wait_seconds: float = 30.0,
    socket: FakeSocket | None = None,
) -> tuple[LiveSession, FakeSocket]:
    """A session over `hearing` that has sent `session.created`."""
    socket = socket or FakeSocket()
    session = LiveSession(
        hearing,
        final_wait_seconds=final_wait_seconds,
        silence_seconds=DEFAULT_LIVE_SILENCE_SECONDS,
        send=socket.send,
        close=socket.close,
    )
    await session.start()
    return session, socket


async def commit_a_turn(session: LiveSession, seconds: float = 0.3) -> None:
    """Feed `seconds` of speech, then commit it."""
    await session.on_message(append(audio(seconds)))
    await session.on_message(commit())


async def one_question(session: LiveSession) -> None:
    """Half a second of speech, then a silence that ends the turn, and a tick."""
    await session.on_message(append(audio(0.5)))
    await session.on_message(append(audio(0.8, 0)))
    await session.on_tick()


def end_is(socket: FakeSocket, code: str, failed: int) -> bool:
    """Whether the session failed `failed` items, sent `error` `code` and closed."""
    types = socket.types()
    error = socket.of_type(ErrorEvent)[0].error
    return (
        types[-1] is ErrorEvent
        and types.count(TranscriptFailed) == failed
        and types[-1 - failed : -1] == [TranscriptFailed] * failed
        and error.code == code
        and socket.closed
    )
