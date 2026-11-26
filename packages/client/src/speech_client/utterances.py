"""Finished utterances from a browser's live audio, for an app's server.

`hear_utterances` builds the `HearingSession` on a token source, relays the
browser's audio to it with `speech_client.relay.relay` and hands the app what
the service heard as typed records, through the `HearingCallbacks` it passes.
The app keeps the route, its admission, the token choice, its rule and when a
hearing ends: it sets `ended` to end it, and the relay's hard end holds, so
after `ended` is set no callback is called again.

- `on_utterance` gets an `Utterance` for each finished item with a non-empty
  transcript: its text, its decoded language and the positions, in milliseconds
  of the audio, at which the item's speech started and stopped (`None` when the
  service did not send that event), its `item_id` and the service's
  `traceparent` of the turn (`None` from a service that sends none): an app runs
  its handling of the turn under `logline.unit_trace(utterance.traceparent)`.
  An empty transcript reaches nobody.
- `on_speech_started`, if given, gets a `SpeechStart` for each `speech_started`
  event, the moment a person starts speaking, in event order with the other
  callbacks: before that item's partials and its utterance. Without it nothing
  of the event reaches the app.
- `on_speech_stopped`, if given, gets a `SpeechStop` for each `speech_stopped`
  event, the moment the service ends a person's turn, `audio_end_ms` into the
  audio, in event order with the other callbacks: before that item's utterance
  and `on_turn_ended`. Without it nothing of the event reaches the app.
- `on_turn_ended`, if given, gets a `TurnEnd` for each finished item, empty or
  not, after that item's `on_utterance`: `heard_words` is false when the
  transcript was empty, a cough for one. Without it nothing of an empty turn
  reaches the app.
- `on_partial`, if given, gets a `Partial` for each delta: the item's text so far.
  Without it nothing of a delta reaches the app.
- `on_reconnected`, if given, gets the `Reconnected` of a session that was
  reopened (`speech_client.hearing`): the old session's unfinished items are
  forgotten, as the new session's items start afresh. Without it nothing of a
  reconnect reaches the app.
- `on_sound`, if given, gets a `HeardSound` for each sound the service hears:
  its name from the wire's closed list and the positions, in milliseconds of the
  audio, that bound where it began, in event order with the other callbacks.
  Passing it is what asks the service for sounds; without it the hearing asks for
  none and sends the bytes it always sent, and a `sound_detected` that arrives
  anyway raises `ServerEventError`.
- `on_text`, if given, gets each text frame the browser's page sends, unread and
  in order with its audio frames, bounded like every frame (`relay`): the app
  validates it. It is a plain function, not an `async def`. Without it a text
  frame ends the hop with `HopClose.TEXT_FRAME`.
- A failed item or an `error` event ends the hearing: the browser is closed
  with `HopClose.SERVICE_FAILED` and the outcome is `ServiceFailed`, which names
  the event's code and nothing else of it.

`hear_utterances` returns one `UtteranceOutcome`, a closed union a caller matches
with `assert_never`: `ServiceFailed`, or the relay's `RelayOutcome` unchanged,
and the relay has then closed the browser with its code, or not at all when the
browser left. A failure of an app callback is raised unchanged. The browser gets
no token and no event, only its close code. Nothing here logs.

`traceparent` (default: the trace running at the call) parents the service's own
lines of the hearing, as for `HearingSession`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from speech_client.hearing import (
    OPENING_SECONDS,
    HearingSession,
    Reconnected,
)
from speech_client.realtime_wire import (
    ErrorEvent,
    ServerEventError,
    SoundDetected,
    SoundName,
    SpeechStarted,
    SpeechStopped,
    TranscriptCompleted,
    TranscriptDelta,
    TranscriptFailed,
)
from speech_client.relay import HOP_FRAME_BYTES, HopClose, RelayOutcome, relay

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Callable

    from speech_client.hearing import TokenSource
    from speech_client.realtime_wire import ServerEvent
    from speech_client.relay import BrowserSocket
    from speech_client.wire import Language, RefusalCode


@dataclass(frozen=True, slots=True)
class Utterance:
    """One finished item of speech the service heard."""

    text: str
    language: Language
    audio_start_ms: int | None
    audio_end_ms: int | None
    item_id: str
    traceparent: str | None = None


@dataclass(frozen=True, slots=True)
class SpeechStart:
    """A person started speaking, `audio_start_ms` into the audio."""

    audio_start_ms: int


@dataclass(frozen=True, slots=True)
class SpeechStop:
    """The service ended a person's turn, `audio_end_ms` into the audio."""

    audio_end_ms: int


@dataclass(frozen=True, slots=True)
class HeardSound:
    """A sound the service heard, between `audio_start_ms` and `audio_end_ms`."""

    sound: SoundName
    audio_start_ms: int
    audio_end_ms: int


@dataclass(frozen=True, slots=True)
class TurnEnd:
    """A person's turn ended; `heard_words` is false when it carried no words."""

    heard_words: bool


@dataclass(frozen=True, slots=True)
class Partial:
    """The text of an item heard so far."""

    text: str


@dataclass(frozen=True, slots=True)
class ServiceFailed:
    """A failed item or an `error` event ended the hearing; `code` is its code."""

    code: RefusalCode


type UtteranceOutcome = RelayOutcome | ServiceFailed


class _ServiceFailedError(Exception):
    """Carries a service failure out of the event handler to `hear_utterances`."""

    def __init__(self, code: RefusalCode) -> None:
        super().__init__()
        self.code = code


@dataclass(slots=True)
class _Items:
    """What is known of the items that have not completed."""

    text: dict[str, str] = field(default_factory=dict)
    started: dict[str, int] = field(default_factory=dict)
    stopped: dict[str, int] = field(default_factory=dict)

    def finish(self, item_id: str) -> tuple[int | None, int | None]:
        self.text.pop(item_id, None)
        return self.started.pop(item_id, None), self.stopped.pop(item_id, None)


@dataclass(frozen=True, slots=True)
class HearingCallbacks:
    """What an app takes of a hearing: `on_utterance` and the optional rest."""

    on_utterance: Callable[[Utterance], Awaitable[None]]
    on_partial: Callable[[Partial], Awaitable[None]] | None = None
    on_speech_started: Callable[[SpeechStart], Awaitable[None]] | None = None
    on_speech_stopped: Callable[[SpeechStop], Awaitable[None]] | None = None
    on_turn_ended: Callable[[TurnEnd], Awaitable[None]] | None = None
    on_sound: Callable[[HeardSound], Awaitable[None]] | None = None
    on_reconnected: Callable[[Reconnected], Awaitable[None]] | None = None
    on_text: Callable[[str], None] | None = None


@dataclass(slots=True)
class _Mapping:
    """Turns the service's events into what the app's callbacks take."""

    callbacks: HearingCallbacks
    ended: asyncio.Event
    items: _Items = field(default_factory=_Items)

    async def on_event(self, event: ServerEvent | Reconnected) -> None:
        match event:
            case Reconnected():
                self.items = _Items()
                if self.callbacks.on_reconnected is not None:
                    await self.callbacks.on_reconnected(event)
            case SpeechStarted() | SpeechStopped():
                await self._on_speech(event)
            case SoundDetected():
                await self._on_sound(event)
            case TranscriptDelta() if self.callbacks.on_partial is not None:
                text = self.items.text.get(event.item_id, "") + event.delta
                self.items.text[event.item_id] = text
                await self.callbacks.on_partial(Partial(text))
            case TranscriptCompleted():
                await self._completed(event)
            case TranscriptFailed() | ErrorEvent():
                raise _ServiceFailedError(event.error.code)
            case _:
                return

    async def _on_speech(self, event: SpeechStarted | SpeechStopped) -> None:
        match event:
            case SpeechStarted():
                self.items.started[event.item_id] = event.audio_start_ms
                if self.callbacks.on_speech_started is not None:
                    await self.callbacks.on_speech_started(
                        SpeechStart(event.audio_start_ms)
                    )
            case _:
                self.items.stopped[event.item_id] = event.audio_end_ms
                if self.callbacks.on_speech_stopped is not None:
                    await self.callbacks.on_speech_stopped(
                        SpeechStop(event.audio_end_ms)
                    )

    async def _on_sound(self, event: SoundDetected) -> None:
        if self.callbacks.on_sound is None:
            raise ServerEventError
        await self.callbacks.on_sound(
            HeardSound(event.sound, event.audio_start_ms, event.audio_end_ms)
        )

    async def _completed(self, event: TranscriptCompleted) -> None:
        start, end = self.items.finish(event.item_id)
        if event.transcript:
            await self.callbacks.on_utterance(
                Utterance(
                    event.transcript,
                    event.languages[0].code,
                    audio_start_ms=start,
                    audio_end_ms=end,
                    item_id=event.item_id,
                    traceparent=event.traceparent,
                )
            )
        if self.callbacks.on_turn_ended is not None and not self.ended.is_set():
            await self.callbacks.on_turn_ended(
                TurnEnd(heard_words=bool(event.transcript))
            )


async def hear_utterances(  # noqa: PLR0913 - one keyword per call-site input
    browser: BrowserSocket,
    origin: str,
    *,
    token_source: TokenSource,
    languages: tuple[Language, ...],
    deadline_seconds: float,
    held_limit_bytes: int,
    ended: asyncio.Event,
    callbacks: HearingCallbacks,
    opening_seconds: float = OPENING_SECONDS,
    traceparent: str | None = None,
) -> UtteranceOutcome:
    """Hear `browser`'s audio at the service at `origin`; see the module."""
    session = HearingSession(
        origin,
        token_source=token_source,
        languages=languages,
        deadline_seconds=deadline_seconds,
        held_limit_bytes=held_limit_bytes,
        opening_seconds=opening_seconds,
        traceparent=traceparent,
        sound_detection=callbacks.on_sound is not None,
    )
    try:
        return await relay(
            browser,
            session,
            app_ended=ended,
            on_event=_Mapping(callbacks, ended).on_event,
            on_text=callbacks.on_text,
            frame_limit_bytes=HOP_FRAME_BYTES,
        )
    except _ServiceFailedError as failure:
        await browser.close(HopClose.SERVICE_FAILED)
        return ServiceFailed(failure.code)
