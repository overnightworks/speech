"""The Realtime transcription wire: the session and the events of its WebSocket.

`speech_client.wire` owns the paths, the refusals and the speak and health bodies;
this module owns the session and the events that travel on the Realtime
transcription WebSocket. The service and its callers import these types and copy
none of them.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, ClassVar, Final, Literal
from uuid import uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from speech_client.wire import MODEL, Language, RefusalCode, absent, refusal

# The Realtime transcription session, as OpenAI types it, narrowed to what this
# service applies. A field it does not apply is refused, never ignored.


class _SessionPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PcmFormat(_SessionPart):
    """PCM16 little-endian mono at 24 kHz, the one input format.

    A sample is `sample_bytes` bytes, a frame has `channels` channels and a
    signed sample peaks at `full_scale`, all class constants, so the session's
    shape names only `type` and `rate`.
    """

    sample_bytes: ClassVar[int] = 2
    channels: ClassVar[int] = 1
    full_scale: ClassVar[float] = 32768.0

    type: Literal["audio/pcm"]
    rate: Literal[24000]


def _distinct(codes: tuple[Language, ...]) -> tuple[Language, ...]:
    if len(set(codes)) != len(codes):
        message = "languages must not repeat a code"
        raise ValueError(message)
    return codes


def _refuse_null(value: object, noun: str) -> object:
    """A field dumped with `exclude_if=absent` refuses an explicit null."""
    if value is None:
        message = f"an explicit null is not {noun}"
        raise ValueError(message)
    return value


AllowedLanguages = Annotated[
    tuple[Language, ...], Field(min_length=1), AfterValidator(_distinct)
]


class TranscriptionSetting(_SessionPart):
    """The model and the languages a session hears.

    A request names `language`, or `languages` (an ordered set whose first element
    is the main language), or neither. Only the field a request named is dumped;
    neither means `language` is German.
    """

    model: Literal["speech"] = MODEL
    language: Language | None = Field(None, exclude_if=absent)
    languages: AllowedLanguages | None = Field(None, exclude_if=absent)

    @field_validator("language", "languages")
    @classmethod
    def _is_not_null(cls, value: object) -> object:
        return _refuse_null(value, "a language")

    @field_validator("languages")
    @classmethod
    def _is_not_beside_language(cls, value: object, info: ValidationInfo) -> object:
        if info.data.get("language") is not None:
            message = "language and languages are not sent together"
            raise ValueError(message)
        return value

    @model_validator(mode="after")
    def _names_german_when_neither_is_named(self) -> TranscriptionSetting:
        if self.language is None and self.languages is None:
            # The model is frozen, and a validator may not return another instance.
            object.__setattr__(self, "language", Language.DE)
        return self

    @property
    def allowed(self) -> tuple[Language, ...]:
        """The languages the session hears; the first is the main one."""
        if self.languages is not None:
            return self.languages
        return (self.language,)


class ServerVad(_SessionPart):
    """Server turn detection; C2's timing is fixed by the service."""

    type: Literal["server_vad"]


class SoundDetection(_SessionPart):
    """Sound detection by the AudioSet tagger, asked for by a session that wants it."""

    type: Literal["audioset"]


class SessionInput(_SessionPart):
    """The input side of a session.

    `sound_detection` is dumped only when it was asked for.
    """

    format: PcmFormat = PcmFormat(type="audio/pcm", rate=24000)
    transcription: TranscriptionSetting = TranscriptionSetting()
    turn_detection: ServerVad = ServerVad(type="server_vad")
    sound_detection: SoundDetection | None = Field(None, exclude_if=absent)

    @field_validator("sound_detection")
    @classmethod
    def _is_not_null(cls, value: object) -> object:
        return _refuse_null(value, "a sound detection")


class SessionAudio(_SessionPart):
    """The audio side of a session."""

    input: SessionInput = SessionInput()


class TranscriptionSession(_SessionPart):
    """A transcription session; an object absent from a request takes its default.

    An object a request supplies names its `type` and, for the format, its `rate`.
    """

    type: Literal["transcription"]
    audio: SessionAudio = SessionAudio()

    def without_sound_detection(self) -> TranscriptionSession:
        """This session as it is applied when its sound detection is not."""
        audio_input = self.audio.input.model_copy(update={"sound_detection": None})
        audio = self.audio.model_copy(update={"input": audio_input})
        return self.model_copy(update={"audio": audio})


class _ClientEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    event_id: str | None = None


class SessionUpdate(_ClientEvent):
    """`session.update`: the one setup, before the first append."""

    type: Literal["session.update"]
    session: TranscriptionSession


class AudioAppend(_ClientEvent):
    """`input_audio_buffer.append`: base64 PCM16 at 24 kHz."""

    type: Literal["input_audio_buffer.append"]
    audio: str


class AudioCommit(_ClientEvent):
    """`input_audio_buffer.commit`: end the open turn now."""

    type: Literal["input_audio_buffer.commit"]


ClientEvent = SessionUpdate | AudioAppend | AudioCommit
_CLIENT_EVENT: Final = TypeAdapter(Annotated[ClientEvent, Field(discriminator="type")])


class ClientEventError(ValueError):
    """A client message that is not one of the events this service accepts."""

    def __init__(self, param: str | None, event_id: str | None) -> None:
        """Name the field at fault and the client event's own id; never its content."""
        super().__init__("the client event is not valid")
        self.param = param
        self.event_id = event_id


def parse_client_event(text: str) -> ClientEvent:
    """The client event in `text`; raises `ClientEventError` for anything else."""
    try:
        return _CLIENT_EVENT.validate_json(text)
    except ValidationError as error:
        first = error.errors(
            include_url=False, include_context=False, include_input=False
        )[0]
        # A field error leads with the event's type, which the param leaves out.
        if first["type"] == "union_tag_invalid":
            param = "type"
        else:
            param = ".".join(map(str, first["loc"][1:])) or None
        raise ClientEventError(param, _event_id(text)) from None


def _event_id(text: str) -> str | None:
    try:
        event = json.loads(text)
    except ValueError:
        return None
    event_id = event.get("event_id") if isinstance(event, dict) else None
    return event_id if isinstance(event_id, str) else None


def new_event_id() -> str:
    """A fresh id for a server event."""
    return f"event_{uuid4().hex}"


def new_item_id() -> str:
    """A fresh id for a conversation item."""
    return f"item_{uuid4().hex}"


class _ServerEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=new_event_id)


class SessionCreated(_ServerEvent):
    """`session.created`: the effective default session, first on every session."""

    type: Literal["session.created"] = "session.created"
    session: TranscriptionSession


class SessionUpdated(_ServerEvent):
    """`session.updated`: the session the service applied."""

    type: Literal["session.updated"] = "session.updated"
    session: TranscriptionSession


class SpeechStarted(_ServerEvent):
    """`input_audio_buffer.speech_started`: C2 first heard speech in a turn."""

    type: Literal["input_audio_buffer.speech_started"] = (
        "input_audio_buffer.speech_started"
    )
    item_id: str
    audio_start_ms: int


class SpeechStopped(_ServerEvent):
    """`input_audio_buffer.speech_stopped`: C2's silence or cap ended the turn."""

    type: Literal["input_audio_buffer.speech_stopped"] = (
        "input_audio_buffer.speech_stopped"
    )
    item_id: str
    audio_end_ms: int


class AudioCommitted(_ServerEvent):
    """`input_audio_buffer.committed`: the turn is closed and will be heard."""

    type: Literal["input_audio_buffer.committed"] = "input_audio_buffer.committed"
    item_id: str
    previous_item_id: str | None


class SoundName(StrEnum):
    """The sounds a session hears, by their AudioSet display names."""

    LAUGHTER = "Laughter"
    COUGH = "Cough"
    CLAPPING = "Clapping"
    SCREAMING = "Screaming"


class SoundDetected(_ServerEvent):
    """`input_audio_buffer.sound_detected`: a sound began in a session that asked."""

    type: Literal["input_audio_buffer.sound_detected"] = (
        "input_audio_buffer.sound_detected"
    )
    sound: SoundName
    audio_start_ms: int
    audio_end_ms: int


class InputAudioPart(BaseModel):
    """The audio of a turn; the audio itself is never echoed."""

    model_config = ConfigDict(frozen=True)

    type: Literal["input_audio"] = "input_audio"
    transcript: str | None = None


class UserMessageItem(BaseModel):
    """The conversation item of one turn."""

    model_config = ConfigDict(frozen=True)

    id: str
    object: Literal["realtime.item"] = "realtime.item"
    type: Literal["message"] = "message"
    role: Literal["user"] = "user"
    status: Literal["completed"] = "completed"
    content: tuple[InputAudioPart, ...] = (InputAudioPart(),)


class ItemCreated(_ServerEvent):
    """`conversation.item.created`: the turn's user message item."""

    type: Literal["conversation.item.created"] = "conversation.item.created"
    previous_item_id: str | None
    item: UserMessageItem


class TranscriptDelta(_ServerEvent):
    """A provisional piece of an item's transcript."""

    type: Literal["conversation.item.input_audio_transcription.delta"] = (
        "conversation.item.input_audio_transcription.delta"
    )
    item_id: str
    content_index: Literal[0] = 0
    delta: str


class DurationUsage(BaseModel):
    """The seconds of audio an item's transcript was made from."""

    model_config = ConfigDict(frozen=True)

    type: Literal["duration"] = "duration"
    seconds: float


class TranscriptLanguage(BaseModel):
    """The language an item was decoded in."""

    model_config = ConfigDict(frozen=True)

    code: Language


class TranscriptCompleted(_ServerEvent):
    """An item's authoritative transcript, possibly empty; its one success terminal."""

    type: Literal["conversation.item.input_audio_transcription.completed"] = (
        "conversation.item.input_audio_transcription.completed"
    )
    item_id: str
    content_index: Literal[0] = 0
    transcript: str
    usage: DurationUsage
    languages: tuple[TranscriptLanguage]
    traceparent: str | None = None

    @field_validator("transcript")
    @classmethod
    def _is_trimmed(cls, value: str) -> str:
        if value != value.strip():
            message = "a transcript carries no leading or trailing whitespace"
            raise ValueError(message)
        return value


class ItemFailure(BaseModel):
    """The closed-list error of a failed item."""

    model_config = ConfigDict(frozen=True)

    message: str
    type: Literal["invalid_request_error", "server_error"]
    code: RefusalCode
    param: str | None = None


class TranscriptFailed(_ServerEvent):
    """An item that will not complete; sent only as a session ends."""

    type: Literal["conversation.item.input_audio_transcription.failed"] = (
        "conversation.item.input_audio_transcription.failed"
    )
    item_id: str
    content_index: Literal[0] = 0
    error: ItemFailure


class RealtimeError(ItemFailure):
    """The `error` object of an `error` event."""

    event_id: str | None = None


class ErrorEvent(_ServerEvent):
    """`error`: the session ends after it."""

    type: Literal["error"] = "error"
    error: RealtimeError


ServerEvent = (
    SessionCreated
    | SessionUpdated
    | SpeechStarted
    | TranscriptDelta
    | SpeechStopped
    | AudioCommitted
    | ItemCreated
    | SoundDetected
    | TranscriptCompleted
    | TranscriptFailed
    | ErrorEvent
)
_SERVER_EVENT: Final = TypeAdapter(Annotated[ServerEvent, Field(discriminator="type")])


class ServerEventError(ValueError):
    """A server message that is not one of the events the wire names."""

    def __init__(self) -> None:
        """Carry no field and no content: a transcript is user content."""
        super().__init__("the server event is not valid")


def parse_server_event(text: str) -> ServerEvent:
    """The server event in `text`; raises `ServerEventError` for anything else."""
    try:
        message = json.loads(text)
        # The models would invent an id that the sender never gave.
        parsed = (
            _SERVER_EVENT.validate_json(text)
            if isinstance(message, dict) and "event_id" in message
            else None
        )
    except ValueError, RecursionError:
        parsed = None
    if parsed is None:
        raise ServerEventError
    return parsed


def item_failure(code: RefusalCode, param: str | None = None) -> ItemFailure:
    """The failure of a closed-list `code`."""
    detail = refusal(code).error
    return ItemFailure(message=detail.message, type=detail.type, code=code, param=param)


def error_event(
    code: RefusalCode, param: str | None = None, event_id: str | None = None
) -> ErrorEvent:
    """The `error` event for `code`, naming the field and the client event at fault."""
    failure = item_failure(code, param)
    return ErrorEvent(error=RealtimeError(**failure.model_dump(), event_id=event_id))
