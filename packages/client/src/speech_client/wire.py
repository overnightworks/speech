"""The wire between an app and the speech service.

This module is the one owner of what travels on `POST /v1/audio/speech`,
`POST /v1/audio/transcriptions` and the Realtime transcription WebSocket: the
paths, the requests, the transcription answer, the Realtime events, and the
closed list of refusals with their statuses, messages and the body they are
answered in (OpenAI's error form). It owns the answer of `GET /health` as well.
The service and its callers import these types and copy none of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Literal
from uuid import uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

SPEECH_PATH: Final = "/v1/audio/speech"
TRANSCRIPTIONS_PATH: Final = "/v1/audio/transcriptions"
REALTIME_PATH: Final = "/v1/realtime"
HEALTH_PATH: Final = "/health"
REALTIME_INTENT_PARAM: Final = "intent"
REALTIME_INTENT: Final = "transcription"
API_VERSION: Final = 1
MODEL: Final = "speech"
RETRY_AFTER_HEADER: Final = "Retry-After"
CONTENT_TYPE_HEADER: Final = "Content-Type"
WORKLOAD_TOKEN_HEADER: Final = "Speech-Workload-Token"  # noqa: S105 - a header name and no secret
AUTHORIZATION_HEADER: Final = "Authorization"
AUTHENTICATE_HEADER: Final = "WWW-Authenticate"

RequestField = Literal[
    "file",
    "model",
    "input",
    "voice",
    "language",
    "instructions",
    "response_format",
    "intent",
]
_INPUT_WITHOUT_CONTENT: Final = "input must hold a letter or a digit"
MAX_INSTRUCTIONS_CHARACTERS: Final = 500


class ResponseFormat(StrEnum):
    """The audio formats the service speaks.

    Each owns the media type and the container signature of its answer.
    """

    WAV = "wav"
    OPUS = "opus"

    @property
    def media_type(self) -> str:
        """The `Content-Type` of a 200 in this format."""
        return _MEDIA_TYPES[self]

    def begins_like_container(self, body: bytes) -> bool:
        """Whether `body` begins with this format's container signature."""
        if self is ResponseFormat.WAV:
            return body[:4] == b"RIFF" and body[8:12] == b"WAVE"
        return body[:4] == b"OggS"


_MEDIA_TYPES: Final = {
    ResponseFormat.WAV: "audio/wav",
    ResponseFormat.OPUS: "audio/ogg",
}


class Language(StrEnum):
    """The languages the service hears and speaks."""

    DE = "de"
    EN = "en"


class SpeakingModel(StrEnum):
    """The engines the service speaks with: a request's `model`."""

    PIPER = "piper"
    QWEN = "qwen3-tts-0.6b"


class VoiceId(StrEnum):
    """The closed catalogue of voices: a request's `voice`."""

    THORSTEN = "de_DE-thorsten-medium"
    KERSTIN = "de_DE-kerstin-medium"
    NORMAN = "en_US-norman-medium"
    LJSPEECH = "en_US-ljspeech-medium"
    CORI = "en_GB-cori-medium"
    RYAN = "ryan"


@dataclass(frozen=True, slots=True)
class CatalogueVoice:
    """What the wire says of a voice: its engine, name, languages and instructions.

    A voice of one language speaks it when a request names none. A voice that
    takes instructions speaks in the tone a request's `instructions` name.
    """

    model: SpeakingModel
    name: str
    languages: tuple[Language, ...]
    takes_instructions: bool = False


VOICES: Final[Mapping[VoiceId, CatalogueVoice]] = {
    VoiceId.THORSTEN: CatalogueVoice(SpeakingModel.PIPER, "Thorsten", (Language.DE,)),
    VoiceId.KERSTIN: CatalogueVoice(SpeakingModel.PIPER, "Kerstin", (Language.DE,)),
    VoiceId.NORMAN: CatalogueVoice(SpeakingModel.PIPER, "Norman", (Language.EN,)),
    VoiceId.LJSPEECH: CatalogueVoice(SpeakingModel.PIPER, "LJ Speech", (Language.EN,)),
    VoiceId.CORI: CatalogueVoice(SpeakingModel.PIPER, "Cori", (Language.EN,)),
    VoiceId.RYAN: CatalogueVoice(
        SpeakingModel.QWEN,
        "Ryan",
        (Language.DE, Language.EN),
        takes_instructions=True,
    ),
}


def _absent(value: object) -> bool:
    return value is None


class SpeechRequest(BaseModel):
    """The body of `POST /v1/audio/speech`: OpenAI's fields and `language`."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, strict=True
    )

    model: SpeakingModel
    input: str
    voice: VoiceId
    response_format: ResponseFormat
    language: Language | None = Field(None, exclude_if=_absent, validate_default=True)
    instructions: str | None = Field(None, exclude_if=_absent)

    @field_validator("input")
    @classmethod
    def _input_has_content(cls, text: str) -> str:
        if not any(character.isalnum() for character in text):
            raise ValueError(_INPUT_WITHOUT_CONTENT)
        return text

    @field_validator("voice")
    @classmethod
    def _voice_belongs_to_the_model(
        cls, voice: VoiceId, info: ValidationInfo
    ) -> VoiceId:
        model = info.data.get("model")
        if model is not None and VOICES[voice].model is not model:
            message = "voice must be a voice of model"
            raise ValueError(message)
        return voice

    @field_validator("language")
    @classmethod
    def _language_belongs_to_the_voice(
        cls, language: Language | None, info: ValidationInfo
    ) -> Language | None:
        voice = info.data.get("voice")
        if voice is None:
            return language
        languages = VOICES[voice].languages
        if language is None:
            if len(languages) > 1:
                message = "language must be named for a voice of several languages"
                raise ValueError(message)
        elif language not in languages:
            message = "language must be a language of voice"
            raise ValueError(message)
        return language

    @field_validator("instructions")
    @classmethod
    def _instructions_are_a_tone_the_voice_takes(
        cls, instructions: str | None, info: ValidationInfo
    ) -> str | None:
        if instructions is None:
            return instructions
        if len(instructions) > MAX_INSTRUCTIONS_CHARACTERS:
            message = (
                f"instructions must be at most {MAX_INSTRUCTIONS_CHARACTERS} characters"
            )
            raise ValueError(message)
        if not any(character.isalpha() for character in instructions):
            message = "instructions must hold a letter"
            raise ValueError(message)
        voice = info.data.get("voice")
        if voice is not None and not VOICES[voice].takes_instructions:
            message = "instructions must be named for a voice that takes them"
            raise ValueError(message)
        return instructions

    @property
    def spoken_language(self) -> Language:
        """The language named, else the one language of the voice."""
        return self.language or VOICES[self.voice].languages[0]


class TranscriptionRequest(BaseModel):
    """The form fields of `POST /v1/audio/transcriptions` beside its `file`."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    model: Literal["speech"]
    language: Language = Language.DE
    response_format: Literal["json"] = "json"


class Transcription(BaseModel):
    """The answer of `POST /v1/audio/transcriptions`."""

    model_config = ConfigDict(frozen=True)

    text: str


class VoiceHealth(BaseModel):
    """One offered voice: its model, name, languages, whether it can speak, its rate."""

    model_config = ConfigDict(frozen=True)

    model: SpeakingModel
    name: str
    languages: tuple[Language, ...]
    takes_instructions: StrictBool
    ready: StrictBool
    sample_rate: StrictInt


class SpeakingHealth(BaseModel):
    """The speaking service's own readiness and the voices it offers by id."""

    model_config = ConfigDict(frozen=True)

    ready: StrictBool
    voices: dict[str, VoiceHealth]


class HearingHealth(BaseModel):
    """The hearing model and whether it can hear."""

    model_config = ConfigDict(frozen=True)

    model: str
    ready: StrictBool


class Health(BaseModel):
    """The answer of `GET /health`."""

    model_config = ConfigDict(frozen=True)

    speaking: SpeakingHealth
    hearing: HearingHealth
    card_memory_mb: StrictInt


def speech_request_body(
    text: str,
    voice: VoiceId,
    response_format: ResponseFormat,
    language: Language | None = None,
    instructions: str | None = None,
) -> bytes:
    """The JSON a caller posts to speak `text` with `voice` in `response_format`.

    `language` is named for a voice of several languages and may be named for any.
    `instructions` name the tone, for a voice that takes them.
    """
    request = SpeechRequest(
        model=VOICES[voice].model,
        input=text,
        voice=voice,
        response_format=response_format,
        language=language,
        instructions=instructions,
    )
    return request.model_dump_json().encode()


def token_headers(*, workload_token: str, person_token: str) -> dict[str, str]:
    """The app's token and the person's, for a request whose transport sets the type."""
    return {
        WORKLOAD_TOKEN_HEADER: workload_token,
        AUTHORIZATION_HEADER: f"Bearer {person_token}",
    }


def speech_request_headers(*, workload_token: str, person_token: str) -> dict[str, str]:
    """The headers a caller posts with the body: the content type and both tokens."""
    return {
        CONTENT_TYPE_HEADER: "application/json",
        **token_headers(workload_token=workload_token, person_token=person_token),
    }


class RefusalCode(StrEnum):
    """Why the service refused a request."""

    UNSUPPORTED_VERSION = "unsupported_version"
    NOT_FOUND = "not_found"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    INVALID_REQUEST = "invalid_request"
    TOO_LARGE = "too_large"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"


REFUSAL_STATUS: Final[dict[RefusalCode, HTTPStatus]] = {
    RefusalCode.UNSUPPORTED_VERSION: HTTPStatus.BAD_REQUEST,
    RefusalCode.NOT_FOUND: HTTPStatus.NOT_FOUND,
    RefusalCode.UNAUTHORIZED: HTTPStatus.UNAUTHORIZED,
    RefusalCode.FORBIDDEN: HTTPStatus.FORBIDDEN,
    RefusalCode.INVALID_REQUEST: HTTPStatus.UNPROCESSABLE_ENTITY,
    RefusalCode.TOO_LARGE: HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    RefusalCode.BUSY: HTTPStatus.TOO_MANY_REQUESTS,
    RefusalCode.UNAVAILABLE: HTTPStatus.SERVICE_UNAVAILABLE,
}

_REFUSAL_MESSAGE: Final[dict[RefusalCode, str]] = {
    RefusalCode.UNSUPPORTED_VERSION: "This API version is not supported.",
    RefusalCode.NOT_FOUND: "There is nothing at this path.",
    RefusalCode.UNAUTHORIZED: (
        "A workload token and the person's token are required, and both must be valid."
    ),
    RefusalCode.FORBIDDEN: "The tokens do not permit this request.",
    RefusalCode.INVALID_REQUEST: "The request is not valid.",
    RefusalCode.TOO_LARGE: "The request body is too large.",
    RefusalCode.BUSY: "The service is busy; try again.",
    RefusalCode.UNAVAILABLE: "The service cannot serve this request right now.",
}


class RefusalDetail(BaseModel):
    """The `error` object of a refusal."""

    model_config = ConfigDict(frozen=True)

    message: str
    type: Literal["invalid_request_error", "server_error"]
    param: RequestField | None
    code: RefusalCode


class Refusal(BaseModel):
    """The JSON body of every refusal under version 1."""

    model_config = ConfigDict(frozen=True)

    error: RefusalDetail

    @property
    def status(self) -> HTTPStatus:
        """The HTTP status this refusal is answered with."""
        return REFUSAL_STATUS[self.error.code]


def refusal(code: RefusalCode, param: RequestField | None = None) -> Refusal:
    """The refusal for `code`, naming the request field at fault, if any."""
    status = REFUSAL_STATUS[code]
    kind = (
        "server_error"
        if status >= HTTPStatus.INTERNAL_SERVER_ERROR
        else "invalid_request_error"
    )
    return Refusal(
        error=RefusalDetail(
            message=_REFUSAL_MESSAGE[code], type=kind, param=param, code=code
        )
    )


class RetryAfterError(ValueError):
    """A `busy` answer whose `Retry-After` is not a whole number of seconds."""


@dataclass(frozen=True)
class BusyWait:
    """How long the service asks the app to wait after a `busy` refusal."""

    retry_after_seconds: int


def busy_wait(status: int, headers: Mapping[str, str]) -> BusyWait | None:
    """The wait a `busy` answer asks for, or `None` for any other status.

    Raises `RetryAfterError` when a `busy` answer lacks a `Retry-After` of whole
    seconds, at least 1: the service never sends a date.
    """
    if status != REFUSAL_STATUS[RefusalCode.BUSY]:
        return None
    wanted = RETRY_AFTER_HEADER.lower()
    value = next((v for name, v in headers.items() if name.lower() == wanted), None)
    if value is None or not (value.isascii() and value.isdigit()) or int(value) < 1:
        message = (
            f"{RETRY_AFTER_HEADER} must be whole seconds, at least 1, got {value!r}"
        )
        raise RetryAfterError(message)
    return BusyWait(retry_after_seconds=int(value))


# The Realtime transcription session, as OpenAI types it, narrowed to what this
# service applies. A field it does not apply is refused, never ignored.


class _SessionPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PcmFormat(_SessionPart):
    """PCM16 little-endian mono at 24 kHz, the one input format.

    A sample is `sample_bytes` bytes and a frame of samples has `channels`
    channels. Both are class constants, so the session's shape names only
    `type` and `rate`.
    """

    sample_bytes: ClassVar[int] = 2
    channels: ClassVar[int] = 1

    type: Literal["audio/pcm"]
    rate: Literal[24000]


def _distinct(codes: tuple[Language, ...]) -> tuple[Language, ...]:
    if len(set(codes)) != len(codes):
        message = "languages must not repeat a code"
        raise ValueError(message)
    return codes


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
    language: Language | None = Field(None, exclude_if=_absent)
    languages: AllowedLanguages | None = Field(None, exclude_if=_absent)

    @field_validator("language", "languages")
    @classmethod
    def _is_not_null(cls, value: object) -> object:
        if value is None:
            message = "an explicit null is not a language"
            raise ValueError(message)
        return value

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


class SessionInput(_SessionPart):
    """The input side of a session."""

    format: PcmFormat = PcmFormat(type="audio/pcm", rate=24000)
    transcription: TranscriptionSetting = TranscriptionSetting()
    turn_detection: ServerVad = ServerVad(type="server_vad")


class SessionAudio(_SessionPart):
    """The audio side of a session."""

    input: SessionInput = SessionInput()


class TranscriptionSession(_SessionPart):
    """A transcription session; an object absent from a request takes its default.

    An object a request supplies names its `type` and, for the format, its `rate`.
    """

    type: Literal["transcription"]
    audio: SessionAudio = SessionAudio()


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
