"""The wire between an app and the speech service.

This module is the one owner of what travels on `POST /v1/audio/speech` and
`POST /v1/audio/transcriptions`, and of the paths, `GET /health` answer and the
closed list of refusals with their statuses, messages and OpenAI-form body; what
travels on the Realtime transcription WebSocket is in `speech_client.realtime_wire`.
The service and its callers import these types and copy none of them.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    StrictBool,
    StrictInt,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_serializer,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


SPEECH_PATH: Final = "/v1/audio/speech"
TRANSCRIPTIONS_PATH: Final = "/v1/audio/transcriptions"
REALTIME_PATH: Final = "/v1/realtime"
HEALTH_PATH: Final = "/health"
REALTIME_INTENT_PARAM: Final = "intent"
REALTIME_INTENT: Final = "transcription"
DRAIN_CLOSE_CODE: Final = 1012  # the stopping service's close, with no `error` sent
API_VERSION: Final = 1
MODEL: Final = "speech"
RETRY_AFTER_HEADER: Final = "Retry-After"
CONTENT_TYPE_HEADER: Final = "Content-Type"
WORKLOAD_TOKEN_HEADER: Final = "Speech-Workload-Token"  # noqa: S105 - a header name and no secret
AUTHORIZATION_HEADER: Final = "Authorization"
AUTHENTICATE_HEADER: Final = "WWW-Authenticate"
TRACEPARENT_HEADER: Final = "traceparent"

RequestField = Literal[
    "file",
    "model",
    "input",
    "voice",
    "language",
    "instructions",
    "response_format",
    "stream_format",
    "intent",
]
_INPUT_WITHOUT_CONTENT: Final = "input must hold a letter or a digit"
MAX_INSTRUCTIONS_CHARACTERS: Final = 500


class ResponseFormat(StrEnum):
    """The audio formats the service speaks, each with its media type and signature."""

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


class StreamFormat(StrEnum):
    """How a speak answer is framed: the audio bytes, or server-sent events."""

    AUDIO = "audio"
    SSE = "sse"


_EVENT_STREAM_MEDIA_TYPE: Final = "text/event-stream"
_DATA_PREFIX: Final = "data: "
EVENT_END: Final = "\n\n"


class AudioEventType(StrEnum):
    """The `type` of each event of a `stream_format` `sse` answer."""

    DELTA = "speech.audio.delta"
    MARKS = "speech.audio.marks"
    DONE = "speech.audio.done"


class Language(StrEnum):
    """The languages the service hears and speaks."""

    DE = "de"
    EN = "en"


class SpeakingModel(StrEnum):
    """The engines the service speaks with: a request's `model`."""

    PIPER = "piper"
    QWEN = "qwen3-tts-1.7b"


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
        SpeakingModel.QWEN, "Ryan", (Language.DE, Language.EN), takes_instructions=True
    ),
}


def absent(value: object) -> bool:
    """Whether a field holds no value, so it is left out of the body."""
    return value is None


def _is_audio_stream(stream_format: StreamFormat) -> bool:
    return stream_format is StreamFormat.AUDIO


def answer_media_type(
    response_format: ResponseFormat, stream_format: StreamFormat
) -> str:
    """The `Content-Type` of a 200: the audio's, or the event stream's."""
    if stream_format is StreamFormat.SSE:
        return _EVENT_STREAM_MEDIA_TYPE
    return response_format.media_type


class SpeechRequest(BaseModel):
    """The body of `POST /v1/audio/speech`: OpenAI's fields and `language`."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, strict=True
    )

    model: SpeakingModel
    input: str
    voice: VoiceId
    response_format: ResponseFormat
    stream_format: StreamFormat = Field(StreamFormat.AUDIO, exclude_if=_is_audio_stream)
    language: Language | None = Field(None, exclude_if=absent, validate_default=True)
    instructions: str | None = Field(None, exclude_if=absent)

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
        if len(instructions) > MAX_INSTRUCTIONS_CHARACTERS or not any(
            character.isalpha() for character in instructions
        ):
            message = "instructions must be a short text with a letter"
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

    @property
    def media_type(self) -> str:
        """The `Content-Type` of a 200: the audio's, or the event stream's."""
        return answer_media_type(self.response_format, self.stream_format)


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


class Word(BaseModel):
    """A word of the input and when it is spoken.

    `start_char` and `end_char` count Unicode code points of the input as received,
    `end_char` exclusive, so `text` is `input[start_char:end_char]`. `start` and
    `end` are seconds from the answer's first sample.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    start_char: StrictInt
    end_char: StrictInt
    start: float
    end: float


class Sound(BaseModel):
    """A sound of a sentence as IPA and when it is spoken.

    `start` and `end` are seconds from the answer's first sample; `estimated`
    says they are an estimate, not what the voice reported. A voice that reports
    its own times leaves the field off the wire.
    """

    model_config = ConfigDict(frozen=True)

    ipa: str
    start: float
    end: float
    estimated: StrictBool = False

    @model_serializer(mode="wrap")
    def _without_the_flag_when_the_times_are_the_voices_own(
        self, serialize: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        fields: dict[str, object] = serialize(self)
        if not self.estimated:
            fields.pop("estimated", None)
        return fields


class SentenceMarks(BaseModel):
    """The marks of one voice sentence; `None` where they are not proven.

    A kind whose `GivesMarks` flag is false is always `None`; otherwise `None`
    means the sentence's marks could not be proven.
    """

    model_config = ConfigDict(frozen=True)

    words: tuple[Word, ...] | None = None
    sounds: tuple[Sound, ...] | None = None


def audio_event(event_type: AudioEventType, **fields: object) -> bytes:
    """One server-sent event of a `sse` answer: a `data:` line, then a blank line."""
    body = json.dumps({"type": event_type.value, **fields})
    return f"{_DATA_PREFIX}{body}{EVENT_END}".encode("ascii")


def _decoded_audio(value: object) -> object:
    return base64.b64decode(value, validate=True) if isinstance(value, str) else value


class _AudioDelta(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[AudioEventType.DELTA]
    audio: Annotated[bytes, BeforeValidator(_decoded_audio)]


class _AudioMarks(SentenceMarks):
    type: Literal[AudioEventType.MARKS]


class _AudioDone(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[AudioEventType.DONE]


_AUDIO_EVENT: Final = TypeAdapter(
    Annotated[_AudioDelta | _AudioMarks | _AudioDone, Field(discriminator="type")]
)


def read_audio_event(frame: bytes) -> bytes | SentenceMarks | None:
    """The content of one event frame: audio bytes, marks, or `None` for `done`.

    `frame` is the event without its blank line. A frame that is no event of
    the wire raises `ValueError` (`pydantic.ValidationError` is one).
    """
    if not frame.startswith(_DATA_PREFIX.encode()):
        message = "an event begins with `data: `"
        raise ValueError(message)
    match _AUDIO_EVENT.validate_json(frame.removeprefix(_DATA_PREFIX.encode())):
        case _AudioDelta(audio=audio):
            return audio
        case _AudioMarks(words=words, sounds=sounds):
            return SentenceMarks(words=words, sounds=sounds)
        case _:
            return None


class GivesMarks(BaseModel):
    """Which marks a voice gives with a `sse` answer."""

    model_config = ConfigDict(frozen=True)

    words: StrictBool
    sounds: StrictBool


class VoiceHealth(BaseModel):
    """One offered voice: its model, name, languages, marks, whether it speaks.

    A service from before `gives_marks` sends none: it reads as no marks.
    """

    model_config = ConfigDict(frozen=True)

    model: SpeakingModel
    name: str
    languages: tuple[Language, ...]
    takes_instructions: StrictBool
    gives_marks: GivesMarks = GivesMarks(words=False, sounds=False)
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


def speech_request_body(  # noqa: PLR0913 - one parameter for each field of the body
    text: str,
    voice: VoiceId,
    response_format: ResponseFormat,
    language: Language | None = None,
    instructions: str | None = None,
    *,
    stream_format: StreamFormat = StreamFormat.AUDIO,
) -> bytes:
    """The JSON a caller posts to speak `text` with `voice` in `response_format`.

    `language` is named for a voice of several languages and may be named for any.
    `instructions` name the tone, for a voice that takes them. `stream_format`
    asks for the audio itself or for the event stream.
    """
    request = SpeechRequest(
        model=VOICES[voice].model,
        input=text,
        voice=voice,
        response_format=response_format,
        stream_format=stream_format,
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
