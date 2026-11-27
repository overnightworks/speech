"""The wire: token headers, hearing request, refusals, speak request, health."""

import json

import pytest
from pydantic import ValidationError

from speech_client.wire import (
    AUTHORIZATION_HEADER,
    CONTENT_TYPE_HEADER,
    MODEL,
    VOICES,
    WORKLOAD_TOKEN_HEADER,
    AudioEventType,
    GivesMarks,
    Health,
    HearingHealth,
    Language,
    RefusalCode,
    ResponseFormat,
    SentenceMarks,
    Sound,
    SpeakingHealth,
    SpeakingModel,
    SpeechRequest,
    StreamFormat,
    TranscriptionRequest,
    VoiceHealth,
    VoiceId,
    Word,
    audio_event,
    refusal,
    speech_request_body,
    speech_request_headers,
    token_headers,
)


@pytest.mark.parametrize("field", ["file", "model", "language", "response_format"])
def test_refusal_names_a_hearing_request_field(field: str) -> None:
    assert refusal(RefusalCode.INVALID_REQUEST, field).error.param == field


@pytest.mark.parametrize("language", list(Language))
def test_hearing_accepts_each_language_of_the_wire(language: Language) -> None:
    request = TranscriptionRequest.model_validate(
        {"model": MODEL, "language": language}
    )

    assert request.language is language


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"language": "fr"}, "language"),
        ({"response_format": "text"}, "response_format"),
        ({"model": "whisper-1"}, "model"),
        ({"temperature": "0"}, "temperature"),
    ],
)
def test_transcription_fields_outside_the_wire_are_refused(
    fields: dict[str, str], named: str
) -> None:
    with pytest.raises(ValidationError, match=named):
        TranscriptionRequest.model_validate({"model": MODEL} | fields)


SAMPLE_RATE = 22050
CARD_MEMORY_MB = 7


def _voice_health_body() -> dict[str, object]:
    return {
        "model": SpeakingModel.PIPER.value,
        "name": "A name",
        "languages": [Language.DE.value],
        "takes_instructions": False,
        "gives_marks": {"words": True, "sounds": False},
        "ready": True,
        "sample_rate": SAMPLE_RATE,
    }


def _health_body() -> dict[str, object]:
    return {
        "speaking": {"ready": True, "voices": {"v": _voice_health_body()}},
        "hearing": {"model": "h", "ready": False},
        "card_memory_mb": CARD_MEMORY_MB,
    }


def test_a_health_body_parses_into_the_typed_answer() -> None:
    health = Health.model_validate(_health_body())

    assert health.speaking.voices["v"].sample_rate == SAMPLE_RATE
    assert health.hearing.ready is False
    assert health.card_memory_mb == CARD_MEMORY_MB


def test_a_health_body_parses_the_name_and_the_languages_of_a_voice() -> None:
    voice = Health.model_validate(_health_body()).speaking.voices["v"]

    assert (voice.model, voice.name, voice.languages) == (
        SpeakingModel.PIPER,
        "A name",
        (Language.DE,),
    )


def test_a_health_answer_ignores_a_key_it_does_not_know() -> None:
    """WIRE-127: a health answer ignores a key it does not know."""
    body = _health_body()
    body["later"] = 1
    body["hearing"] = {"model": "h", "ready": False, "later": 1}

    assert Health.model_validate(body) == Health.model_validate(_health_body())


def _with_value(path: tuple[str, ...], value: object) -> dict[str, object]:
    body = _health_body()
    node: dict[str, object] = body
    for key in path[:-1]:
        node = node[key]  # type: ignore[assignment]
    node[path[-1]] = value
    return body


@pytest.mark.parametrize(
    "path",
    [
        ("speaking", "ready"),
        ("speaking", "voices", "v", "ready"),
        ("speaking", "voices", "v", "takes_instructions"),
        ("hearing", "ready"),
    ],
)
@pytest.mark.parametrize("value", ["true", 1, None])
def test_a_health_readiness_that_is_not_a_boolean_is_refused(
    path: tuple[str, ...], value: object
) -> None:
    with pytest.raises(ValidationError):
        Health.model_validate(_with_value(path, value))


@pytest.mark.parametrize(
    "path", [("speaking", "voices", "v", "sample_rate"), ("card_memory_mb",)]
)
@pytest.mark.parametrize("value", ["7", 7.5])
def test_a_health_integer_that_is_not_an_integer_is_refused(
    path: tuple[str, ...], value: object
) -> None:
    with pytest.raises(ValidationError):
        Health.model_validate(_with_value(path, value))


def test_a_health_body_built_through_the_models_still_parses() -> None:
    built = Health(
        speaking=SpeakingHealth(
            ready=True,
            voices={
                "v": VoiceHealth(
                    model=SpeakingModel.PIPER,
                    name="A name",
                    languages=(Language.DE,),
                    takes_instructions=False,
                    gives_marks=GivesMarks(words=True, sounds=False),
                    ready=True,
                    sample_rate=SAMPLE_RATE,
                )
            },
        ),
        hearing=HearingHealth(model="h", ready=False),
        card_memory_mb=CARD_MEMORY_MB,
    )

    assert Health.model_validate(built.model_dump(mode="json")) == built


def test_a_health_answer_is_frozen() -> None:
    health = Health.model_validate(_health_body())

    with pytest.raises(ValidationError):
        health.card_memory_mb = 8


def test_the_token_headers_are_the_two_tokens_under_the_wire_names() -> None:
    tokens = {"workload_token": "app", "person_token": "person"}

    assert token_headers(**tokens) == {
        WORKLOAD_TOKEN_HEADER: "app",
        AUTHORIZATION_HEADER: "Bearer person",
    }


def test_the_speech_request_headers_are_the_token_headers_and_the_json_type() -> None:
    tokens = {"workload_token": "app", "person_token": "person"}

    assert speech_request_headers(**tokens) == {
        **token_headers(**tokens),
        CONTENT_TYPE_HEADER: "application/json",
    }


def test_the_catalogue_covers_every_voice() -> None:
    assert set(VOICES) == set(VoiceId)


@pytest.mark.parametrize("voice", list(VoiceId))
def test_a_request_pairs_each_voice_with_its_model(voice: VoiceId) -> None:
    body = speech_request_body(
        "Hallo", voice, ResponseFormat.WAV, VOICES[voice].languages[0]
    )

    request = SpeechRequest.model_validate_json(body)

    assert (request.model, request.voice) == (VOICES[voice].model, voice)


@pytest.mark.parametrize(
    ("voice", "model"),
    [(v, m) for v in VoiceId for m in SpeakingModel if m is not VOICES[v].model],
)
def test_a_request_pairing_a_voice_with_another_model_is_refused_on_voice(
    voice: VoiceId, model: SpeakingModel
) -> None:
    with pytest.raises(ValidationError) as refused:
        SpeechRequest(
            model=model, input="Hallo", voice=voice, response_format=ResponseFormat.WAV
        )

    assert [error["loc"] for error in refused.value.errors()] == [("voice",)]


@pytest.mark.parametrize("model", ["speech", "tts-1", ""])
def test_a_request_naming_another_model_is_refused_on_model(model: str) -> None:
    body = json.dumps(
        {
            "model": model,
            "input": "Hallo",
            "voice": next(iter(VOICES)),
            "response_format": "wav",
        }
    )

    with pytest.raises(ValidationError) as refused:
        SpeechRequest.model_validate_json(body)

    assert [error["loc"] for error in refused.value.errors()] == [("model",)]


@pytest.mark.parametrize("voice", ["piper", "qwen3-tts-1.7b", "alloy", "", {"id": "x"}])
def test_a_request_naming_a_voice_outside_the_catalogue_is_refused_on_voice(
    voice: object,
) -> None:
    body = json.dumps(
        {
            "model": SpeakingModel.PIPER,
            "input": "Hallo",
            "voice": voice,
            "response_format": "wav",
        }
    )

    with pytest.raises(ValidationError) as refused:
        SpeechRequest.model_validate_json(body)

    assert [error["loc"] for error in refused.value.errors()] == [("voice",)]


_ONE_LANGUAGE = [v for v in VoiceId if len(VOICES[v].languages) == 1]
_VOICE_AND_ITS_LANGUAGE = [(v, lang) for v in VoiceId for lang in VOICES[v].languages]


@pytest.mark.parametrize(("voice", "language"), _VOICE_AND_ITS_LANGUAGE)
def test_a_request_speaks_the_language_it_names_and_the_body_carries_it(
    voice: VoiceId, language: Language
) -> None:
    body = speech_request_body("Hallo", voice, ResponseFormat.WAV, language)

    assert SpeechRequest.model_validate_json(body).spoken_language is language
    assert json.loads(body)["language"] == language


@pytest.mark.parametrize("voice", _ONE_LANGUAGE)
def test_a_voice_of_one_language_speaks_it_without_a_language_in_the_body(
    voice: VoiceId,
) -> None:
    body = speech_request_body("Hallo", voice, ResponseFormat.WAV)

    assert "language" not in json.loads(body)
    request = SpeechRequest.model_validate_json(body)
    assert request.spoken_language is VOICES[voice].languages[0]


@pytest.mark.parametrize("voice", [v for v in VoiceId if VOICES[v].takes_instructions])
def test_a_request_carries_the_instructions_a_voice_takes(voice: VoiceId) -> None:
    """WIRE-141: the body carries `instructions` and parses back to them."""
    body = speech_request_body(
        "Hallo",
        voice,
        ResponseFormat.WAV,
        VOICES[voice].languages[0],
        "Speak angrily",
    )

    assert json.loads(body)["instructions"] == "Speak angrily"
    assert SpeechRequest.model_validate_json(body).instructions == "Speak angrily"


@pytest.mark.parametrize("voice", list(VoiceId))
def test_a_request_without_instructions_has_none_and_no_key(voice: VoiceId) -> None:
    body = speech_request_body(
        "Hallo", voice, ResponseFormat.WAV, VOICES[voice].languages[0]
    )

    assert "instructions" not in json.loads(body)
    assert SpeechRequest.model_validate_json(body).instructions is None


@pytest.mark.parametrize(
    "voice", [v for v in VoiceId if not VOICES[v].takes_instructions]
)
def test_instructions_for_a_voice_that_takes_none_are_refused_on_instructions(
    voice: VoiceId,
) -> None:
    """WIRE-142: instructions for a voice that takes none are refused."""
    with pytest.raises(ValidationError) as refused:
        SpeechRequest(
            model=VOICES[voice].model,
            input="Hallo",
            voice=voice,
            response_format=ResponseFormat.WAV,
            instructions="Speak angrily",
        )

    assert [error["loc"] for error in refused.value.errors()] == [("instructions",)]


def test_a_health_body_parses_what_a_voice_gives_as_marks() -> None:
    """WIRE-122: each voice reports `gives_marks` for words and sounds."""
    voice = Health.model_validate(_health_body()).speaking.voices["v"]

    assert voice.gives_marks == GivesMarks(words=True, sounds=False)


@pytest.mark.parametrize("kind", ["words", "sounds"])
@pytest.mark.parametrize("value", ["true", 1, None])
def test_a_gives_marks_flag_that_is_not_a_boolean_is_refused(
    kind: str, value: object
) -> None:
    body = _health_body()
    body["speaking"]["voices"]["v"]["gives_marks"][kind] = value  # type: ignore[index]

    with pytest.raises(ValidationError):
        Health.model_validate(body)


def _speech_body(**more: object) -> bytes:
    voice = VoiceId.NORMAN
    return json.dumps(
        {
            "model": VOICES[voice].model,
            "input": "Hallo",
            "voice": voice,
            "response_format": "wav",
            **more,
        }
    ).encode()


def test_a_request_without_a_stream_format_streams_the_audio_unnamed() -> None:
    """WIRE-151: `audio` is the default and a request body never carries it."""
    request = SpeechRequest.model_validate_json(_speech_body())

    assert request.stream_format is StreamFormat.AUDIO
    assert "stream_format" not in json.loads(request.model_dump_json())


def test_a_request_asking_for_audio_by_name_is_the_default_request() -> None:
    named = SpeechRequest.model_validate_json(_speech_body(stream_format="audio"))

    assert named == SpeechRequest.model_validate_json(_speech_body())


def test_a_request_for_the_event_stream_keeps_it_in_its_body_and_names_its_type() -> (
    None
):
    """WIRE-151: `sse` answers as `text/event-stream`, whatever the audio's format."""
    request = SpeechRequest.model_validate_json(_speech_body(stream_format="sse"))

    assert json.loads(request.model_dump_json())["stream_format"] == "sse"
    assert request.media_type == "text/event-stream"


@pytest.mark.parametrize("response_format", list(ResponseFormat))
def test_a_request_for_the_audio_answers_with_the_media_type_of_its_format(
    response_format: ResponseFormat,
) -> None:
    request = SpeechRequest.model_validate_json(
        _speech_body(response_format=response_format)
    )

    assert request.media_type == response_format.media_type


@pytest.mark.parametrize("value", ["mp3", "", "SSE", None, 1, True])
def test_a_stream_format_outside_the_wire_is_refused_on_stream_format(
    value: object,
) -> None:
    with pytest.raises(ValidationError) as refused:
        SpeechRequest.model_validate_json(_speech_body(stream_format=value))

    assert [error["loc"] for error in refused.value.errors()] == [("stream_format",)]


def test_a_refusal_names_stream_format_as_a_request_field() -> None:
    """WIRE-94: `param` may be `stream_format`."""
    assert refusal(RefusalCode.INVALID_REQUEST, "stream_format").error.param == (
        "stream_format"
    )


def test_an_event_is_one_data_line_with_the_type_and_a_blank_line() -> None:
    """WIRE-158: an event is `data: <JSON object with type>` and a blank line."""
    frame = audio_event(AudioEventType.DELTA, audio="AAA=")

    assert frame.endswith(b"\n\n")
    assert frame.count(b"\n") == len(b"\n\n")
    assert json.loads(frame.removeprefix(b"data: ")) == {
        "type": "speech.audio.delta",
        "audio": "AAA=",
    }


@pytest.mark.parametrize(
    ("event_type", "wire_type"),
    [
        pytest.param(AudioEventType.MARKS, "speech.audio.marks", id="marks"),
        pytest.param(AudioEventType.DONE, "speech.audio.done", id="done"),
    ],
)
def test_an_event_type_goes_on_the_wire_as_its_ruled_token(
    event_type: AudioEventType, wire_type: str
) -> None:
    """WIRE-158: marks and done are the ruled `speech.audio.*` tokens."""
    frame = audio_event(event_type)

    assert json.loads(frame.removeprefix(b"data: "))["type"] == wire_type


def test_an_event_keeps_a_word_of_any_script_on_one_ascii_line() -> None:
    marks = SentenceMarks(
        words=(Word(text="Grüße\n😀", start_char=0, end_char=7, start=0.0, end=0.5),)
    )

    frame = audio_event(AudioEventType.MARKS, **marks.model_dump(mode="json"))

    assert frame.isascii()
    assert frame.count(b"\n") == len(b"\n\n")
    sent = json.loads(frame.removeprefix(b"data: "))
    assert sent["words"][0]["text"] == "Grüße\n😀"
    assert sent["sounds"] is None


@pytest.mark.parametrize(
    "exclusion",
    [{"exclude_defaults": True}, {"exclude_unset": True}, {"exclude": {"estimated"}}],
    ids=["defaults", "unset", "named"],
)
def test_a_sound_serializes_when_the_options_already_leave_the_flag_out(
    exclusion: dict[str, object],
) -> None:
    sound = Sound(ipa="a", start=0.0, end=0.05)

    assert "estimated" not in sound.model_dump(**exclusion)
