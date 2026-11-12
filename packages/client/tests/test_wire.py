"""The wire: a `busy` wait, token headers, hearing request, refusals, events, health."""

import json
import traceback
from http import HTTPStatus

import pytest
from pydantic import ValidationError

from speech_client.wire import (
    AUTHORIZATION_HEADER,
    CONTENT_TYPE_HEADER,
    MODEL,
    REFUSAL_STATUS,
    RETRY_AFTER_HEADER,
    VOICES,
    WORKLOAD_TOKEN_HEADER,
    AudioAppend,
    AudioCommit,
    AudioCommitted,
    BusyWait,
    ClientEventError,
    DurationUsage,
    Health,
    HearingHealth,
    ItemCreated,
    Language,
    RefusalCode,
    ResponseFormat,
    RetryAfterError,
    ServerEvent,
    ServerEventError,
    SessionCreated,
    SessionUpdate,
    SessionUpdated,
    SpeakingHealth,
    SpeakingModel,
    SpeechRequest,
    SpeechStarted,
    SpeechStopped,
    TranscriptCompleted,
    TranscriptDelta,
    TranscriptFailed,
    TranscriptionRequest,
    TranscriptionSession,
    TranscriptLanguage,
    VoiceHealth,
    VoiceId,
    busy_wait,
    error_event,
    item_failure,
    parse_client_event,
    parse_server_event,
    refusal,
    speech_request_body,
    speech_request_headers,
    token_headers,
)

BUSY = REFUSAL_STATUS[RefusalCode.BUSY]


def test_busy_answer_gives_the_wait_in_seconds() -> None:
    """WIRE-104: a busy answer gives the wait in whole seconds."""
    assert busy_wait(BUSY, {RETRY_AFTER_HEADER: "5"}) == BusyWait(retry_after_seconds=5)


def test_header_name_is_matched_case_insensitively() -> None:
    assert busy_wait(int(BUSY), {RETRY_AFTER_HEADER.lower(): "1"}) == BusyWait(1)


@pytest.mark.parametrize(
    "status",
    [s for s in (*REFUSAL_STATUS.values(), HTTPStatus.OK) if s != BUSY],
)
def test_other_statuses_give_no_wait(status: HTTPStatus) -> None:
    """WIRE-104: any status but busy gives no wait."""
    assert busy_wait(status, {RETRY_AFTER_HEADER: "5"}) is None


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {RETRY_AFTER_HEADER: "-1"},
        {RETRY_AFTER_HEADER: "1.5"},
        {RETRY_AFTER_HEADER: "Wed, 21 Oct 2026 07:28:00 GMT"},
        {RETRY_AFTER_HEADER: ""},
        {RETRY_AFTER_HEADER: "0"},
    ],
    ids=["missing", "negative", "fraction", "http-date", "empty", "zero"],
)
def test_busy_answer_without_whole_seconds_fails_loud(headers: dict[str, str]) -> None:
    """WIRE-104: a busy answer without whole seconds fails loud."""
    with pytest.raises(RetryAfterError, match=RETRY_AFTER_HEADER):
        busy_wait(BUSY, headers)


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


def _session_update(session: object) -> str:
    return json.dumps({"type": "session.update", "event_id": "e1", "session": session})


def test_a_session_update_without_parts_takes_the_default_session() -> None:
    event = parse_client_event(_session_update({"type": "transcription"}))

    assert event == SessionUpdate(
        type="session.update",
        event_id="e1",
        session=TranscriptionSession(type="transcription"),
    )


def test_a_session_update_names_the_language_it_applies() -> None:
    update = {"audio": {"input": {"transcription": {"language": "en"}}}}

    event = parse_client_event(_session_update({"type": "transcription", **update}))

    assert isinstance(event, SessionUpdate)
    assert event.session.audio.input.transcription.language is Language.EN


@pytest.mark.parametrize(
    ("audio_input", "param"),
    [
        ({"format": {"type": "audio/pcm", "rate": 16000}}, "audio.input.format.rate"),
        ({"format": {"type": "audio/pcmu"}}, "audio.input.format.type"),
        ({"transcription": {"model": "whisper-1"}}, "audio.input.transcription.model"),
        (
            {"transcription": {"language": "fr"}},
            "audio.input.transcription.language",
        ),
        ({"transcription": {"prompt": "x"}}, "audio.input.transcription.prompt"),
        ({"turn_detection": None}, "audio.input.turn_detection"),
        (
            {"turn_detection": {"type": "semantic_vad"}},
            "audio.input.turn_detection.type",
        ),
        ({"noise_reduction": None}, "audio.input.noise_reduction"),
        ({"format": {"type": "audio/pcm"}}, "audio.input.format.rate"),
        ({"format": {"rate": 24000}}, "audio.input.format.type"),
        ({"turn_detection": {}}, "audio.input.turn_detection.type"),
    ],
    ids=[
        "other-rate",
        "other-format",
        "other-model",
        "other-language",
        "prompt",
        "null-turn-detection",
        "semantic-vad",
        "noise-reduction",
        "format-without-rate",
        "format-without-type",
        "empty-turn-detection",
    ],
)
def test_a_session_update_with_a_value_the_service_does_not_apply_is_refused(
    audio_input: dict[str, object], param: str
) -> None:
    text = _session_update({"type": "transcription", "audio": {"input": audio_input}})

    with pytest.raises(ClientEventError) as refused:
        parse_client_event(text)

    assert refused.value.param == f"session.{param}"
    assert refused.value.event_id == "e1"


def _transcription(setting: dict[str, object]) -> str:
    return _session_update(
        {"type": "transcription", "audio": {"input": {"transcription": setting}}}
    )


_PATH = "session.audio.input.transcription"


def test_a_session_update_names_the_ordered_set_of_languages_it_applies() -> None:
    """WIRE-57: the update accepts `languages`; its order carries the main language."""
    event = parse_client_event(_transcription({"languages": ["en", "de"]}))

    assert isinstance(event, SessionUpdate)
    setting = event.session.audio.input.transcription
    assert setting.allowed == (Language.EN, Language.DE)


@pytest.mark.parametrize(
    ("setting", "allowed"),
    [
        ({}, (Language.DE,)),
        ({"language": "en"}, (Language.EN,)),
        ({"languages": ["en"]}, (Language.EN,)),
    ],
    ids=["neither", "language", "one-language-in-the-set"],
)
def test_a_session_that_names_one_language_hears_exactly_that_one(
    setting: dict[str, object], allowed: tuple[Language, ...]
) -> None:
    event = parse_client_event(_transcription(setting))

    assert isinstance(event, SessionUpdate)
    assert event.session.audio.input.transcription.allowed == allowed


@pytest.mark.parametrize(
    ("setting", "param"),
    [
        ({"language": "de", "languages": ["de"]}, f"{_PATH}.languages"),
        ({"languages": ["de", "de"]}, f"{_PATH}.languages"),
        ({"languages": []}, f"{_PATH}.languages"),
        ({"languages": ["de", "fr"]}, f"{_PATH}.languages.1"),
        ({"language": None}, f"{_PATH}.language"),
        ({"languages": None}, f"{_PATH}.languages"),
        ({"language": None, "languages": ["de"]}, f"{_PATH}.language"),
        ({"language": "de", "languages": None}, f"{_PATH}.languages"),
    ],
    ids=[
        "both",
        "duplicate",
        "empty",
        "unknown-code",
        "null-language",
        "null-languages",
        "null-language-with-languages",
        "null-languages-with-language",
    ],
)
def test_a_set_the_service_cannot_hear_is_refused_naming_the_field_at_fault(
    setting: dict[str, object], param: str
) -> None:
    """WIRE-58: each refusal names the dotted path of the field at fault."""
    with pytest.raises(ClientEventError) as refused:
        parse_client_event(_transcription(setting))

    assert refused.value.param == param
    assert refused.value.event_id == "e1"


@pytest.mark.parametrize(
    ("setting", "echoed"),
    [
        ({"languages": ["en", "de"]}, {"model": MODEL, "languages": ["en", "de"]}),
        ({"language": "en"}, {"model": MODEL, "language": "en"}),
        ({}, {"model": MODEL, "language": "de"}),
    ],
    ids=["languages", "language", "neither"],
)
def test_the_echo_names_the_setting_in_the_form_the_client_sent(
    setting: dict[str, object], echoed: dict[str, object]
) -> None:
    """WIRE-57: `session.updated` echoes the applied setting as it was sent."""
    event = parse_client_event(_transcription(setting))
    assert isinstance(event, SessionUpdate)

    sent = json.loads(SessionUpdated(session=event.session).model_dump_json())

    assert sent["session"]["audio"]["input"]["transcription"] == echoed
    reparsed = parse_client_event(_session_update(sent["session"]))
    assert reparsed == event


def test_session_created_carries_no_languages_key() -> None:
    created = SessionCreated(
        session=TranscriptionSession(type="transcription")
    ).model_dump_json()

    assert "languages" not in created
    assert parse_server_event(created).type == "session.created"


def test_a_session_update_whose_session_names_no_type_is_refused() -> None:
    with pytest.raises(ClientEventError) as refused:
        parse_client_event(_session_update({}))

    assert refused.value.param == "session.type"


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        json.dumps({"type": "input_audio_buffer.clear"}),
        json.dumps({"type": "response.create"}),
        json.dumps({"type": "input_audio_buffer.append"}),
        json.dumps({"type": "input_audio_buffer.commit", "extra": 1}),
    ],
    ids=["malformed", "not-an-object", "clear", "unknown", "no-audio", "extra-field"],
)
def test_anything_but_the_three_client_events_is_refused(text: str) -> None:
    with pytest.raises(ClientEventError):
        parse_client_event(text)


def test_an_append_and_a_commit_are_parsed_with_their_event_ids() -> None:
    append = parse_client_event(
        json.dumps({"type": "input_audio_buffer.append", "audio": "AAA="})
    )
    commit = parse_client_event(
        json.dumps({"type": "input_audio_buffer.commit", "event_id": "c1"})
    )

    assert (append, commit) == (
        AudioAppend(type="input_audio_buffer.append", audio="AAA="),
        AudioCommit(type="input_audio_buffer.commit", event_id="c1"),
    )


def test_the_default_session_is_german_pcm_at_24_khz_with_server_detection() -> None:
    created = json.loads(
        SessionCreated(
            session=TranscriptionSession(type="transcription")
        ).model_dump_json()
    )

    assert created["session"] == {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": MODEL, "language": "de"},
                "turn_detection": {"type": "server_vad"},
            }
        },
    }


def test_every_server_event_carries_a_fresh_event_id() -> None:
    """WIRE-78: every server event has a fresh event_id."""
    first = TranscriptDelta(item_id="item", delta="a")
    second = TranscriptDelta(item_id="item", delta="a")

    assert first.event_id != second.event_id


def test_an_error_event_names_the_closed_code_the_field_and_the_client_event() -> None:
    event = json.loads(
        error_event(RefusalCode.INVALID_REQUEST, "audio", "c1").model_dump_json()
    )

    assert event["type"] == "error"
    assert event["error"] == {
        "message": refusal(RefusalCode.INVALID_REQUEST).error.message,
        "type": "invalid_request_error",
        "code": "invalid_request",
        "param": "audio",
        "event_id": "c1",
    }


def test_a_failed_item_carries_a_closed_list_error_without_an_event_id() -> None:
    event = json.loads(
        TranscriptFailed(
            item_id="item", error=item_failure(RefusalCode.BUSY)
        ).model_dump_json()
    )

    assert event["error"]["code"] == "busy"
    assert "event_id" not in event["error"]


def test_the_committed_item_with_no_predecessor_names_null() -> None:
    event = json.loads(
        AudioCommitted(item_id="i", previous_item_id=None).model_dump_json()
    )

    assert event["previous_item_id"] is None


SAMPLE_RATE = 22050
CARD_MEMORY_MB = 7


def _voice_health_body() -> dict[str, object]:
    return {
        "model": SpeakingModel.PIPER.value,
        "name": "A name",
        "languages": [Language.DE.value],
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


def test_a_health_body_names_a_voice_this_client_does_not_know() -> None:
    body = _health_body()
    body["speaking"] = {
        "ready": True,
        "voices": {"a-later-voice": _voice_health_body()},
    }

    assert "a-later-voice" in Health.model_validate(body).speaking.voices


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


def _each_server_event() -> list[ServerEvent]:
    session = TranscriptionSession(type="transcription")
    return [
        SessionCreated(session=session),
        SessionUpdated(session=session),
        SpeechStarted(item_id="item", audio_start_ms=10),
        TranscriptDelta(item_id="item", delta="Wie"),
        SpeechStopped(item_id="item", audio_end_ms=900),
        AudioCommitted(item_id="item", previous_item_id=None),
        ItemCreated(previous_item_id=None, item={"id": "item"}),
        TranscriptCompleted(
            item_id="item",
            transcript="Wie spät",
            usage=DurationUsage(seconds=1.5),
            languages=(TranscriptLanguage(code=Language.DE),),
        ),
        TranscriptFailed(
            item_id="item", error=item_failure(RefusalCode.BUSY, param="audio")
        ),
        error_event(RefusalCode.BUSY, param="audio"),
    ]


@pytest.mark.parametrize("event", _each_server_event(), ids=lambda e: e.type)
def test_a_server_event_is_parsed_into_its_own_class(event: ServerEvent) -> None:
    parsed = parse_server_event(event.model_dump_json())

    assert type(parsed) is type(event)
    assert parsed == event


def _started(**changes: object) -> str:
    """A `SpeechStarted` event as text, with `changes` applied to its fields."""
    event = SpeechStarted(item_id="i", audio_start_ms=1).model_dump()
    event.update(changes)
    return json.dumps(event)


def _started_without(field: str) -> str:
    event = json.loads(_started())
    del event[field]
    return json.dumps(event)


_DEEP = "[" * 100000 + "]" * 100000


@pytest.mark.parametrize(
    "text",
    [
        _started(type="no.such.event"),
        _started_without("type"),
        "not json",
        "[]",
        _DEEP,
        _started_without("event_id"),
        _started_without("audio_start_ms"),
        _started()[:-1] + ',"extra":' + _DEEP + "}",
        json.dumps(
            {
                "type": "session.created",
                "event_id": "e",
                "session": {"type": "transcription", "unknown": 1},
            }
        ),
    ],
    ids=[
        "unknown type",
        "no type",
        "not json",
        "json array",
        "deeply nested array",
        "no event id",
        "fails validation",
        "deeply nested field",
        "unknown key inside session",
    ],
)
def test_a_malformed_server_event_is_refused(text: str) -> None:
    with pytest.raises(ServerEventError):
        parse_server_event(text)


def test_a_completed_item_names_exactly_the_one_language_it_was_heard_in() -> None:
    """WIRE-136: `languages` holds exactly one element, the language of the decode."""
    event = TranscriptCompleted(
        item_id="item",
        transcript="",
        usage=DurationUsage(seconds=1.0),
        languages=(TranscriptLanguage(code=Language.EN),),
    )

    sent = json.loads(event.model_dump_json())

    assert sent["languages"] == [{"code": "en"}]


@pytest.mark.parametrize(
    "languages",
    [[], [{"code": "de"}, {"code": "en"}], [{"code": "fr"}]],
    ids=["none", "two", "unknown-code"],
)
def test_a_completed_item_without_exactly_one_known_language_is_not_an_event(
    languages: list[dict[str, str]],
) -> None:
    event = TranscriptCompleted(
        item_id="item",
        transcript="Wie",
        usage=DurationUsage(seconds=1.0),
        languages=(TranscriptLanguage(code=Language.DE),),
    ).model_dump(mode="json")

    with pytest.raises(ServerEventError):
        parse_server_event(json.dumps({**event, "languages": languages}))


def test_a_server_event_ignores_a_key_it_does_not_know() -> None:
    delta = TranscriptDelta(item_id="item", delta="Wie")
    text = json.dumps({**delta.model_dump(), "unknown": "x"})

    assert parse_server_event(text) == delta


def test_a_refused_server_event_leaks_none_of_its_content() -> None:
    words = "distinctive transcript"
    event = TranscriptCompleted(
        item_id="item",
        transcript=words,
        usage=DurationUsage(seconds=1.0),
        languages=(TranscriptLanguage(code=Language.DE),),
    ).model_dump()
    del event["usage"]

    with pytest.raises(ServerEventError) as raised:
        parse_server_event(json.dumps(event))

    error = raised.value
    assert error.__cause__ is None
    assert error.__context__ is None
    assert words not in str(error)
    assert words not in "".join(traceback.format_exception(error))


def test_the_catalogue_covers_every_voice() -> None:
    assert set(VOICES) == set(VoiceId)


@pytest.mark.parametrize("voice", list(VoiceId))
def test_a_request_pairs_each_voice_with_its_model(voice: VoiceId) -> None:
    body = speech_request_body("Hallo", voice, ResponseFormat.WAV)

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


@pytest.mark.parametrize("voice", ["piper", "qwen3-tts-0.6b", "alloy", "", {"id": "x"}])
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


@pytest.mark.parametrize("voice", list(VoiceId))
def test_a_request_speaks_the_first_language_of_its_voice(voice: VoiceId) -> None:
    request = SpeechRequest.model_validate_json(
        speech_request_body("Hallo", voice, ResponseFormat.WAV)
    )

    assert request.spoken_language is VOICES[voice].languages[0]
