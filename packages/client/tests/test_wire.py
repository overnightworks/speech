"""The wire: a `busy` answer's wait, the hearing request, its refusals, health."""

import json
from http import HTTPStatus

import pytest
from pydantic import ValidationError

from speech_runtime_client.wire import (
    MODEL,
    REFUSAL_STATUS,
    RETRY_AFTER_HEADER,
    AudioAppend,
    AudioCommit,
    AudioCommitted,
    BusyWait,
    ClientEventError,
    Health,
    HearingHealth,
    HearingLanguage,
    RefusalCode,
    RetryAfterError,
    SessionCreated,
    SessionUpdate,
    SpeakingHealth,
    TranscriptDelta,
    TranscriptFailed,
    TranscriptionRequest,
    TranscriptionSession,
    VoiceHealth,
    busy_wait,
    error_event,
    item_failure,
    parse_client_event,
    refusal,
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


@pytest.mark.parametrize("language", list(HearingLanguage))
def test_hearing_accepts_each_language_of_the_wire(language: HearingLanguage) -> None:
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
    assert event.session.audio.input.transcription.language is HearingLanguage.EN


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
        (
            {"transcription": {"languages": ["de"]}},
            "audio.input.transcription.languages",
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
        "languages",
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


def _health_body() -> dict[str, object]:
    return {
        "speaking": {
            "ready": True,
            "voices": {"v": {"model": "m", "ready": True, "sample_rate": SAMPLE_RATE}},
        },
        "hearing": {"model": "h", "ready": False},
        "card_memory_mb": CARD_MEMORY_MB,
    }


def test_a_health_body_parses_into_the_typed_answer() -> None:
    health = Health.model_validate(_health_body())

    assert health.speaking.voices["v"].sample_rate == SAMPLE_RATE
    assert health.hearing.ready is False
    assert health.card_memory_mb == CARD_MEMORY_MB


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
            voices={"v": VoiceHealth(model="m", ready=True, sample_rate=SAMPLE_RATE)},
        ),
        hearing=HearingHealth(model="h", ready=False),
        card_memory_mb=CARD_MEMORY_MB,
    )

    assert Health.model_validate(built.model_dump(mode="json")) == built


def test_a_health_answer_is_frozen() -> None:
    health = Health.model_validate(_health_body())

    with pytest.raises(ValidationError):
        health.card_memory_mb = 8
