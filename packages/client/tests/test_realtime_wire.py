"""The Realtime transcription wire: the session, its client and server events."""

import json
import traceback

import pytest

from speech_client.realtime_wire import (
    AudioAppend,
    AudioCommit,
    AudioCommitted,
    ClientEventError,
    DurationUsage,
    ItemCreated,
    ServerEvent,
    ServerEventError,
    SessionCreated,
    SessionUpdate,
    SessionUpdated,
    SoundDetected,
    SoundName,
    SpeechStarted,
    SpeechStopped,
    SpeechSustained,
    SpeechSustainedFrom,
    SustainFrom,
    TranscriptCompleted,
    TranscriptDelta,
    TranscriptFailed,
    TranscriptionSession,
    TranscriptLanguage,
    error_event,
    input_milliseconds,
    item_failure,
    parse_client_event,
    parse_server_event,
)
from speech_client.wire import (
    MODEL,
    Language,
    RefusalCode,
    refusal,
)

_SUSTAINED_DURATION = "audio.input.sustained_speech.duration_ms"


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
        ({"sound_detection": {"type": "other"}}, "audio.input.sound_detection.type"),
        ({"sound_detection": {}}, "audio.input.sound_detection.type"),
        ({"sound_detection": None}, "audio.input.sound_detection"),
        ({"sustained_speech": None}, "audio.input.sustained_speech"),
        ({"sustained_speech": {"duration_ms": None}}, _SUSTAINED_DURATION),
        ({"sustained_speech": {"duration_ms": "400"}}, _SUSTAINED_DURATION),
        ({"sustained_speech": {"duration_ms": 400.5}}, _SUSTAINED_DURATION),
        ({"sustained_speech": {"duration_ms": True}}, _SUSTAINED_DURATION),
        ({"sustained_speech": {"duration_ms": 0}}, _SUSTAINED_DURATION),
        ({"sustained_speech": {"duration_ms": -1}}, _SUSTAINED_DURATION),
        (
            {"sustained_speech": {"duration_ms": 400, "pause_ms": 100}},
            "audio.input.sustained_speech.pause_ms",
        ),
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
        "other-sound-detection",
        "sound-detection-without-type",
        "null-sound-detection",
        "null-sustained-speech",
        "null-duration",
        "string-duration",
        "float-duration",
        "bool-duration",
        "zero-duration",
        "negative-duration",
        "unknown-sustained-speech-key",
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


def test_a_sound_detection_ask_is_echoed_and_an_absent_one_is_never_dumped() -> None:
    asked = parse_client_event(
        _session_update(
            {
                "type": "transcription",
                "audio": {"input": {"sound_detection": {"type": "audioset"}}},
            }
        )
    )
    assert isinstance(asked, SessionUpdate)

    echoed = json.loads(SessionUpdated(session=asked.session).model_dump_json())
    plain = SessionUpdated(
        session=TranscriptionSession(type="transcription")
    ).model_dump_json()

    assert echoed["session"]["audio"]["input"]["sound_detection"] == {
        "type": "audioset"
    }
    assert "sound_detection" not in plain


def _asking_for_sustained_speech(ask: dict[str, object]) -> SessionUpdate:
    event = parse_client_event(
        _session_update(
            {
                "type": "transcription",
                "audio": {"input": {"sustained_speech": ask}},
            }
        )
    )
    assert isinstance(event, SessionUpdate)
    return event


@pytest.mark.parametrize(
    "ask", [{}, {"duration_ms": 250}], ids=["without-a-duration", "with-a-duration"]
)
def test_a_sustained_speech_ask_is_echoed_as_sent_and_an_absent_one_is_never_dumped(
    ask: dict[str, object],
) -> None:
    """WIRE-172, WIRE-174: the ask's part is dumped as sent; an absent part never is."""
    echoed = json.loads(
        SessionUpdated(
            session=_asking_for_sustained_speech(ask).session
        ).model_dump_json()
    )
    plain = SessionUpdated(
        session=TranscriptionSession(type="transcription")
    ).model_dump_json()

    assert echoed["session"]["audio"]["input"]["sustained_speech"] == ask
    assert "sustained_speech" not in plain


def test_a_session_applies_the_duration_it_was_given_to_its_ask() -> None:
    """WIRE-174: the applied duration replaces whatever the ask held."""
    asked = _asking_for_sustained_speech({}).session

    applied = asked.with_sustained_duration(400)

    sent = json.loads(SessionUpdated(session=applied).model_dump_json())
    assert sent["session"]["audio"]["input"]["sustained_speech"] == {"duration_ms": 400}


def test_a_session_without_the_ask_is_what_the_ask_leaves_out() -> None:
    asked = TranscriptionSession.model_validate(
        {
            "type": "transcription",
            "audio": {"input": {"sound_detection": {"type": "audioset"}}},
        }
    )

    assert asked.without_sound_detection() == TranscriptionSession(type="transcription")


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
def test_anything_but_the_four_client_events_is_refused(text: str) -> None:
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
        SoundDetected(sound=SoundName.LAUGHTER, audio_start_ms=1500, audio_end_ms=3500),
        SpeechSustained(item_id="item", audio_end_ms=416),
        SpeechSustainedFrom(item_id="item", from_ms=96, audio_end_ms=416),
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


def test_a_completed_item_parses_with_and_without_a_traceparent() -> None:
    """WIRE-75, WIRE-150: the field is optional on reading."""
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    event = TranscriptCompleted(
        item_id="item",
        transcript="Wie",
        usage=DurationUsage(seconds=1.0),
        languages=(TranscriptLanguage(code=Language.DE),),
        traceparent=traceparent,
    )
    older = json.loads(event.model_dump_json())
    del older["traceparent"]

    parsed = parse_server_event(event.model_dump_json())
    assert isinstance(parsed, TranscriptCompleted)
    assert parsed.traceparent == traceparent
    undated = parse_server_event(json.dumps(older))
    assert isinstance(undated, TranscriptCompleted)
    assert undated.traceparent is None


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


def test_a_sound_event_carries_the_audioset_display_name_and_no_item() -> None:
    event = json.loads(
        SoundDetected(
            sound=SoundName.CLAPPING, audio_start_ms=500, audio_end_ms=2500
        ).model_dump_json()
    )

    assert event["type"] == "input_audio_buffer.sound_detected"
    assert event["sound"] == "Clapping"
    assert "item_id" not in event
    assert {name.value for name in SoundName} == {
        "Laughter",
        "Cough",
        "Clapping",
        "Screaming",
    }


def test_a_sustained_speech_event_names_its_item_and_the_end_of_the_window() -> None:
    """WIRE-171: the event carries `item_id` and `audio_end_ms`."""
    event = json.loads(
        SpeechSustained(item_id="item", audio_end_ms=416).model_dump_json()
    )

    assert event["type"] == "input_audio_buffer.speech_sustained"
    assert (event["item_id"], event["audio_end_ms"]) == ("item", 416)


def test_a_moment_event_names_its_item_the_moment_and_the_end_of_the_window() -> None:
    """WIRE-177: the event carries `item_id`, `from_ms` and `audio_end_ms`."""
    event = json.loads(
        SpeechSustainedFrom(
            item_id="item", from_ms=96, audio_end_ms=416
        ).model_dump_json()
    )

    assert event["type"] == "input_audio_buffer.speech_sustained_from"
    assert (event["item_id"], event["from_ms"], event["audio_end_ms"]) == (
        "item",
        96,
        416,
    )


def _moment(**fields: object) -> str:
    return json.dumps({"type": "input_audio_buffer.sustain_from", **fields})


def test_a_moment_is_parsed_with_its_position_and_event_id() -> None:
    """WIRE-176: a client names a moment with an integer `from_ms` from 0."""
    event = parse_client_event(_moment(from_ms=0, event_id="m1"))

    assert event == SustainFrom(
        type="input_audio_buffer.sustain_from", from_ms=0, event_id="m1"
    )


@pytest.mark.parametrize(
    ("fields", "param"),
    [
        pytest.param({}, "from_ms", id="absent"),
        pytest.param({"from_ms": None}, "from_ms", id="null"),
        pytest.param({"from_ms": "5"}, "from_ms", id="text"),
        pytest.param({"from_ms": 5.5}, "from_ms", id="a-fraction"),
        pytest.param({"from_ms": True}, "from_ms", id="a-boolean"),
        pytest.param({"from_ms": -1}, "from_ms", id="below-zero"),
        pytest.param({"from_ms": 5, "other": 1}, "other", id="an-unknown-key"),
    ],
)
def test_a_moment_the_parser_refuses_names_the_field_and_its_type(
    fields: dict[str, object], param: str
) -> None:
    """WIRE-176: the refusal names the field, and the raw type is kept for the ask."""
    with pytest.raises(ClientEventError) as refused:
        parse_client_event(_moment(**fields, event_id="m1"))

    assert (refused.value.param, refused.value.event_id, refused.value.event_type) == (
        param,
        "m1",
        "input_audio_buffer.sustain_from",
    )


def test_the_milliseconds_of_the_input_audio_round_down() -> None:
    """WIRE-70: 24 kHz PCM16 is 48 bytes a millisecond, and a part of one is dropped."""
    assert [input_milliseconds(size) for size in (0, 47, 48, 3004 * 2)] == [
        0,
        0,
        1,
        125,
    ]
