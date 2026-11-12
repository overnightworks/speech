"""The voice a request names is judged by the catalogue, after the tokens."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_401_UNAUTHORIZED,
    HTTP_422_UNPROCESSABLE_CONTENT,
)

from speech_client.wire import (
    VOICES,
    Language,
    RefusalCode,
    SpeakingModel,
    VoiceId,
)
from speech_service.service import Runtime
from speech_service.voices import PIPER_VOICES
from tests.support.fakes_support import FakeSpeaking
from tests.support.identity_support import valid_callers
from tests.support.speak_support import (
    app_over,
    assert_refusal,
    assert_unavailable,
    fake_runtime,
    post_speech,
    speak_app,
    speak_body,
)

if TYPE_CHECKING:
    from httpx2 import Response as ClientResponse


_VOICES_OF_ONE_LANGUAGE = [v for v in VoiceId if len(VOICES[v].languages) == 1]
_VOICES_OF_SEVERAL_LANGUAGES = [v for v in VoiceId if len(VOICES[v].languages) > 1]


def test_missing_tokens_are_unauthorized_before_the_voice_is_checked() -> None:
    """WIRE-16: missing tokens are unauthorized before the voice is checked."""
    body = json.loads(speak_body()) | {"voice": "alloy"}
    response = post_speech(
        TestClient(speak_app(callers=valid_callers())), json.dumps(body).encode(), {}
    )

    assert_refusal(response, HTTP_401_UNAUTHORIZED, RefusalCode.UNAUTHORIZED, None)


@pytest.mark.parametrize("voice", ["alloy", "no-such-voice", "unknown", "piper"])
def test_a_voice_outside_the_catalogue_is_invalid(voice: str) -> None:
    """WIRE-19: a voice outside the catalogue is invalid."""
    body = json.loads(speak_body()) | {"voice": voice}
    response = post_speech(
        TestClient(speak_app(callers=valid_callers())), json.dumps(body).encode()
    )

    assert_refusal(
        response, HTTP_422_UNPROCESSABLE_CONTENT, RefusalCode.INVALID_REQUEST, "voice"
    )


@pytest.mark.parametrize(
    ("voice", "language"),
    [(voice, language) for voice in VoiceId for language in VOICES[voice].languages],
)
def test_each_voice_is_spoken_by_its_own_engine_in_the_language_asked_for(
    voice: VoiceId, language: Language
) -> None:
    """WIRE-07, WIRE-11: each voice is spoken by its engine in the named language."""
    engines = {offered: FakeSpeaking() for offered in VoiceId}
    app = app_over(Runtime(dict(engines), memory_probe=lambda: 0))

    response = post_speech(
        TestClient(app), speak_body(voice, "Guten Morgen", language=language)
    )

    assert response.status_code == HTTP_200_OK
    assert {offered: engine.heard_text for offered, engine in engines.items()} == {
        offered: "Guten Morgen" if offered is voice else "" for offered in engines
    }
    assert engines[voice].heard_language == language


@pytest.mark.parametrize("voice", _VOICES_OF_ONE_LANGUAGE)
def test_a_voice_of_one_language_speaks_it_when_no_language_is_named(
    voice: VoiceId,
) -> None:
    """WIRE-11: a voice of one language speaks it without a language."""
    engine = FakeSpeaking()
    app = app_over(_offering_every_voice(voice, engine))

    response = post_speech(TestClient(app), speak_body(voice))

    assert "language" not in json.loads(speak_body(voice))
    assert response.status_code == HTTP_200_OK
    assert engine.heard_language == VOICES[voice].languages[0]


def _changed(**changes: object) -> bytes:
    return json.dumps(json.loads(speak_body()) | changes).encode()


@pytest.mark.parametrize(
    ("body", "param"),
    [
        pytest.param(_changed(model="speech"), "model", id="model-of-hearing"),
        pytest.param(_changed(model="speech", voice="alloy"), "model", id="both-wrong"),
        pytest.param(
            _changed(voice=SpeakingModel.PIPER), "voice", id="voice-is-a-model"
        ),
        pytest.param(
            _changed(voice=SpeakingModel.QWEN), "voice", id="voice-is-another-model"
        ),
        pytest.param(_changed(voice={"id": "alloy"}), "voice", id="voice-object"),
        *(
            pytest.param(
                _changed(model=model, voice=voice),
                "voice",
                id=f"{voice.name.lower()}-under-{model.name.lower()}",
            )
            for voice in VoiceId
            for model in SpeakingModel
            if model is not VOICES[voice].model
        ),
    ],
)
def test_a_wrong_model_or_voice_is_refused_naming_the_first_wrong_field(
    body: bytes, param: str
) -> None:
    """WIRE-05, WIRE-19, WIRE-137: a wrong model or voice is invalid_request."""
    response = post_speech(TestClient(speak_app()), body)

    assert_refusal(
        response, HTTP_422_UNPROCESSABLE_CONTENT, RefusalCode.INVALID_REQUEST, param
    )


@pytest.mark.parametrize("voice", [v for v in VoiceId if v is not VoiceId.THORSTEN])
def test_a_catalogue_voice_that_is_not_offered_is_unavailable(voice: VoiceId) -> None:
    """WIRE-19: a catalogue voice that is not offered is unavailable."""
    response = post_speech(TestClient(speak_app()), speak_body(voice))

    assert_unavailable(response)


@pytest.mark.parametrize("voice", PIPER_VOICES)
def test_a_piper_voice_is_unavailable_while_the_start_load_runs(
    voice: VoiceId,
) -> None:
    runtime = fake_runtime(voice=voice)
    runtime.begin_loading()

    assert_unavailable(post_speech(TestClient(app_over(runtime)), speak_body(voice)))


def _offering_every_voice(voice: VoiceId, engine: FakeSpeaking) -> Runtime:
    return Runtime(
        {
            offered: engine if offered is voice else FakeSpeaking()
            for offered in VoiceId
        },
        memory_probe=lambda: 0,
    )


def _body_naming(voice: VoiceId, /, **changes: object) -> bytes:
    return json.dumps(json.loads(speak_body(voice)) | changes).encode()


def _without_language(voice: VoiceId) -> bytes:
    body = json.loads(speak_body(voice))
    body.pop("language", None)
    return json.dumps(body).encode()


def _assert_language_refused(response: ClientResponse) -> None:
    assert_refusal(
        response,
        HTTP_422_UNPROCESSABLE_CONTENT,
        RefusalCode.INVALID_REQUEST,
        "language",
    )


@pytest.mark.parametrize(
    ("voice", "language"),
    [
        (voice, language)
        for voice in VoiceId
        for language in Language
        if language not in VOICES[voice].languages
    ],
)
def test_a_language_the_voice_lacks_is_refused_naming_language(
    voice: VoiceId, language: Language
) -> None:
    """WIRE-11: a language the voice does not speak is invalid_request on `language`."""
    engine = FakeSpeaking()
    app = app_over(_offering_every_voice(voice, engine))

    response = post_speech(TestClient(app), _body_naming(voice, language=language))

    _assert_language_refused(response)
    assert engine.heard_text == ""


@pytest.mark.parametrize("voice", _VOICES_OF_SEVERAL_LANGUAGES)
def test_a_voice_of_several_languages_without_a_language_is_refused(
    voice: VoiceId,
) -> None:
    """WIRE-11: a voice of several languages needs the language named."""
    engine = FakeSpeaking()
    app = app_over(_offering_every_voice(voice, engine))

    response = post_speech(TestClient(app), _without_language(voice))

    _assert_language_refused(response)
    assert engine.heard_text == ""


@pytest.mark.parametrize("language", ["fr", "", "DE", "de-DE", 1, ["de"]])
@pytest.mark.parametrize("voice", list(VoiceId))
def test_a_language_outside_the_wire_is_refused_naming_language(
    voice: VoiceId, language: object
) -> None:
    """WIRE-11: a language that is neither de nor en is invalid_request."""
    response = post_speech(
        TestClient(speak_app()), _body_naming(voice, language=language)
    )

    _assert_language_refused(response)


@pytest.mark.parametrize("voice", list(VoiceId))
def test_a_wrong_language_is_refused_before_the_voice_is_found_unavailable(
    voice: VoiceId,
) -> None:
    """WIRE-11, WIRE-19: every 422 comes before any 503."""
    response = post_speech(TestClient(speak_app()), _body_naming(voice, language="fr"))

    _assert_language_refused(response)


@pytest.mark.parametrize(
    ("changes", "param"),
    [
        pytest.param({"model": "speech"}, "model", id="refused-model"),
        pytest.param({"voice": "alloy"}, "voice", id="refused-voice"),
        pytest.param(
            {"model": SpeakingModel.QWEN}, "voice", id="voice-of-another-model"
        ),
    ],
)
def test_a_wrong_model_or_voice_is_named_before_a_wrong_language(
    changes: dict[str, object], param: str
) -> None:
    """WIRE-11: the language check follows the model and voice checks."""
    response = post_speech(
        TestClient(speak_app()),
        _body_naming(VoiceId.THORSTEN, language="fr", **changes),
    )

    assert_refusal(
        response, HTTP_422_UNPROCESSABLE_CONTENT, RefusalCode.INVALID_REQUEST, param
    )
