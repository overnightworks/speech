"""The voice a request names is judged by the catalogue, after the tokens."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_401_UNAUTHORIZED,
    HTTP_422_UNPROCESSABLE_CONTENT,
)

from speech_client.wire import VOICES, RefusalCode, SpeakingModel, VoiceId
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
    [
        (VoiceId.THORSTEN, "de"),
        (VoiceId.NORMAN, "en"),
        (VoiceId.LJSPEECH, "en"),
        (VoiceId.CORI, "en"),
        (VoiceId.RYAN, "de"),
    ],
)
def test_each_voice_is_spoken_by_its_own_engine_in_its_first_language(
    voice: VoiceId, language: str
) -> None:
    """WIRE-07, WIRE-11: each voice is spoken by its own engine."""
    engines = {offered: FakeSpeaking() for offered in VoiceId}
    app = app_over(Runtime(dict(engines), memory_probe=lambda: 0))

    response = post_speech(TestClient(app), speak_body(voice, "Guten Morgen"))

    assert response.status_code == HTTP_200_OK
    assert {offered: engine.heard_text for offered, engine in engines.items()} == {
        offered: "Guten Morgen" if offered is voice else "" for offered in engines
    }
    assert engines[voice].heard_language == language


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


def test_a_voice_of_another_model_is_refused_though_it_is_not_offered() -> None:
    body = _changed(model=SpeakingModel.PIPER, voice=VoiceId.RYAN)

    response = post_speech(TestClient(speak_app()), body)

    assert_refusal(
        response, HTTP_422_UNPROCESSABLE_CONTENT, RefusalCode.INVALID_REQUEST, "voice"
    )


@pytest.mark.parametrize("voice", PIPER_VOICES)
def test_a_piper_voice_is_unavailable_while_the_start_load_runs(
    voice: VoiceId,
) -> None:
    runtime = fake_runtime(voice=voice)
    runtime.begin_loading()

    assert_unavailable(post_speech(TestClient(app_over(runtime)), speak_body(voice)))
