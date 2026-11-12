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

from speech_client.wire import VOICES, RefusalCode, VoiceId
from speech_service.service import Runtime
from tests.support.fakes_support import FakeSpeaking
from tests.support.identity_support import valid_callers
from tests.support.speak_support import (
    app_over,
    assert_refusal,
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


@pytest.mark.parametrize("voice", list(VoiceId))
def test_each_voice_is_spoken_by_its_own_engine_in_its_first_language(
    voice: VoiceId,
) -> None:
    """WIRE-07, WIRE-11: each voice is spoken by its own engine."""
    engines = {offered: FakeSpeaking() for offered in VoiceId}
    app = app_over(Runtime(dict(engines), memory_probe=lambda: 0))

    response = post_speech(TestClient(app), speak_body(voice, "Guten Morgen"))

    assert response.status_code == HTTP_200_OK
    assert {offered: engine.heard_text for offered, engine in engines.items()} == {
        offered: "Guten Morgen" if offered is voice else "" for offered in engines
    }
    assert engines[voice].heard_language == VOICES[voice].languages[0].value
