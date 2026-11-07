"""The voice a request names is judged by the catalogue, after the tokens."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_422_UNPROCESSABLE_CONTENT

from speech_runtime_client.wire import RefusalCode
from tests.support.identity_support import valid_callers
from tests.support.speak_support import (
    assert_refusal,
    post_speech,
    speak_app,
    speak_body,
)


def test_missing_tokens_are_unauthorized_before_the_voice_is_checked() -> None:
    """WIRE-16: missing tokens are unauthorized before the voice is checked."""
    response = post_speech(
        TestClient(speak_app(callers=valid_callers())), speak_body("alloy"), {}
    )

    assert_refusal(response, HTTP_401_UNAUTHORIZED, RefusalCode.UNAUTHORIZED, None)


@pytest.mark.parametrize("voice", ["alloy", "no-such-voice", "unknown"])
def test_a_voice_outside_the_catalogue_is_invalid(voice: str) -> None:
    """WIRE-19: a voice outside the catalogue is invalid."""
    response = post_speech(
        TestClient(speak_app(callers=valid_callers())), speak_body(voice)
    )

    assert_refusal(
        response, HTTP_422_UNPROCESSABLE_CONTENT, RefusalCode.INVALID_REQUEST, "voice"
    )
