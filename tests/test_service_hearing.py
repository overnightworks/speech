"""Hearing starts beside speaking; a missing resource isolates it."""

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette.status import HTTP_200_OK, HTTP_503_SERVICE_UNAVAILABLE

from speech_client.wire import HearingHealth, RefusalCode
from speech_service import hearing_models
from speech_service.clock import LoopClock
from speech_service.config import Settings
from speech_service.hearing import Hearing, ResourceUnavailableError
from speech_service.hearing_models import (
    HearingModel,
    HearingModelMissingError,
)
from speech_service.service import Runtime, RuntimeDependencies
from speech_service.voices import VoiceId
from tests.support.fakes_support import FakeSpeaking
from tests.support.hear_support import (
    FakeHearing,
    hear_app,
    multipart_body,
    post_recording,
    recording,
    write_snapshot,
)
from tests.support.speak_support import assert_refusal, post_speech, speak_body

if TYPE_CHECKING:
    from pathlib import Path


def _starting(hearing: FakeHearing | None) -> Runtime:
    return Runtime(
        None,
        memory_probe=lambda: 0,
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: FakeSpeaking(),
            missing_artifacts=lambda _voice: (),
        ),
        hearing=Hearing(hearing, LoopClock()),
    )


def test_a_load_that_lacks_the_resource_leaves_speaking_up_hearing_unavailable() -> (
    None
):
    """WIRE-40, WIRE-129: a load that lacks the resource leaves speaking up."""
    engine = FakeHearing(load_error=ResourceUnavailableError("the card is full"))
    runtime = _starting(engine)

    runtime.load()
    client = TestClient(hear_app(runtime))

    health = client.get("/health").json()
    transcription = post_recording(client, multipart_body(file=recording()))
    speech = post_speech(client, speak_body(VoiceId.PIPER.value))
    assert health["hearing"]["ready"] is False
    assert health["speaking"]["ready"] is True
    assert_refusal(
        transcription, HTTP_503_SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE, None
    )
    assert speech.status_code == HTTP_200_OK
    assert engine.closed == 1


def test_a_load_that_fails_otherwise_stops_the_start_naming_the_setting() -> None:
    engine = FakeHearing(load_error=RuntimeError("weights are corrupt"))
    runtime = _starting(engine)

    with pytest.raises(RuntimeError, match="SPEECH_HEARING_MODEL"):
        runtime.load()

    assert engine.closed == 1


def test_a_loaded_engine_is_ready_in_health() -> None:
    """WIRE-128, WIRE-129: a loaded engine is ready in health."""
    engine = FakeHearing()
    runtime = _starting(engine)

    runtime.load()

    assert runtime.health().hearing == HearingHealth(
        model=engine.model_name, ready=True
    )


def test_closing_the_runtime_frees_the_hearing_engine() -> None:
    engine = FakeHearing()
    runtime = _starting(engine)
    runtime.load()

    runtime.close()

    assert engine.closed == 1


def test_without_a_hearing_model_health_names_hearing_unavailable() -> None:
    """WIRE-128: without a hearing model health names hearing unavailable."""
    runtime = _starting(None)

    runtime.load()

    assert runtime.health().hearing == HearingHealth(model="unavailable", ready=False)


def test_a_configured_model_that_was_not_fetched_stops_the_start(
    tmp_path: Path,
) -> None:
    settings = Settings(hearing_model=HearingModel.SMALL, huggingface_cache=tmp_path)

    with pytest.raises(HearingModelMissingError, match="SPEECH_HEARING_MODEL"):
        Runtime.from_settings(settings)


def test_a_configured_model_that_was_fetched_is_named_in_health_before_it_loads(
    tmp_path: Path,
) -> None:
    """WIRE-128, WIRE-129: a fetched model is named in health before it loads."""
    hearing_models.fetch_model(HearingModel.SMALL, tmp_path, write_snapshot)
    settings = Settings(hearing_model=HearingModel.SMALL, huggingface_cache=tmp_path)

    runtime = Runtime.from_settings(settings)

    assert runtime.hearing.health() == {
        "model": HearingModel.SMALL.value,
        "ready": False,
    }
