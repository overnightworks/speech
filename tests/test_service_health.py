"""GET /health answers the contract, including while the model is still loading."""

from fastapi.testclient import TestClient
from speech_provider_contract import QWEN_SAMPLE_RATE

from speech_client.wire import VoiceHealth
from speech_service.service import Runtime
from speech_service.voices import VoiceId, model_for_voice
from tests.support.fakes_support import (
    SPEAK_SAMPLE_RATE,
    FakeSpeaking,
    an_app,
)


def test_health_answers_while_models_are_not_ready() -> None:
    """WIRE-122, WIRE-123, WIRE-124, WIRE-125, WIRE-126: health answers mid-load."""
    card_memory_mb = 17
    client = TestClient(
        an_app(
            FakeSpeaking(ready=False),
            memory_probe=lambda: card_memory_mb,
        ),
    )

    body = client.get("/health").json()

    assert set(body) == {"speaking", "hearing", "card_memory_mb"}
    assert body["speaking"] == {
        "ready": False,
        "voices": {
            VoiceId.PIPER.value: {
                "model": "fake-voice",
                "ready": False,
                "sample_rate": SPEAK_SAMPLE_RATE,
            }
        },
    }
    assert body["card_memory_mb"] == card_memory_mb


def test_health_reports_a_voices_sample_rate() -> None:
    """WIRE-125: health reports a voices sample rate."""
    speaking = FakeSpeaking()
    speaking.sample_rate = QWEN_SAMPLE_RATE

    body = TestClient(an_app(speaking, memory_probe=lambda: 1)).get("/health").json()

    voice = body["speaking"]["voices"][VoiceId.PIPER.value]
    assert voice["sample_rate"] == QWEN_SAMPLE_RATE


def test_health_reports_ready_models() -> None:
    """WIRE-124, WIRE-126: health reports ready models."""
    card_memory_mb = 9
    body = TestClient(an_app(memory_probe=lambda: card_memory_mb)).get("/health").json()

    voice = body["speaking"]["voices"][VoiceId.PIPER.value]
    assert body["speaking"]["ready"] is True
    assert voice["ready"] is True
    assert voice["sample_rate"] == SPEAK_SAMPLE_RATE
    assert body["card_memory_mb"] == card_memory_mb


def test_a_voice_that_is_not_loaded_reports_its_catalogue_model_not_ready() -> None:
    """WIRE-125, WIRE-126: a voice that is not loaded reports its catalogue model."""
    runtime = Runtime({VoiceId.PIPER: None})

    voice = runtime.health().speaking.voices[VoiceId.PIPER.value]

    assert voice == VoiceHealth(
        model=model_for_voice(VoiceId.PIPER), ready=False, sample_rate=0
    )
