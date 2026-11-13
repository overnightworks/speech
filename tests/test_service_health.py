"""GET /health answers the contract, including while the model is still loading."""

import pytest
from fastapi.testclient import TestClient
from speech_provider_contract import QWEN_SAMPLE_RATE

from speech_client.wire import VOICES, VoiceHealth, VoiceId
from speech_service.service import Runtime
from speech_service.voices import PIPER_VOICES
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
            VoiceId.THORSTEN.value: {
                "model": VOICES[VoiceId.THORSTEN].model,
                "name": VOICES[VoiceId.THORSTEN].name,
                "languages": [*VOICES[VoiceId.THORSTEN].languages],
                "takes_instructions": VOICES[VoiceId.THORSTEN].takes_instructions,
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

    voice = body["speaking"]["voices"][VoiceId.THORSTEN.value]
    assert voice["sample_rate"] == QWEN_SAMPLE_RATE


def test_health_reports_ready_models() -> None:
    """WIRE-124, WIRE-126: health reports ready models."""
    card_memory_mb = 9
    body = TestClient(an_app(memory_probe=lambda: card_memory_mb)).get("/health").json()

    voice = body["speaking"]["voices"][VoiceId.THORSTEN.value]
    assert body["speaking"]["ready"] is True
    assert voice["ready"] is True
    assert voice["sample_rate"] == SPEAK_SAMPLE_RATE
    assert body["card_memory_mb"] == card_memory_mb


@pytest.mark.parametrize("voice", list(VoiceId))
def test_a_voice_that_is_not_loaded_reports_its_catalogue_facts_not_ready(
    voice: VoiceId,
) -> None:
    """WIRE-125, WIRE-126: a voice that is not loaded reports its catalogue facts."""
    runtime = Runtime({voice: None})

    reported = runtime.health().speaking.voices[voice.value]

    assert reported == VoiceHealth(
        model=VOICES[voice].model,
        name=VOICES[voice].name,
        languages=VOICES[voice].languages,
        takes_instructions=VOICES[voice].takes_instructions,
        ready=False,
        sample_rate=reported.sample_rate,
    )
    assert reported.sample_rate == (0 if voice in PIPER_VOICES else QWEN_SAMPLE_RATE)


def test_health_lists_exactly_the_offered_voices() -> None:
    """WIRE-125: health lists exactly the offered voices."""
    offered = {VoiceId.NORMAN: FakeSpeaking(), VoiceId.CORI: FakeSpeaking()}

    health = Runtime(offered, memory_probe=lambda: 0).health()

    assert set(health.speaking.voices) == {voice.value for voice in offered}
