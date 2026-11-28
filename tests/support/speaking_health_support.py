"""Health answers of a speaking service, shared by the speaking proofs' tests."""

from __future__ import annotations

from shipped_catalogue import CATALOGUE
from speech_client.wire import (
    Health,
    HearingHealth,
    Language,
    SpeakingHealth,
    VoiceHealth,
    VoiceId,
)
from tests.support.fakes_support import SPEAK_SAMPLE_RATE


def health_offering(
    voice: VoiceId, languages: tuple[Language, ...], *, ready: bool = True
) -> Health:
    """The health of a service whose one voice speaks `languages`."""
    return Health(
        speaking=SpeakingHealth(
            ready=ready,
            voices={
                voice: VoiceHealth(
                    model=CATALOGUE[voice].engine.wire_model,
                    name=CATALOGUE[voice].name,
                    languages=languages,
                    takes_instructions=False,
                    ready=ready,
                    sample_rate=SPEAK_SAMPLE_RATE,
                )
            },
        ),
        hearing=HearingHealth(model="any", ready=True),
        card_memory_mb=0,
    )
