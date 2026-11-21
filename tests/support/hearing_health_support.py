"""Health answers the hearing proof's tests share."""

from __future__ import annotations

from speech_client.wire import Health


def health_of(*, speaking: bool, hearing: bool) -> Health:
    return Health.model_validate(
        {
            "speaking": {"ready": speaking, "voices": {}},
            "hearing": {"model": "small", "ready": hearing},
            "card_memory_mb": 0,
        }
    )


HEALTH_READY = health_of(speaking=True, hearing=True)
HEALTH_NO_HEARING = health_of(speaking=True, hearing=False)
HEALTH_LOADING = health_of(speaking=False, hearing=False)
