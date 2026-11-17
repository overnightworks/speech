"""The client at HEAD reads the answers the last released image gives."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from speech_client.health import fetch_health
from speech_client.speak import Spoken, speak
from speech_client.wire import Health, ResponseFormat, VoiceId

if TYPE_CHECKING:
    from support.listener_support import RawListener

# The service of image v2026.11.14.6 over fake engines, as `GET /health` and a
# wav `POST /v1/audio/speech` for one Piper voice answered them.
RELEASED_ANSWERS = Path(__file__).parent / "released_answers" / "v2026.11.14.6"
DEADLINE_SECONDS = 30.0
CAP_BYTES = 1024 * 1024


def _chunked(head: str, body: bytes) -> bytes:
    return (
        f"HTTP/1.1 200 OK\r\n{head}\r\nTransfer-Encoding: chunked\r\n\r\n".encode()
        + b"%x\r\n%b\r\n0\r\n\r\n" % (len(body), body)
    )


def test_the_health_of_the_last_released_image_gives_no_marks(
    raw_listener: RawListener,
) -> None:
    body = (RELEASED_ANSWERS / "health.json").read_bytes()
    raw = f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body

    with raw_listener(raw) as (base_url, _):
        outcome = fetch_health(
            base_url, deadline_seconds=DEADLINE_SECONDS, cap_bytes=CAP_BYTES
        )

    assert isinstance(outcome, Health)
    assert {
        voice.gives_marks.words or voice.gives_marks.sounds
        for voice in outcome.speaking.voices.values()
    } == {False}


def test_the_speak_answer_of_the_last_released_image_is_spoken(
    raw_listener: RawListener,
) -> None:
    audio = (RELEASED_ANSWERS / "speak.wav").read_bytes()
    raw = _chunked("Content-Type: audio/wav", audio)

    with raw_listener(raw) as (base_url, _):
        outcome = speak(
            base_url,
            "Hallo Welt.",
            VoiceId.THORSTEN,
            response_format=ResponseFormat.WAV,
            workload_token="workload-token",  # noqa: S106 - a stand-in the raw listener never checks
            person_token="person-token",  # noqa: S106 - a stand-in the raw listener never checks
            deadline_seconds=DEADLINE_SECONDS,
            cap_bytes=CAP_BYTES,
        )

    assert outcome == Spoken(audio=audio)
