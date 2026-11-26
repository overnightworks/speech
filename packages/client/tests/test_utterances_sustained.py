"""The sustained-speech ask and event of the finished-utterance helper."""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING

import pytest

from packages.client.tests.support.utterances_support import (
    Recorder,
    completed,
    hear,
    hearing,
    service_saying,
    started,
    stopped,
    sustained,
    utterance_of,
)
from speech_client.realtime_wire import (
    ServerEventError,
    SustainedSpeech,
    parse_client_event,
    session_update,
)
from speech_client.utterances import (
    SpeechStart,
    SpeechStop,
    SustainedVoice,
)
from speech_client.wire import Language

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection


@pytest.mark.parametrize(
    ("callback", "duration_ms", "ask"),
    [
        pytest.param(True, 2500, SustainedSpeech(duration_ms=2500), id="with-duration"),
        pytest.param(True, None, SustainedSpeech(), id="default-duration"),
        pytest.param(False, None, None, id="without-callback"),
    ],
)
def test_the_session_asks_for_sustained_speech_only_when_the_app_passes_the_callback(
    *, callback: bool, duration_ms: int | None, ask: SustainedSpeech | None
) -> None:
    arrived: list[str | bytes] = []

    async def records(connection: ServerConnection) -> None:
        arrived.append(await connection.recv())

    asyncio.run(
        hear(
            functools.partial(service_saying, records),
            sustained=callback,
            sustained_speech_ms=duration_ms,
        )
    )

    (first,) = arrived
    assert parse_client_event(first).session.audio.input.sustained_speech == ask
    if not callback:
        assert first == session_update((Language.DE,))
    elif duration_ms is None:
        assert b"duration_ms" not in first.encode()


@pytest.mark.parametrize(
    ("callback", "duration_ms"),
    [
        pytest.param(False, 2500, id="duration-without-callback"),
        pytest.param(True, 0, id="duration-below-the-bound"),
    ],
)
def test_a_duration_the_ask_cannot_carry_raises_before_any_connection(
    *, callback: bool, duration_ms: int
) -> None:
    connected: list[bool] = []

    async def records(connection: ServerConnection) -> None:
        connected.append(True)
        await connection.wait_closed()

    with pytest.raises(ValueError, match="sustained"):
        asyncio.run(
            hear(
                functools.partial(service_saying, records),
                sustained=callback,
                sustained_speech_ms=duration_ms,
            )
        )

    assert connected == []


def test_sustained_speech_reaches_its_callback_between_the_items_start_and_stop() -> (
    None
):
    recorder = Recorder()

    asyncio.run(
        hear(
            hearing(
                started("a", 100),
                sustained("a", 2600),
                stopped("a", 4000),
                completed("a", "ha"),
            ),
            None,
            recorder,
            speech_marks=True,
            sustained=True,
        )
    )

    assert recorder.got == [
        SpeechStart(100),
        SustainedVoice("a", 2600),
        SpeechStop(4000),
        utterance_of("ha", start=100, end=4000),
    ]


def test_sustained_speech_the_app_did_not_ask_for_raises_a_server_event_error() -> None:
    recorder = Recorder()

    with pytest.raises(ServerEventError):
        asyncio.run(
            hear(
                hearing(sustained("a", 2600), stays_open=True),
                None,
                recorder,
            )
        )

    assert recorder.got == []
