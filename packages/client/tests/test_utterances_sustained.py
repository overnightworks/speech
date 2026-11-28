"""The sustained-speech ask and event of the finished-utterance helper."""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING

import pytest

from packages.client.tests.support.hearing_support import CREATED_EVENT
from packages.client.tests.support.scripted_hearing_support import sustained_from
from packages.client.tests.support.utterances_support import (
    UNASKED_SESSION_UPDATE_DE,
    Browser,
    NamingBrowser,
    Recorder,
    binary_message,
    completed,
    hear,
    hearing,
    service_saying,
    started,
    stopped,
    sustained,
    utterance_of,
)
from speech_client.moments import SustainMoments
from speech_client.outcomes import Malformed
from speech_client.realtime_wire import (
    ServerEventError,
    SoundDetection,
    SpeechSustainedFrom,
    SustainedSpeech,
    parse_client_event,
)
from speech_client.utterances import (
    HearingCallbacks,
    SpeechStart,
    SpeechStop,
    SustainedFrom,
    SustainedVoice,
    hear_utterances,
)

# The one frame is 100 ms of 24 kHz audio; seven 32 ms windows are 200 ms of voice.
FIRST_FRAME_MS = 100
SEVEN_WINDOWS_MS = 224

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
        assert first == UNASKED_SESSION_UPDATE_DE
    elif duration_ms is None:
        assert b"duration_ms" not in first.encode()


def test_a_session_with_both_callbacks_asks_for_sounds_and_sustained_speech() -> None:
    arrived: list[str | bytes] = []

    async def records(connection: ServerConnection) -> None:
        arrived.append(await connection.recv())

    asyncio.run(
        hear(
            functools.partial(service_saying, records),
            sounds=True,
            sustained=True,
            sustained_speech_ms=2500,
        )
    )

    (first,) = arrived
    asked = parse_client_event(first).session.audio.input
    assert asked.sound_detection == SoundDetection(type="audioset")
    assert asked.sustained_speech == SustainedSpeech(duration_ms=2500)


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


async def _unused(_: object) -> None:
    raise AssertionError


@pytest.mark.parametrize(
    ("handle", "from_callback", "sustained_callback"),
    [
        pytest.param(True, False, True, id="a-handle-without-the-callback"),
        pytest.param(False, True, True, id="a-callback-without-the-handle"),
        pytest.param(True, True, False, id="both-without-on-speech-sustained"),
    ],
)
def test_a_moment_handle_or_callback_without_the_rest_raises_before_any_connection(
    *, handle: bool, from_callback: bool, sustained_callback: bool
) -> None:
    callbacks = HearingCallbacks(
        on_utterance=_unused,
        on_speech_sustained=_unused if sustained_callback else None,
        on_speech_sustained_from=_unused if from_callback else None,
    )

    with pytest.raises(ValueError, match=r"moments|on_speech_sustained"):
        asyncio.run(
            hear_utterances(
                Browser(),
                "http://127.0.0.1:1",
                token_source=_unused,
                languages=(),
                deadline_seconds=1,
                held_limit_bytes=1,
                ended=asyncio.Event(),
                callbacks=callbacks,
                moments=SustainMoments() if handle else None,
            )
        )


def test_a_handle_names_a_moment_only_while_a_hearing_runs() -> None:
    moments = SustainMoments()
    asked = 0

    def name() -> int:
        nonlocal asked
        asked += 1
        return 7

    before = moments.now()
    with moments.running(name):
        during = moments.now()
    after = moments.now()

    assert (before, during, after, asked) == (None, 7, None, 1)


def test_the_event_of_the_moment_the_app_named_reaches_its_callback() -> None:
    """The app names a moment at the audio handed over; its event follows `now()`."""
    moments = SustainMoments()
    browser = NamingBrowser(moments, (binary_message(4800),))
    recorder = Recorder(ends_after_first=True)

    async def answers_the_moment(connection: ServerConnection) -> None:
        await connection.recv()
        await connection.send(CREATED_EVENT)
        async for frame in connection:
            if "sustain_from" in str(frame):
                from_ms = parse_client_event(str(frame)).from_ms
                await connection.send(sustained_from("a", from_ms, from_ms + 224))

    asyncio.run(
        hear(
            functools.partial(service_saying, answers_the_moment),
            browser,
            recorder,
            sustained=True,
            sustained_speech_ms=200,
            moments=moments,
        )
    )

    assert browser.position == FIRST_FRAME_MS
    assert recorder.got == [SustainedFrom("a", FIRST_FRAME_MS + SEVEN_WINDOWS_MS)]


def test_an_event_of_a_moment_when_none_was_written_is_malformed() -> None:
    recorder = Recorder()

    outcome = asyncio.run(
        hear(
            hearing(
                SpeechSustainedFrom(item_id="a", from_ms=0, audio_end_ms=300),
                stays_open=True,
            ),
            None,
            recorder,
            sustained=True,
            sustained_speech_ms=200,
            moments=SustainMoments(),
        )
    )

    assert recorder.got == []
    assert outcome == Malformed()
