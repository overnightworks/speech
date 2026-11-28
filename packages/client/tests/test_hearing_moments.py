"""A hearing names moments at the audio queued and delivers the newest one's event."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from packages.client.tests.support.hearing_support import session_to
from packages.client.tests.support.reconnect_support import (
    CHUNK,
    WHOLE_RUN_SECONDS,
    Tokens,
    collected,
    reads,
)
from packages.client.tests.support.scripted_hearing_support import (
    Visit,
    opens,
    says,
    scripted,
    sustained_from,
    takes_audio,
)
from speech_client.outcomes import Malformed
from speech_client.realtime_wire import SpeechSustainedFrom, SustainedSpeech
from speech_client.wire import DRAIN_CLOSE_CODE

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection

    from packages.client.tests.support.scripted_hearing_support import Script
    from speech_client.hearing import HearingSession

pytestmark = pytest.mark.usefixtures("quick_bound")

# A chunk is 100 ms of the browser's 24 kHz audio, and the cases count under 200 ms.
SUSTAINED = SustainedSpeech(duration_ms=200)
HELD_LIMIT = 100_000
QUEUED_BY_THEN_MS = 300


def _session(port: int, tokens: Tokens | None = None) -> HearingSession:
    return session_to(
        port,
        30,
        HELD_LIMIT,
        5,
        *(() if tokens is None else (tokens,)),
        sustained_speech=SUSTAINED,
    )


def _moments_heard(events: list[object]) -> list[SpeechSustainedFrom]:
    return [event for event in events if isinstance(event, SpeechSustainedFrom)]


async def _hear_all(session: HearingSession) -> list[object]:
    async with asyncio.timeout(WHOLE_RUN_SECONDS):
        return list(await collected(session))


def _after_moments(
    chunks: int, moments: int, *events: str, closes_with: int = 1000
) -> Script:
    """Open, take `chunks` chunks and `moments` moments, say `events`, then close."""
    return reads(chunks, moments=moments, says=events, closes_with=closes_with)


@pytest.mark.parametrize(
    ("chunk_sizes", "position_ms"),
    [
        pytest.param((len(CHUNK),) * 2, 200, id="whole-chunks"),
        pytest.param((len(CHUNK), 100), 102, id="rounded-down"),
        pytest.param((47,), 0, id="under-a-millisecond"),
    ],
)
def test_a_moment_is_written_after_the_audio_before_it_at_its_position(
    chunk_sizes: tuple[int, ...], position_ms: int
) -> None:
    """WIRE-176: the moment's `from_ms` is its position, written after the audio."""

    async def scenario() -> tuple[int | None, list[Visit]]:
        async with scripted(_after_moments(len(chunk_sizes) + 1, 1)) as (port, visits):
            async with _session(port) as session:
                for size in chunk_sizes:
                    session.send(bytes(size))
                named = session.sustain_from_now()
                session.send(bytes(2))
                await _hear_all(session)
            return named, visits

    named, visits = asyncio.run(scenario())

    assert named == position_ms
    assert visits[0].moments == [(len(chunk_sizes), position_ms)]


def test_a_moment_at_the_position_of_the_newest_is_not_written_again() -> None:
    """WIRE-178: an equal moment changes nothing."""

    async def scenario() -> tuple[tuple[int | None, int | None], list[Visit]]:
        async with scripted(_after_moments(2, 1)) as (port, visits):
            async with _session(port) as session:
                session.send(CHUNK)
                named = (session.sustain_from_now(), session.sustain_from_now())
                session.send(CHUNK)
                await _hear_all(session)
            return named, visits

    named, visits = asyncio.run(scenario())

    assert named == (100, 100)
    assert visits[0].moments == [(1, 100)]


def test_a_moment_named_before_the_older_one_was_written_replaces_it() -> None:
    """WIRE-178: the session holds one moment, and only the newest travels."""

    async def scenario() -> list[Visit]:
        async with scripted(_after_moments(2, 1)) as (port, visits):
            async with _session(port) as session:
                session.send(CHUNK)
                session.sustain_from_now()
                session.send(CHUNK)
                session.sustain_from_now()
                await _hear_all(session)
            return visits

    (visit,) = asyncio.run(scenario())

    assert visit.moments == [(2, 200)]


def test_without_a_moment_the_frames_are_the_audio_appends_alone() -> None:
    """The bytes a hearing writes without a moment are the ones it always wrote."""

    async def scenario() -> list[Visit]:
        async with scripted(_after_moments(2, 0)) as (port, visits):
            async with _session(port) as session:
                session.send(b"ab")
                session.send(b"c" * 3 + b"d")
                await _hear_all(session)
            return visits

    (visit,) = asyncio.run(scenario())

    assert visit.frames == [
        '{"type":"input_audio_buffer.append","audio":"YWI="}',
        '{"type":"input_audio_buffer.append","audio":"Y2NjZA=="}',
    ]


def test_a_session_that_did_not_ask_for_sustained_speech_names_no_moment() -> None:
    async def scenario() -> None:
        async with (
            scripted(says(then_close=1000)) as (port, _),
            session_to(port) as session,
        ):
            session.sustain_from_now()

    with pytest.raises(ValueError, match="sustained speech"):
        asyncio.run(scenario())


def test_no_moment_is_named_while_no_hearing_runs() -> None:
    async def scenario() -> tuple[int | None, int | None]:
        async with scripted(says(then_close=1000)) as (port, _):
            session = _session(port)
            before = session.sustain_from_now()
            async with session:
                await _hear_all(session)
            return before, session.sustain_from_now()

    assert asyncio.run(scenario()) == (None, None)


def test_only_the_event_of_the_newest_moment_is_delivered() -> None:
    """WIRE-178: the event of an older moment arrives late and is dropped."""
    first_written = asyncio.Event()

    async def newest_after_older(connection: ServerConnection, visit: Visit) -> None:
        await opens(connection, visit)
        await takes_audio(connection, visit, 1, 1)
        first_written.set()
        await takes_audio(connection, visit, 2, 2)
        await connection.send(sustained_from("a", 100, 400))
        await connection.send(sustained_from("a", 200, 500))
        await connection.close(1000)

    async def scenario() -> list[object]:
        async with (
            scripted(newest_after_older) as (port, _),
            _session(port) as session,
        ):
            session.send(CHUNK)
            session.sustain_from_now()
            async with asyncio.timeout(WHOLE_RUN_SECONDS):
                await first_written.wait()
            session.send(CHUNK)
            session.sustain_from_now()
            return await _hear_all(session)

    delivered = _moments_heard(asyncio.run(scenario()))

    assert [(event.from_ms, event.audio_end_ms) for event in delivered] == [(200, 500)]


def test_an_event_when_no_moment_was_written_is_malformed() -> None:
    async def scenario() -> object:
        async with scripted(says(sustained_from("a", 0, 300), then_close=1000)) as (
            port,
            _,
        ):
            async with _session(port) as session:
                await _hear_all(session)
            return session.outcome

    assert asyncio.run(scenario()) == Malformed()


def test_a_moment_written_before_a_cut_is_written_on_the_next_wire_from_its_start() -> (
    None
):
    """WIRE-176: a moment at the cut opens the next wire, its event on the timeline."""
    first = _after_moments(3, 1, closes_with=DRAIN_CLOSE_CODE)
    second = _after_moments(0, 1, sustained_from("a", 0, 340))

    async def scenario() -> tuple[list[object], list[Visit]]:
        tokens = Tokens(2)
        async with (
            scripted(first, second) as (port, visits),
            _session(port, tokens) as session,
        ):
            for _ in range(3):
                session.send(CHUNK)
            session.sustain_from_now()
            await tokens.reached(2)
            tokens.release(2)
            return await _hear_all(session), visits

    events, visits = asyncio.run(scenario())

    assert visits[1].moments == [(0, 0)]
    (heard,) = _moments_heard(events)
    assert (heard.from_ms, heard.audio_end_ms) == (300, 640)


def test_a_moment_below_the_next_wires_start_is_not_written_again() -> None:
    """WIRE-176: every item of the new session begins after such a moment."""
    first = _after_moments(3, 1, closes_with=DRAIN_CLOSE_CODE)

    async def scenario() -> list[Visit]:
        tokens = Tokens(2)
        async with scripted(first, says(then_close=1000)) as (port, visits):
            async with _session(port, tokens) as session:
                session.send(CHUNK)
                session.sustain_from_now()
                session.send(CHUNK)
                session.send(CHUNK)
                await tokens.reached(2)
                tokens.release(2)
                await _hear_all(session)
            return visits

    first_visit, second_visit = asyncio.run(scenario())

    assert first_visit.moments == [(1, 100)]
    assert second_visit.moments == []


def test_a_moment_named_during_the_pause_is_written_on_the_next_wire() -> None:
    """WIRE-176: queued audio and the moment after it go out in order on a new wire."""
    first = _after_moments(1, 0, closes_with=DRAIN_CLOSE_CODE)
    second = _after_moments(2, 1)

    async def scenario() -> tuple[int | None, list[Visit]]:
        tokens = Tokens(2)
        async with scripted(first, second) as (port, visits):
            async with _session(port, tokens) as session:
                session.send(CHUNK)
                await tokens.reached(2)
                session.send(CHUNK)
                session.send(CHUNK)
                named = session.sustain_from_now()
                tokens.release(2)
                await _hear_all(session)
            return named, visits

    named, visits = asyncio.run(scenario())

    assert named == QUEUED_BY_THEN_MS
    assert visits[1].moments == [(2, 200)]
