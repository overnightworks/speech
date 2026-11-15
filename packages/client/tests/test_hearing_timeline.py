"""A reconnect holds audio under the held limit and keeps one audio timeline."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest

from packages.client.tests.support.hearing_support import (
    AT_ONCE_SECONDS,
    HEARD_DELTA,
    session_to,
)
from packages.client.tests.support.reconnect_support import (
    CHUNK,
    CHUNK_MS,
    UNAVAILABLE,
    WHOLE_RUN_SECONDS,
    Tokens,
    collected,
    hear,
    reads,
)
from packages.client.tests.support.scripted_hearing_support import (
    Visit,
    completed,
    drains,
    ends_in_order,
    opens,
    says,
    scripted,
    speech_started,
    speech_stopped,
    takes_audio,
)
from speech_client import hearing
from speech_client.hearing import (
    NEEDED_HOLD_BYTES,
    OPENING_SECONDS,
    RECONNECT_BOUND,
    Closed,
    ReconnectBound,
    ReconnectCause,
    Reconnected,
)
from speech_client.outcomes import (
    OverCap,
)
from speech_client.wire import (
    DRAIN_CLOSE_CODE,
    PcmFormat,
    ServerEvent,
    SessionInput,
    SpeechStarted,
)

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection

    from packages.client.tests.support.scripted_hearing_support import Script
    from speech_client.hearing import HearingOutcome, HearingSession

pytestmark = pytest.mark.usefixtures("quick_bound")


def test_one_byte_over_the_held_limit_ends_the_hearing_while_the_source_runs() -> None:
    async def scenario() -> tuple[HearingOutcome | None, float]:
        tokens = Tokens(2)
        async with scripted(drains, ends_in_order) as (port, _):
            session = session_to(port, 30, 3, 5, tokens)
            async with session:
                await tokens.reached(2)
                started = time.monotonic()
                session.send(b"ab")
                session.send(b"cd")
                async with asyncio.timeout(WHOLE_RUN_SECONDS):
                    async for _ in session.events():
                        pass
            return session.outcome, time.monotonic() - started

    outcome, elapsed = asyncio.run(scenario())

    assert outcome == OverCap()
    assert elapsed < AT_ONCE_SECONDS


def test_one_byte_over_the_held_limit_ends_the_hearing_during_a_pause() -> None:
    long_pause = ReconnectBound(attempts=3, pause_seconds=10, within_seconds=30)

    async def scenario() -> tuple[HearingOutcome | None, float]:
        async with scripted(drains, UNAVAILABLE) as (port, visits):
            session = session_to(port, 30, 1, 5)
            async with session:
                async with asyncio.timeout(WHOLE_RUN_SECONDS):
                    while len(visits) < 2:  # noqa: ASYNC110, PLR2004 - a visit has no signal
                        await asyncio.sleep(0.01)
                started = time.monotonic()
                session.send(b"ab")
                async with asyncio.timeout(WHOLE_RUN_SECONDS):
                    async for _ in session.events():
                        pass
            return session.outcome, time.monotonic() - started

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hearing, "RECONNECT_BOUND", long_pause)
        outcome, elapsed = asyncio.run(scenario())

    assert outcome == OverCap()
    assert elapsed < AT_ONCE_SECONDS


def test_the_needed_hold_covers_the_opening_bound_and_the_reconnect_bound() -> None:
    audio_rate = SessionInput().format.rate * PcmFormat.sample_bytes
    assert OPENING_SECONDS * audio_rate <= NEEDED_HOLD_BYTES
    assert RECONNECT_BOUND.within_seconds * audio_rate <= NEEDED_HOLD_BYTES


@pytest.mark.parametrize(
    ("extra_bytes", "outcome", "reaching_the_new_session"),
    [
        pytest.param(0, Closed(), NEEDED_HOLD_BYTES, id="at-the-limit"),
        pytest.param(1, OverCap(), 0, id="one-byte-over"),
    ],
)
def test_the_needed_hold_is_held_through_a_reconnect_and_one_byte_more_is_not(
    extra_bytes: int, outcome: HearingOutcome, reaching_the_new_session: int
) -> None:
    async def scenario() -> tuple[HearingOutcome | None, list[Visit]]:
        tokens = Tokens(2)
        async with scripted(drains, reads(2)) as (port, visits):
            session = session_to(port, 30, NEEDED_HOLD_BYTES, 5, tokens)
            async with session:
                await tokens.reached(2)
                session.send(bytes(NEEDED_HOLD_BYTES - 1))
                session.send(bytes(1 + extra_bytes))
                tokens.release(2)
                async with asyncio.timeout(WHOLE_RUN_SECONDS):
                    async for _ in session.events():
                        pass
            return session.outcome, visits

    ended, visits = asyncio.run(scenario())

    assert ended == outcome
    assert sum(len(chunk) for visit in visits[1:] for chunk in visit.audio) == (
        reaching_the_new_session
    )


def test_audio_held_while_reconnecting_reaches_the_new_session_in_order_once() -> None:
    async def scenario() -> list[list[bytes]]:
        tokens = Tokens(2)
        async with scripted(reads(2, closes_with=DRAIN_CLOSE_CODE), reads(2)) as (
            port,
            visits,
        ):
            session = session_to(port, 30, 10, 5, tokens)
            async with session:
                session.send(b"a")
                session.send(b"b")
                await tokens.reached(2)
                session.send(b"c")
                session.send(b"d")
                tokens.release(2)
                async with asyncio.timeout(WHOLE_RUN_SECONDS):
                    async for _ in session.events():
                        pass
            return [visit.audio for visit in visits]

    assert asyncio.run(scenario()) == [[b"a", b"b"], [b"c", b"d"]]


async def _positions_of(
    first: Script, second: Script, chunk_sizes: tuple[int, ...]
) -> list[int]:
    tokens = Tokens(2)
    async with scripted(first, second) as (port, _):
        session = session_to(port, 30, sum(chunk_sizes) + 1, 5, tokens)
        async with session:
            for size in chunk_sizes:
                session.send(bytes(size))
            await tokens.reached(2)
            tokens.release(2)
            async with asyncio.timeout(WHOLE_RUN_SECONDS):
                return [
                    event.audio_start_ms
                    async for event in session.events()
                    if isinstance(event, SpeechStarted)
                ]


@pytest.mark.parametrize(
    ("chunk_sizes", "shift_ms"),
    [
        pytest.param((len(CHUNK),) * 3, 300, id="whole-chunks"),
        pytest.param((len(CHUNK), 100), 102, id="rounded-down"),
        pytest.param((47,), 0, id="under-a-millisecond"),
    ],
)
def test_positions_count_every_byte_whose_write_began_even_one_the_peer_never_read(
    chunk_sizes: tuple[int, ...], shift_ms: int
) -> None:
    # The old peer reads all chunks but the last, so the last was lost in flight.
    first = reads(len(chunk_sizes) - 1, closes_with=DRAIN_CLOSE_CODE)
    second = says(speech_started("item", 40), then_close=1000)

    positions = asyncio.run(_positions_of(first, second, chunk_sizes))

    assert positions == [40 + shift_ms]


def test_positions_stay_on_one_timeline_across_two_reconnects() -> None:
    async def scenario() -> list[object]:
        tokens = Tokens(2, 3)
        first = reads(3, closes_with=DRAIN_CLOSE_CODE)
        second = reads(2, closes_with=DRAIN_CLOSE_CODE, says=(speech_started("x", 50),))
        third = says(speech_started("y", 10), then_close=1000)
        async with scripted(first, second, third) as (port, _):
            session = session_to(port, 30, 100_000, 5, tokens)
            async with session:
                reading = asyncio.create_task(collected(session))
                for _ in range(3):
                    session.send(CHUNK)
                await tokens.reached(2)
                for _ in range(2):
                    session.send(CHUNK)
                tokens.release(2)
                await tokens.reached(3)
                tokens.release(3)
                async with asyncio.timeout(WHOLE_RUN_SECONDS):
                    return [
                        event
                        for event in await reading
                        if isinstance(event, SpeechStarted | Reconnected)
                    ]

    events = asyncio.run(scenario())

    assert [
        (event.audio_start_ms if isinstance(event, SpeechStarted) else event)
        for event in events
    ] == [
        Reconnected(ReconnectCause.DRAIN, 0, 300),
        350,
        Reconnected(ReconnectCause.DRAIN, 300, 500),
        510,
    ]


@pytest.mark.parametrize(
    ("turn_events", "unheard_start_ms"),
    [
        pytest.param(
            (
                speech_started("a", 100),
                speech_stopped("a", 900),
                completed("a"),
                speech_started("b", 1000),
            ),
            900,
            id="after-the-last-delivered-turn",
        ),
        pytest.param(
            (speech_started("a", 100), speech_stopped("a", 900)),
            0,
            id="no-turn-delivered",
        ),
    ],
)
def test_reconnected_spans_the_audio_after_the_last_delivered_turn(
    turn_events: tuple[str, ...], unheard_start_ms: int
) -> None:
    chunks = 15

    async def hears_and_drops(connection: ServerConnection, visit: Visit) -> None:
        await opens(connection, visit)
        await takes_audio(connection, visit, chunks)
        for event in turn_events:
            await connection.send(event)
        connection.transport.abort()

    async def scenario() -> list[ServerEvent | Reconnected]:
        async with scripted(hears_and_drops, ends_in_order) as (port, _):

            def send(session: HearingSession) -> None:
                for _ in range(chunks):
                    session.send(CHUNK)

            return (
                await hear(port, held_limit_bytes=chunks * len(CHUNK), before=send)
            )[0]

    events = asyncio.run(scenario())

    reconnected = [event for event in events if isinstance(event, Reconnected)]
    assert reconnected == [
        Reconnected(ReconnectCause.CUT, unheard_start_ms, chunks * CHUNK_MS)
    ]


def test_reconnected_comes_before_the_new_sessions_events() -> None:
    async def scenario() -> list[ServerEvent | Reconnected]:
        async with scripted(drains, says(HEARD_DELTA, then_close=1000)) as (port, _):
            return (await hear(port))[0]

    kinds = [type(event).__name__ for event in asyncio.run(scenario())]

    assert kinds == [
        "SessionCreated",
        "Reconnected",
        "SessionCreated",
        "TranscriptDelta",
    ]


def test_a_failed_reconnect_yields_no_reconnected() -> None:
    async def scenario() -> list[ServerEvent | Reconnected]:
        async with scripted(drains, UNAVAILABLE) as (port, _):
            return (await hear(port))[0]

    assert not any(isinstance(event, Reconnected) for event in asyncio.run(scenario()))
