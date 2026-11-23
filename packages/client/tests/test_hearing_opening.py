"""The hearing session's bound on the opening, against scripted peers."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from typing import TYPE_CHECKING

import pytest
from websockets.asyncio.server import serve

from packages.client.tests.support.elapsed_support import (
    CLOSE_TIMEOUT_SECONDS,
    SESSION_DEADLINE_SECONDS,
    well_before,
)
from packages.client.tests.support.hearing_support import (
    CREATED_EVENT,
    LOOPBACK,
    OPENING_MARGIN,
    outcome_at,
    session_to,
    silent_peer,
    stalled_service,
    takes_every_event,
)
from speech_client.hearing import Closed
from speech_client.outcomes import Unreachable

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection

    from speech_client.hearing import HearingOutcome, HearingSession

OPENING_SHORT = 0.3  # passed at once by a stalled peer, far below the deadline


@pytest.mark.parametrize(
    "stall",
    [
        pytest.param("handshake", id="handshake-never-answered"),
        pytest.param("first-event", id="no-event-after-the-handshake"),
    ],
)
def test_a_stalled_opening_ends_with_unreachable_at_the_bound(stall: str) -> None:
    async def scenario() -> tuple[HearingOutcome | None, float]:
        async with contextlib.AsyncExitStack() as stack:
            if stall == "handshake":
                server = await stack.enter_async_context(
                    await asyncio.start_server(silent_peer, LOOPBACK, 0)
                )
                port = server.sockets[0].getsockname()[1]
            else:
                port, _ = await stack.enter_async_context(stalled_service(()))
            started = time.monotonic()
            ended = await outcome_at(port, SESSION_DEADLINE_SECONDS, OPENING_SHORT)
            return ended, time.monotonic() - started

    ended, elapsed = asyncio.run(scenario())

    assert ended == Unreachable()
    assert elapsed < well_before(SESSION_DEADLINE_SECONDS)


def test_a_deadline_equal_to_the_opening_bound_leaves_the_opening_to_end_it() -> None:
    async def scenario() -> HearingOutcome | None:
        async with stalled_service(()) as (port, _):
            return await outcome_at(port, OPENING_SHORT, OPENING_SHORT)

    assert asyncio.run(scenario()) == Unreachable()


def test_a_service_that_opened_in_time_may_stay_silent_past_the_bound() -> None:
    async def scenario() -> HearingOutcome | None:
        async def says_created_then_waits(connection: ServerConnection) -> None:
            await connection.recv()
            await connection.send(CREATED_EVENT)
            await connection.wait_closed()

        async with serve(says_created_then_waits, LOOPBACK, 0) as server:
            port = server.sockets[0].getsockname()[1]
            session = session_to(port, 10, opening_seconds=OPENING_MARGIN)
            async with session:
                async with asyncio.timeout(OPENING_MARGIN):
                    await anext(session.events())
                # The one wait past the bound: it must not end the open session.
                await asyncio.sleep(OPENING_MARGIN + 0.3)
                assert session.outcome is None
            return session.outcome

    assert asyncio.run(scenario()) == Closed()


async def _takes_every_event_once_open(
    session: HearingSession, opened: asyncio.Event
) -> None:
    await opened.wait()
    await takes_every_event(session)


def test_a_caller_timeout_during_the_wait_for_the_first_event_leaves_no_task() -> None:
    async def scenario() -> float:
        async with stalled_service(()) as (port, opened):
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                async with (
                    asyncio.timeout(0.2),
                    session_to(port, 30, opening_seconds=10) as session,
                ):
                    await _takes_every_event_once_open(session, opened)
            elapsed = time.monotonic() - started
        assert asyncio.all_tasks() == {asyncio.current_task()}
        return elapsed

    assert asyncio.run(scenario()) < well_before(CLOSE_TIMEOUT_SECONDS)


def test_audio_held_for_a_stalled_opening_never_reaches_the_peer() -> None:
    async def scenario() -> tuple[HearingOutcome | None, list[bytes]]:
        received: list[bytes] = []
        closed = asyncio.Event()

        async def never_says_created(connection: ServerConnection) -> None:
            try:
                async for message in connection:
                    audio = json.loads(message).get("audio")
                    if audio is not None:
                        received.append(base64.b64decode(audio))
            finally:
                closed.set()

        async with serve(never_says_created, LOOPBACK, 0) as server:
            port = server.sockets[0].getsockname()[1]
            session = session_to(
                port, 30, held_limit_bytes=10, opening_seconds=OPENING_SHORT
            )
            async with session:
                session.send(b"a")
                session.send(b"b")
                async with asyncio.timeout(OPENING_MARGIN):
                    await takes_every_event(session)
                session.send(b"c")
            async with asyncio.timeout(5):
                await closed.wait()
            return session.outcome, received

    assert asyncio.run(scenario()) == (Unreachable(), [])


@pytest.mark.parametrize("opening_seconds", [0, -1])
def test_an_opening_bound_of_zero_or_less_raises(opening_seconds: float) -> None:
    with pytest.raises(ValueError, match="opening_seconds"):
        session_to(1, opening_seconds=opening_seconds)
