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
from websockets.frames import CloseCode

from packages.client.tests.support.elapsed_support import (
    CLOSE_TIMEOUT_SECONDS,
    SESSION_DEADLINE_SECONDS,
    well_before,
)
from packages.client.tests.support.hearing_support import (
    CREATED_EVENT,
    LOOPBACK,
    OPENING_MARGIN,
    fixed_tokens,
    leaves,
    outcome_at,
    passes_the_limit,
    session_to,
    silent_peer,
    stalled_service,
    takes_every_event,
)
from packages.client.tests.support.reconnect_support import WHOLE_RUN_SECONDS
from packages.client.tests.support.scripted_hearing_support import Visit, scripted
from speech_client.hearing import Closed, SessionTokens
from speech_client.outcomes import OverCap, Unreachable

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

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


@pytest.mark.parametrize(
    ("end", "outcome"),
    [
        pytest.param(passes_the_limit, OverCap(), id="over-limit"),
        pytest.param(leaves, Closed(), id="leave"),
    ],
)
def test_an_end_before_the_connection_opens_ends_the_session_at_once(
    end: Callable[[HearingSession], Awaitable[None]], outcome: HearingOutcome
) -> None:
    async def scenario() -> tuple[HearingOutcome | None, float]:
        async with await asyncio.start_server(silent_peer, LOOPBACK, 0) as server:
            session = session_to(server.sockets[0].getsockname()[1], 30)
            started = time.monotonic()
            async with session:
                await end(session)
            return session.outcome, time.monotonic() - started

    ended, elapsed = asyncio.run(scenario())

    assert ended == outcome
    assert elapsed < well_before(CLOSE_TIMEOUT_SECONDS)


async def no_token() -> None:
    return None


@pytest.mark.parametrize(
    ("late_answer", "connects"),
    [
        pytest.param(fixed_tokens, True, id="a-connection-is-aborted"),
        pytest.param(no_token, False, id="no-token-is-left-alone"),
    ],
)
def test_an_opening_that_settles_after_the_session_ended_leaves_its_outcome(
    late_answer: Callable[[], Awaitable[SessionTokens | None]], *, connects: bool
) -> None:
    async def scenario() -> tuple[HearingOutcome | None, list[int | None]]:
        asked = asyncio.Event()

        async def answers_only_once_cancelled() -> SessionTokens | None:
            asked.set()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.Event().wait()
            return await late_answer()

        closed_with: list[int | None] = []
        closed = asyncio.Event()

        async def waits_for_the_client_to_go(
            connection: ServerConnection, _: Visit
        ) -> None:
            await connection.wait_closed()
            closed_with.append(connection.close_code)
            closed.set()

        async with scripted(waits_for_the_client_to_go) as (port, _):
            session = session_to(port, 30, 1, 5, answers_only_once_cancelled)
            async with session:
                await asked.wait()
                session.send(b"over the held limit")
            if connects:
                async with asyncio.timeout(WHOLE_RUN_SECONDS):
                    await closed.wait()
        return session.outcome, closed_with

    outcome, close_codes = asyncio.run(scenario())

    assert outcome == OverCap()
    assert close_codes == ([CloseCode.ABNORMAL_CLOSURE] if connects else [])


def test_a_caller_timeout_during_a_stalled_opening_leaves_no_task_behind() -> None:
    async def scenario() -> float:
        peers: set[asyncio.Task[object] | None] = set()

        async def silent(reader: asyncio.StreamReader, writer: object) -> None:
            peers.add(asyncio.current_task())
            await silent_peer(reader, writer)

        async with await asyncio.start_server(silent, LOOPBACK, 0) as server:
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                async with (
                    asyncio.timeout(0.2),
                    session_to(server.sockets[0].getsockname()[1], 30) as session,
                ):
                    await takes_every_event(session)
            elapsed = time.monotonic() - started
            assert asyncio.all_tasks() - peers == {asyncio.current_task()}
        return elapsed

    assert asyncio.run(scenario()) < well_before(CLOSE_TIMEOUT_SECONDS)
