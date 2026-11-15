"""The hearing session reopens after a drain or a cut and bounds the attempts."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

import pytest
from websockets.asyncio.server import serve

from packages.client.tests.support.hearing_support import (
    AT_ONCE_SECONDS,
    CREATED_EVENT,
    HEARD_DELTA,
    LOOPBACK,
    session_to,
)
from packages.client.tests.support.reconnect_support import (
    BOUND,
    UNAVAILABLE,
    Tokens,
    hear,
    outcome_of,
    skip_ahead,
    visits_and_outcome,
)
from packages.client.tests.support.scripted_hearing_support import (
    Refusal,
    Visit,
    drains,
    drains_before_opening,
    drops_after_opening,
    ends_in_order,
    errored,
    failed,
    says,
    scripted,
    speech_started,
    takes_audio,
)
from speech_client import hearing
from speech_client.hearing import (
    Closed,
    ReconnectBound,
    ReconnectCause,
    Reconnected,
    ReconnectFailed,
    Refused,
    accept_first_event,
)
from speech_client.outcomes import (
    Cut,
    Malformed,
    PastDeadline,
)
from speech_client.wire import (
    DRAIN_CLOSE_CODE,
    RefusalCode,
    ServerEvent,
    SessionCreated,
    parse_server_event,
)

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection

    from packages.client.tests.support.scripted_hearing_support import Script
    from speech_client.hearing import HearingOutcome, HearingSession
pytestmark = pytest.mark.usefixtures("quick_bound")


def test_a_drain_reopens_at_the_same_origin_with_the_same_update() -> None:
    async def scenario() -> tuple[list[ServerEvent | Reconnected], list[Visit]]:
        async with scripted(drains, ends_in_order) as (port, visits):
            events, outcome = await hear(port)
            assert outcome == Closed()
            return events, visits

    events, visits = asyncio.run(scenario())

    assert [type(event) for event in events] == [
        SessionCreated,
        Reconnected,
        SessionCreated,
    ]
    assert len(visits) == 2  # noqa: PLR2004 - the drained session and its successor
    assert visits[0].update is not None
    assert visits[1].update == visits[0].update


@pytest.mark.parametrize(
    ("script", "cause"),
    [
        pytest.param(drains, ReconnectCause.DRAIN, id="drain"),
        pytest.param(drops_after_opening, ReconnectCause.CUT, id="cut-after-opening"),
    ],
)
def test_a_session_that_ends_without_error_after_it_opened_is_reopened(
    script: Script, cause: ReconnectCause
) -> None:
    async def scenario() -> list[ServerEvent | Reconnected]:
        async with scripted(script, ends_in_order) as (port, _):
            return (await hear(port))[0]

    events = asyncio.run(scenario())

    assert Reconnected(cause, 0, 0) in events


@pytest.mark.parametrize(
    ("script", "outcome"),
    [
        pytest.param(
            says(errored(), then_close=DRAIN_CLOSE_CODE), Cut(), id="error-then-drain"
        ),
        pytest.param(
            says(failed("i"), then_close=DRAIN_CLOSE_CODE),
            Cut(),
            id="failed-then-drain",
        ),
        pytest.param(says(errored(), then_close=1011), Cut(), id="error-then-1011"),
        pytest.param(ends_in_order, Closed(), id="close-1000"),
        pytest.param(says(then_close=1001), Closed(), id="close-1001"),
        pytest.param(drains_before_opening, Cut(), id="drain-before-the-first-opening"),
    ],
)
def test_these_ends_do_not_reopen_the_hearing(
    script: Script, outcome: HearingOutcome
) -> None:
    visits, ended = visits_and_outcome(script, ends_in_order)

    assert (visits, ended) == (1, outcome)


@pytest.mark.parametrize(
    "unopened",
    [
        pytest.param(drains_before_opening, id="drain-before-created"),
        pytest.param(UNAVAILABLE, id="unavailable"),
        pytest.param(Refusal(RefusalCode.BUSY, retry_after="30"), id="busy"),
    ],
)
def test_a_failed_attempt_is_retried_until_the_hearing_goes_on(
    unopened: Script,
) -> None:
    visits, ended = visits_and_outcome(drains, unopened, unopened, ends_in_order)

    assert (visits, ended) == (4, Closed())


async def stays_silent(connection: ServerConnection, visit: Visit) -> None:
    del visit
    await connection.wait_closed()


def test_an_attempt_whose_opening_bound_expires_counts_as_failed() -> None:
    visits, ended = visits_and_outcome(
        drains, stays_silent, ends_in_order, opening_seconds=0.2
    )

    assert (visits, ended) == (3, Closed())


@pytest.mark.parametrize(
    "failing",
    [
        pytest.param(drains_before_opening, id="cut-before-created"),
        pytest.param(UNAVAILABLE, id="unavailable"),
        pytest.param(Refusal(RefusalCode.BUSY, retry_after="30"), id="busy"),
    ],
)
def test_attempts_that_all_fail_end_the_hearing_with_reconnect_failed(
    failing: Script,
) -> None:
    started = time.monotonic()
    visits, ended = visits_and_outcome(drains, failing)

    assert (visits, ended) == (1 + BOUND.attempts, ReconnectFailed())
    # A `busy` is not waited out: its Retry-After is 30 s.
    assert time.monotonic() - started < AT_ONCE_SECONDS * 3


def test_an_unreachable_service_counts_every_attempt_and_ends_reconnect_failed() -> (
    None
):
    async def scenario() -> HearingOutcome | None:
        async def drains_and_goes(connection: ServerConnection, visit: Visit) -> None:
            await drains(connection, visit)
            server.close()

        async def handler(connection: ServerConnection) -> None:
            await drains_and_goes(connection, Visit(connection.request.headers))

        async with serve(handler, LOOPBACK, 0) as server:
            port = server.sockets[0].getsockname()[1]
            return (await hear(port))[1]

    assert asyncio.run(scenario()) == ReconnectFailed()


def test_a_session_created_in_hand_after_the_bound_expired_fails_the_attempt() -> None:
    created = parse_server_event(CREATED_EVENT)

    with pytest.raises(TimeoutError):
        accept_first_event(created, bound_expired=True)


def test_a_deadline_falling_due_with_the_last_opening_bound_ends_reconnect_failed() -> (
    None
):
    deadline_seconds = 1.5

    async def stalls_past_both(connection: ServerConnection, visit: Visit) -> None:
        visit.update = str(await connection.recv())
        # Both bounds are due in the pass that follows, before the attempt ends.
        skip_ahead(deadline_seconds + 0.1)
        await connection.wait_closed()

    silent_attempts = (stays_silent,) * (BOUND.attempts - 1)

    visits, outcome = visits_and_outcome(
        drains,
        *silent_attempts,
        stalls_past_both,
        opening_seconds=0.2,
        deadline_seconds=deadline_seconds,
    )

    assert (visits, outcome) == (1 + BOUND.attempts, ReconnectFailed())


def test_the_last_failed_attempt_ends_the_hearing_without_a_pause() -> None:
    one_attempt = ReconnectBound(attempts=1, pause_seconds=5, within_seconds=30)

    async def scenario() -> tuple[HearingOutcome | None, float]:
        async with scripted(drains, UNAVAILABLE) as (port, _):
            started = time.monotonic()
            outcome = (await hear(port, deadline_seconds=1))[1]
            return outcome, time.monotonic() - started

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hearing, "RECONNECT_BOUND", one_attempt)
        outcome, elapsed = asyncio.run(scenario())

    assert outcome == ReconnectFailed()
    assert elapsed < AT_ONCE_SECONDS


@pytest.mark.parametrize(
    ("refusal", "outcome"),
    [
        pytest.param(
            Refusal(RefusalCode.UNAUTHORIZED),
            Refused(401, RefusalCode.UNAUTHORIZED),
            id="unauthorized",
        ),
        pytest.param(Refusal(None), Refused(502, None), id="off-contract"),
        pytest.param(
            Refusal(RefusalCode.BUSY), Malformed(), id="busy-without-retry-after"
        ),
    ],
)
def test_any_other_refusal_of_an_attempt_ends_the_hearing_with_it(
    refusal: Refusal, outcome: HearingOutcome
) -> None:
    visits, ended = visits_and_outcome(drains, refusal, ends_in_order)

    assert (visits, ended) == (2, outcome)


@pytest.mark.parametrize(
    "first_event",
    [HEARD_DELTA, speech_started("i", 0)],
    ids=["a-delta", "speech-started"],
)
@pytest.mark.parametrize("reopening", [False, True], ids=["first", "reopened"])
def test_a_first_event_that_is_not_session_created_is_malformed(
    first_event: str, *, reopening: bool
) -> None:
    async def says_it_first(connection: ServerConnection, visit: Visit) -> None:
        visit.update = str(await connection.recv())
        await connection.send(first_event)
        await connection.wait_closed()

    scripts = (drains, says_it_first) if reopening else (says_it_first,)

    assert outcome_of(*scripts) == Malformed()


def test_no_audio_reaches_a_session_before_its_session_created() -> None:
    async def scenario() -> tuple[bool, list[bytes]]:
        async def opens_late(connection: ServerConnection, visit: Visit) -> None:
            visit.update = str(await connection.recv())
            try:
                async with asyncio.timeout(0.3):
                    await connection.recv()
            except TimeoutError:
                quiet.append(True)
            await connection.send(CREATED_EVENT)
            await takes_audio(connection, visit, 2)
            await connection.close()

        quiet: list[bool] = []
        async with scripted(opens_late) as (port, visits):

            def send(session: HearingSession) -> None:
                session.send(b"a")
                session.send(b"b")

            await hear(port, held_limit_bytes=10, before=send)
            return bool(quiet), visits[0].audio

    assert asyncio.run(scenario()) == (True, [b"a", b"b"])


def test_a_successful_opening_resets_the_bound() -> None:
    visits, ended = visits_and_outcome(
        drains,
        UNAVAILABLE,
        UNAVAILABLE,
        drains,
        UNAVAILABLE,
        UNAVAILABLE,
        ends_in_order,
    )

    assert (visits, ended) == (7, Closed())


def test_the_reconnect_bound_ends_the_hearing_within_its_time() -> None:
    long_bound = ReconnectBound(attempts=100, pause_seconds=0.1, within_seconds=0.45)

    async def scenario() -> tuple[HearingOutcome | None, float]:
        async def stays_silent(connection: ServerConnection, visit: Visit) -> None:
            del visit
            await connection.wait_closed()

        async with scripted(drains, stays_silent) as (port, _):
            started = time.monotonic()
            outcome = (await hear(port, opening_seconds=30))[1]
            return outcome, time.monotonic() - started

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hearing, "RECONNECT_BOUND", long_bound)
        outcome, elapsed = asyncio.run(scenario())

    assert outcome == ReconnectFailed()
    assert long_bound.within_seconds <= elapsed < long_bound.within_seconds + 1


def test_a_deadline_below_the_reconnect_bound_wins() -> None:
    slow_bound = ReconnectBound(attempts=5, pause_seconds=5, within_seconds=30)

    async def scenario() -> tuple[HearingOutcome | None, float]:
        async with scripted(drains, UNAVAILABLE) as (port, _):
            started = time.monotonic()
            outcome = (await hear(port, deadline_seconds=0.5))[1]
            return outcome, time.monotonic() - started

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hearing, "RECONNECT_BOUND", slow_bound)
        outcome, elapsed = asyncio.run(scenario())

    assert outcome == PastDeadline()
    assert elapsed < AT_ONCE_SECONDS * 2


def test_a_deadline_due_with_the_bound_during_a_pause_ends_reconnect_failed() -> None:
    pausing_bound = ReconnectBound(attempts=5, pause_seconds=30, within_seconds=1)
    deadline_seconds = 2

    async def fails_into_the_pause(connection: ServerConnection, visit: Visit) -> None:
        del visit
        await connection.wait_closed()
        # The client pauses now; the bound and the deadline are due in one pass.
        skip_ahead(deadline_seconds + pausing_bound.within_seconds)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hearing, "RECONNECT_BOUND", pausing_bound)
        outcome = outcome_of(
            drains,
            fails_into_the_pause,
            opening_seconds=0.2,
            deadline_seconds=deadline_seconds,
        )

    assert outcome == ReconnectFailed()


def test_leaving_during_a_reconnect_ends_the_session_closed_at_once() -> None:
    async def scenario() -> tuple[HearingOutcome | None, float]:
        tokens = Tokens(2)
        async with scripted(drains, ends_in_order) as (port, _):
            session = session_to(port, 30, 1, 5, tokens)
            async with session:
                await tokens.reached(2)
                started = time.monotonic()
            return session.outcome, time.monotonic() - started

    outcome, elapsed = asyncio.run(scenario())

    assert outcome == Closed()
    assert elapsed < AT_ONCE_SECONDS


def test_a_cancel_during_a_reconnect_leaves_no_task_behind() -> None:
    async def scenario() -> None:
        tokens = Tokens(2)
        async with scripted(drains) as (port, _):
            with contextlib.suppress(TimeoutError):
                async with (
                    asyncio.timeout(0.5),
                    session_to(port, 30, 1, 5, tokens) as session,
                ):
                    async for _ in session.events():
                        pass
            assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())
