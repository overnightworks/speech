"""The hearing session takes its tokens from a source before every opening."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest
from logline import current_traceparent, unit_trace

from packages.client.tests.support.elapsed_support import well_before
from packages.client.tests.support.hearing_support import (
    PERSON_TOKEN,
    WORKLOAD_TOKEN,
    fixed_tokens,
    session_to,
)
from packages.client.tests.support.reconnect_support import (
    BOUND,
    Tokens,
    hear,
    hear_through,
    outcome_of,
)
from packages.client.tests.support.scripted_hearing_support import (
    answers_then_drains,
    drains,
    ends_in_order,
    scripted,
)
from speech_client.hearing import (
    Closed,
    NoToken,
    ReconnectFailed,
    SessionTokens,
)
from speech_client.outcomes import (
    PastDeadline,
    Unreachable,
)
from speech_client.wire import (
    AUTHORIZATION_HEADER,
    TRACEPARENT_HEADER,
    WORKLOAD_TOKEN_HEADER,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from packages.client.tests.support.scripted_hearing_support import Script
    from speech_client.hearing import HearingOutcome

pytestmark = pytest.mark.usefixtures("quick_bound")

SESSION_DEADLINE_SECONDS = 30


def test_the_source_is_awaited_before_every_opening_and_its_tokens_travel() -> None:
    async def scenario() -> tuple[int, list[tuple[str | None, str | None]]]:
        tokens = Tokens()
        async with scripted(drains, ends_in_order) as (port, visits):
            await hear(port, token_source=tokens)
            return tokens.calls, [
                (
                    visit.headers.get(WORKLOAD_TOKEN_HEADER),
                    visit.headers.get(AUTHORIZATION_HEADER),
                )
                for visit in visits
            ]

    calls, sent = asyncio.run(scenario())

    assert calls == len(sent) == 2  # noqa: PLR2004 - the opening and the reopening
    assert sent == [
        ("workload-1", "Bearer person-1"),
        ("workload-2", "Bearer person-2"),
    ]


@pytest.mark.parametrize("trace", ["caller's", "ambient", "none"])
def test_every_opening_sends_the_callers_traceparent_over_the_ambient_one_or_none(
    trace: str,
) -> None:
    """WIRE-149: every upgrade carries the same traceparent, reopenings included.

    The caller's is given inside a running trace, which it must win over.
    """
    given = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

    async def scenario() -> tuple[list[str | None], str | None]:
        async with scripted(drains, ends_in_order) as (port, visits):
            if trace == "none":
                await hear(port)
                ambient = None
            else:
                with unit_trace(None):
                    ambient = current_traceparent()
                    await hear(port, traceparent=given if trace == "caller's" else None)
            return [visit.headers.get(TRACEPARENT_HEADER) for visit in visits], ambient

    sent, ambient = asyncio.run(scenario())

    expected = given if trace == "caller's" else ambient
    assert sent == [expected, expected]


def test_a_token_repr_carries_no_token() -> None:
    shown = repr(SessionTokens(WORKLOAD_TOKEN, PERSON_TOKEN))

    assert WORKLOAD_TOKEN not in shown
    assert PERSON_TOKEN not in shown


@pytest.mark.parametrize("scripts", [(ends_in_order,), (drains, ends_in_order)])
def test_a_source_without_tokens_ends_the_hearing_with_no_token(
    scripts: tuple[Script, ...],
) -> None:
    async def scenario() -> tuple[HearingOutcome | None, int]:
        calls = 0

        async def lapses_at_the_second_call() -> SessionTokens | None:
            nonlocal calls
            calls += 1
            if calls == len(scripts):
                return None
            return await fixed_tokens()

        async with scripted(*scripts) as (port, visits):
            outcome = (await hear(port, token_source=lapses_at_the_second_call))[1]
            return outcome, len(visits)

    assert asyncio.run(scenario()) == (NoToken(), len(scripts) - 1)


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("no"), TimeoutError("slow"), OSError("down")],
    ids=["runtime", "timeout", "os"],
)
@pytest.mark.parametrize("reopening", [False, True], ids=["first", "reopened"])
def test_an_exception_of_the_source_is_raised_from_leaving_and_opens_nothing_more(
    failure: Exception, *, reopening: bool
) -> None:
    async def scenario() -> tuple[HearingOutcome | None, int]:
        calls = 0

        async def raises() -> SessionTokens:
            nonlocal calls
            calls += 1
            if calls == (2 if reopening else 1):
                raise failure
            return await fixed_tokens()

        async with scripted(drains, ends_in_order) as (port, visits):
            session = session_to(port, 10, token_source=raises)
            with pytest.raises(type(failure)) as raised:
                await hear_through(session)
            assert raised.value is failure
            return session.outcome, len(visits)

    assert asyncio.run(scenario()) == (None, 1 if reopening else 0)


def test_a_cancel_the_source_raises_propagates_with_the_closed_outcome() -> None:
    async def scenario() -> HearingOutcome | None:
        async def cancels() -> SessionTokens:
            raise asyncio.CancelledError

        async with scripted(ends_in_order) as (port, _):
            session = session_to(port, 10, token_source=cancels)
            with pytest.raises(asyncio.CancelledError):
                await hear_through(session)
            return session.outcome

    assert asyncio.run(scenario()) == Closed()


async def _stalls() -> SessionTokens:
    await asyncio.Event().wait()
    raise AssertionError


async def _fails_when_cancelled() -> SessionTokens:
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        message = "the source's own failure"
        raise OSError(message) from None
    raise AssertionError


@pytest.mark.parametrize(
    "source", [_stalls, _fails_when_cancelled], ids=["stalls", "fails-when-cancelled"]
)
def test_a_bound_that_expires_while_the_source_runs_ends_as_that_bound_says(
    source: Callable[[], Awaitable[SessionTokens]],
) -> None:
    started = time.monotonic()

    outcome = outcome_of(
        ends_in_order,
        token_source=source,
        deadline_seconds=SESSION_DEADLINE_SECONDS,
        opening_seconds=0.2,
    )

    assert outcome == Unreachable()
    assert time.monotonic() - started < well_before(SESSION_DEADLINE_SECONDS)


def test_a_deadline_that_expires_while_the_source_runs_is_past_deadline() -> None:
    outcome = outcome_of(
        ends_in_order, token_source=_stalls, deadline_seconds=0.3, opening_seconds=5
    )

    assert outcome == PastDeadline()


def test_a_source_that_stalls_at_the_reopening_counts_every_attempt_as_failed() -> None:
    async def scenario() -> tuple[HearingOutcome | None, int]:
        calls = 0

        async def stalls_from_the_second_call() -> SessionTokens:
            nonlocal calls
            calls += 1
            if calls > 1:
                await asyncio.Event().wait()
            return await fixed_tokens()

        async with scripted(answers_then_drains) as (port, _):
            outcome = (
                await hear(
                    port, token_source=stalls_from_the_second_call, opening_seconds=0.1
                )
            )[1]
            return outcome, calls

    assert asyncio.run(scenario()) == (ReconnectFailed(), 1 + BOUND.attempts)
