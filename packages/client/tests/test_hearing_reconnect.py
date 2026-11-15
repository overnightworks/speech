"""The hearing session reopens after a drain or a cut, against scripted peers."""

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
    PERSON_TOKEN,
    WORKLOAD_TOKEN,
    fixed_tokens,
    session_to,
)
from packages.client.tests.support.scripted_hearing_support import (
    Refusal,
    Visit,
    completed,
    drains,
    drains_before_opening,
    drops_after_opening,
    ends_in_order,
    errored,
    failed,
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
    NoToken,
    ReconnectBound,
    ReconnectCause,
    Reconnected,
    ReconnectFailed,
    Refused,
    SessionTokens,
)
from speech_client.outcomes import (
    Busy,
    BusyWait,
    Cut,
    Malformed,
    OverCap,
    PastDeadline,
    Unreachable,
)
from speech_client.wire import (
    AUTHORIZATION_HEADER,
    DRAIN_CLOSE_CODE,
    WORKLOAD_TOKEN_HEADER,
    PcmFormat,
    RefusalCode,
    ServerEvent,
    SessionCreated,
    SessionInput,
    SpeechStarted,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from websockets.asyncio.server import ServerConnection

    from packages.client.tests.support.scripted_hearing_support import Script
    from speech_client.hearing import HearingOutcome, HearingSession

BOUND = ReconnectBound(attempts=3, pause_seconds=0.05, within_seconds=3)
CHUNK_MS = 100
CHUNK = bytes(SessionInput().format.rate * PcmFormat.sample_bytes * CHUNK_MS // 1000)
WHOLE_RUN_SECONDS = 15
UNAVAILABLE = Refusal(RefusalCode.UNAVAILABLE)


@pytest.fixture(autouse=True)
def quick_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hearing, "RECONNECT_BOUND", BOUND)


class Tokens:
    """A token source that numbers its calls and holds the calls named in `gated`."""

    def __init__(self, *gated: int) -> None:
        """Hold the calls numbered `gated` until they are released."""
        self.calls = 0
        self._reached = {call: asyncio.Event() for call in gated}
        self._released = {call: asyncio.Event() for call in gated}

    async def __call__(self) -> SessionTokens:
        """The tokens of this call, numbered from 1."""
        self.calls += 1
        call = self.calls
        if call in self._reached:
            self._reached[call].set()
            await self._released[call].wait()
        return SessionTokens(f"workload-{call}", f"person-{call}")

    async def reached(self, call: int) -> None:
        """Wait until the held call `call` has been made."""
        async with asyncio.timeout(WHOLE_RUN_SECONDS):
            await self._reached[call].wait()

    def release(self, call: int) -> None:
        """Let the held call `call` return its tokens."""
        self._released[call].set()


async def _hear(  # noqa: PLR0913 - one keyword per thing a scenario varies
    port: int,
    *,
    token_source: hearing.TokenSource = fixed_tokens,
    deadline_seconds: float = WHOLE_RUN_SECONDS,
    opening_seconds: float = 1,
    held_limit_bytes: int = 1,
    before: Callable[[HearingSession], None] = lambda _: None,
) -> tuple[list[ServerEvent | Reconnected], HearingOutcome | None]:
    session = session_to(
        port,
        deadline_seconds,
        held_limit_bytes,
        opening_seconds,
        token_source,
    )
    async with session:
        before(session)
        async with asyncio.timeout(WHOLE_RUN_SECONDS):
            events = [event async for event in session.events()]
    return events, session.outcome


def _outcome_of(*scripts: Script, **inputs: object) -> HearingOutcome | None:
    async def scenario() -> HearingOutcome | None:
        async with scripted(*scripts) as (port, _):
            return (await _hear(port, **inputs))[1]

    return asyncio.run(scenario())


def _visits_and_outcome(
    *scripts: Script, **inputs: object
) -> tuple[int, HearingOutcome | None]:
    async def scenario() -> tuple[int, HearingOutcome | None]:
        async with scripted(*scripts) as (port, visits):
            outcome = (await _hear(port, **inputs))[1]
            return len(visits), outcome

    return asyncio.run(scenario())


async def _hear_through(session: HearingSession) -> None:
    async with session:
        async with asyncio.timeout(WHOLE_RUN_SECONDS):
            async for _ in session.events():
                pass


async def _collected(session: HearingSession) -> list[ServerEvent | Reconnected]:
    return [event async for event in session.events()]


def _reads(
    chunks: int, *, closes_with: int = 1000, says: tuple[str, ...] = ()
) -> Script:
    """Open, say `says`, take `chunks` chunks of audio, then close with a code."""

    async def play(connection: ServerConnection, visit: Visit) -> None:
        await opens(connection, visit)
        for event in says:
            await connection.send(event)
        await takes_audio(connection, visit, chunks)
        await connection.close(closes_with)

    return play


def test_a_drain_reopens_at_the_same_origin_with_the_same_update() -> None:
    async def scenario() -> tuple[list[ServerEvent | Reconnected], list[Visit]]:
        async with scripted(drains, ends_in_order) as (port, visits):
            events, outcome = await _hear(port)
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
            return (await _hear(port))[0]

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
    visits, ended = _visits_and_outcome(script, ends_in_order)

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
    visits, ended = _visits_and_outcome(drains, unopened, unopened, ends_in_order)

    assert (visits, ended) == (4, Closed())


def test_an_attempt_whose_opening_bound_expires_counts_as_failed() -> None:
    async def stays_silent(connection: ServerConnection, visit: Visit) -> None:
        del visit
        await connection.wait_closed()

    visits, ended = _visits_and_outcome(
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
    visits, ended = _visits_and_outcome(drains, failing)

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
            return (await _hear(port))[1]

    assert asyncio.run(scenario()) == ReconnectFailed()


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
    visits, ended = _visits_and_outcome(drains, refusal, ends_in_order)

    assert (visits, ended) == (2, outcome)


def test_a_busy_refusal_of_an_attempt_is_not_the_busy_outcome() -> None:
    visits, ended = _visits_and_outcome(
        drains, Refusal(RefusalCode.BUSY, retry_after="7")
    )

    assert ended != Busy(BusyWait(retry_after_seconds=7))
    assert visits > 2  # noqa: PLR2004 - retried at least once


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

    assert _outcome_of(*scripts) == Malformed()


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

            await _hear(port, held_limit_bytes=10, before=send)
            return bool(quiet), visits[0].audio

    assert asyncio.run(scenario()) == (True, [b"a", b"b"])


def test_a_successful_opening_resets_the_bound() -> None:
    visits, ended = _visits_and_outcome(
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
            outcome = (await _hear(port, opening_seconds=30))[1]
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
            outcome = (await _hear(port, deadline_seconds=0.5))[1]
            return outcome, time.monotonic() - started

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hearing, "RECONNECT_BOUND", slow_bound)
        outcome, elapsed = asyncio.run(scenario())

    assert outcome == PastDeadline()
    assert elapsed < AT_ONCE_SECONDS * 2


def test_the_source_is_awaited_before_every_opening_and_its_tokens_travel() -> None:
    async def scenario() -> tuple[int, list[tuple[str | None, str | None]]]:
        tokens = Tokens()
        async with scripted(drains, ends_in_order) as (port, visits):
            await _hear(port, token_source=tokens)
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
            outcome = (await _hear(port, token_source=lapses_at_the_second_call))[1]
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
                await _hear_through(session)
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
                await _hear_through(session)
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

    outcome = _outcome_of(ends_in_order, token_source=source, opening_seconds=0.2)

    assert outcome == Unreachable()
    assert time.monotonic() - started < AT_ONCE_SECONDS


def test_a_deadline_that_expires_while_the_source_runs_is_past_deadline() -> None:
    outcome = _outcome_of(
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

        async with scripted(drains) as (port, _):
            outcome = (
                await _hear(
                    port, token_source=stalls_from_the_second_call, opening_seconds=0.1
                )
            )[1]
            return outcome, calls

    assert asyncio.run(scenario()) == (ReconnectFailed(), 1 + BOUND.attempts)


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


def test_the_needed_hold_is_held_through_a_reconnect_and_one_byte_more_is_not() -> None:
    async def scenario(extra: int) -> tuple[HearingOutcome | None, list[Visit]]:
        tokens = Tokens(2)
        async with scripted(drains, _reads(2)) as (port, visits):
            session = session_to(port, 30, NEEDED_HOLD_BYTES, 5, tokens)
            async with session:
                await tokens.reached(2)
                session.send(bytes(NEEDED_HOLD_BYTES - 1))
                session.send(bytes(1 + extra))
                tokens.release(2)
                async with asyncio.timeout(WHOLE_RUN_SECONDS):
                    async for _ in session.events():
                        pass
            return session.outcome, visits

    held, visits = asyncio.run(scenario(0))
    over, _ = asyncio.run(scenario(1))

    assert held == Closed()
    assert sum(map(len, visits[1].audio)) == NEEDED_HOLD_BYTES
    assert over == OverCap()


def test_audio_held_while_reconnecting_reaches_the_new_session_in_order_once() -> None:
    async def scenario() -> list[list[bytes]]:
        tokens = Tokens(2)
        async with scripted(_reads(2, closes_with=DRAIN_CLOSE_CODE), _reads(2)) as (
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
    first = _reads(len(chunk_sizes) - 1, closes_with=DRAIN_CLOSE_CODE)
    second = says(speech_started("item", 40), then_close=1000)

    positions = asyncio.run(_positions_of(first, second, chunk_sizes))

    assert positions == [40 + shift_ms]


def test_positions_stay_on_one_timeline_across_two_reconnects() -> None:
    async def scenario() -> list[object]:
        tokens = Tokens(2, 3)
        first = _reads(3, closes_with=DRAIN_CLOSE_CODE)
        second = _reads(
            2, closes_with=DRAIN_CLOSE_CODE, says=(speech_started("x", 50),)
        )
        third = says(speech_started("y", 10), then_close=1000)
        async with scripted(first, second, third) as (port, _):
            session = session_to(port, 30, 100_000, 5, tokens)
            async with session:
                reading = asyncio.create_task(_collected(session))
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
                await _hear(port, held_limit_bytes=chunks * len(CHUNK), before=send)
            )[0]

    events = asyncio.run(scenario())

    reconnected = [event for event in events if isinstance(event, Reconnected)]
    assert reconnected == [
        Reconnected(ReconnectCause.CUT, unheard_start_ms, chunks * CHUNK_MS)
    ]


def test_reconnected_comes_before_the_new_sessions_events() -> None:
    async def scenario() -> list[ServerEvent | Reconnected]:
        async with scripted(drains, says(HEARD_DELTA, then_close=1000)) as (port, _):
            return (await _hear(port))[0]

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
            return (await _hear(port))[0]

    assert not any(isinstance(event, Reconnected) for event in asyncio.run(scenario()))


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
