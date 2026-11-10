"""The hearing session against scripted peers the real route cannot be made to play."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING, cast

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.frames import CloseCode

from speech_client.hearing import Closed, HearingSession, Refused
from speech_client.outcomes import (
    Busy,
    Cut,
    Malformed,
    OverCap,
    PastDeadline,
    Unreachable,
)
from speech_client.wire import (
    BusyWait,
    HearingLanguage,
    RefusalCode,
    SessionAudio,
    SessionCreated,
    SessionInput,
    TranscriptDelta,
    TranscriptionSession,
    TranscriptionSetting,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

    from websockets.asyncio.server import ServerConnection

    from speech_client.hearing import HearingOutcome

LOOPBACK = "127.0.0.1"
WORKLOAD_TOKEN = "workload-secret"  # noqa: S105 - a scenario value
PERSON_TOKEN = "person-secret"  # noqa: S105 - a scenario value
HEARD_TEXT = "heard-secret-words"
STALL_SECONDS = 1.2
CLOSE_TIMEOUT_SECONDS = 10
# Well below the socket close timeout (10 s) and the session deadline (30 s).
AT_ONCE_SECONDS = 1
CLOSE_NORMAL = CloseCode.NORMAL_CLOSURE
# A bound the scripted peers meet in time, below the close timeout (10 s).
OPENING_WITH_MARGIN = 2
# A bound the stalled peers pass at once, far below the deadline.
OPENING_SHORT = 0.3


async def _says_nonsense(connection: ServerConnection) -> None:
    await connection.send("not json")
    await connection.wait_closed()


async def _says_bytes(connection: ServerConnection) -> None:
    await connection.send(b"\0")
    await connection.wait_closed()


async def _drops_the_connection(connection: ServerConnection) -> None:
    connection.transport.abort()


async def _hears(connection: ServerConnection) -> None:
    delta = TranscriptDelta(item_id="i", delta=HEARD_TEXT)
    await connection.send(delta.model_dump_json())
    await connection.close()


async def _silent(reader: asyncio.StreamReader, _: object) -> None:
    await reader.read()


async def _leaves(_: HearingSession) -> None:
    return None


async def _passes_the_limit(session: HearingSession) -> None:
    session.send(b"ab")


async def _takes_every_event(session: HearingSession) -> None:
    async for _ in session.events():
        pass


def _session(
    port: int,
    deadline_seconds: float = 5,
    held_limit_bytes: int = 1,
    opening_seconds: float | None = None,
) -> HearingSession:
    # Without a value the session keeps its own default.
    bound = {} if opening_seconds is None else {"opening_seconds": opening_seconds}
    return HearingSession(
        f"http://{LOOPBACK}:{port}",
        workload_token=WORKLOAD_TOKEN,
        person_token=PERSON_TOKEN,
        language=HearingLanguage.DE,
        deadline_seconds=deadline_seconds,
        held_limit_bytes=held_limit_bytes,
        **bound,
    )


async def _outcome_at(
    port: int, deadline_seconds: float = 5, opening_seconds: float | None = None
) -> HearingOutcome | None:
    async with _session(
        port, deadline_seconds, opening_seconds=opening_seconds
    ) as session:
        async with asyncio.timeout(deadline_seconds + 5):
            async for _ in session.events():
                pass
        return session.outcome


async def _outcome_against(handler: object) -> HearingOutcome | None:
    async with serve(handler, LOOPBACK, 0) as server:
        return await _outcome_at(server.sockets[0].getsockname()[1])


def _http_answer(status_line: str, *headers: str, body: bytes = b"") -> bytes:
    head = [status_line, *headers, f"Content-Length: {len(body)}", "Connection: close"]
    return ("\r\n".join(head) + "\r\n\r\n").encode() + body


def _upgrade_answered(answer: bytes) -> Callable[..., object]:
    async def peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(answer)
        await writer.drain()
        writer.close()

    return peer


async def _outcome_against_raw(peer: Callable[..., object]) -> HearingOutcome | None:
    server = await asyncio.start_server(peer, LOOPBACK, 0)
    async with server:
        return await _outcome_at(server.sockets[0].getsockname()[1])


@pytest.mark.parametrize(
    ("handler", "outcome"),
    [
        (_says_nonsense, Malformed()),
        (_says_bytes, Malformed()),
        (_drops_the_connection, Cut()),
    ],
)
def test_a_server_that_breaks_the_wire_or_drops_ends_the_session_with_its_outcome(
    handler: object, outcome: HearingOutcome
) -> None:
    assert asyncio.run(_outcome_against(handler)) == outcome


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        (b"", Cut()),
        (b"HELLO\r\n\r\n", Malformed()),
        (_http_answer("HTTP/1.1 200 OK"), Malformed()),
        (_http_answer("HTTP/1.1 101 Switching Protocols"), Malformed()),
        (
            _http_answer("HTTP/1.1 302 Found", "Location: ws://127.0.0.1:1/"),
            Malformed(),
        ),
        (
            _http_answer("HTTP/1.1 503 Service Unavailable", body=b"no"),
            Refused(503, None),
        ),
        (
            _http_answer("HTTP/1.1 429 Too Many Requests", "Retry-After: 7"),
            Busy(BusyWait(retry_after_seconds=7)),
        ),
        (_http_answer("HTTP/1.1 429 Too Many Requests"), Malformed()),
        (
            _http_answer("HTTP/1.1 429 Too Many Requests", "Retry-After: soon"),
            Malformed(),
        ),
    ],
)
def test_an_upgrade_answered_off_the_wire_ends_the_session_with_its_outcome_at_once(
    answer: bytes, outcome: HearingOutcome
) -> None:
    ended = asyncio.run(_outcome_against_raw(_upgrade_answered(answer)))

    assert ended == outcome


def test_a_refused_upgrade_with_the_contract_body_gives_its_code() -> None:
    body = (
        b'{"error":{"message":"m","type":"server_error","param":null,'
        b'"code":"unavailable"}}'
    )
    answer = _http_answer("HTTP/1.1 503 Service Unavailable", body=body)

    ended = asyncio.run(_outcome_against_raw(_upgrade_answered(answer)))

    assert ended == Refused(503, RefusalCode.UNAVAILABLE)


def test_a_redirect_is_not_followed() -> None:
    async def scenario() -> tuple[HearingOutcome | None, list[bytes]]:
        arrivals: list[bytes] = []

        async def target(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            arrivals.append(await reader.read(1))
            writer.close()

        async with await asyncio.start_server(target, LOOPBACK, 0) as elsewhere:
            port = elsewhere.sockets[0].getsockname()[1]
            redirect = _http_answer(
                "HTTP/1.1 302 Found", f"Location: ws://{LOOPBACK}:{port}/"
            )
            ended = await _outcome_against_raw(_upgrade_answered(redirect))
        return ended, arrivals

    ended, arrivals = asyncio.run(scenario())

    assert ended == Malformed()
    assert arrivals == []


def test_a_finished_session_ends_every_further_events_call_at_once() -> None:
    async def scenario() -> None:
        async with (
            serve(_says_nonsense, LOOPBACK, 0) as server,
            _session(server.sockets[0].getsockname()[1]) as session,
        ):
            async for _ in session.events():
                pass
            async with asyncio.timeout(1):
                async for _ in session.events():
                    pass

    asyncio.run(scenario())


def test_a_deadline_below_the_opening_bound_ends_a_stalled_opening_past_deadline() -> (
    None
):
    async def scenario() -> float:
        async with await asyncio.start_server(_silent, LOOPBACK, 0) as server:
            started = time.monotonic()
            ended = await _outcome_at(server.sockets[0].getsockname()[1], 1.5)
            assert ended == PastDeadline()
            return time.monotonic() - started

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(connect.__init__.__kwdefaults__, "open_timeout", 0.2)
        elapsed = asyncio.run(scenario())

    assert elapsed > STALL_SECONDS


def test_the_sockets_log_at_debug_carries_no_token_and_no_heard_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    asyncio.run(_outcome_against(_hears))

    client_log = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name != "websockets.server"
    )
    for secret in (WORKLOAD_TOKEN, PERSON_TOKEN, HEARD_TEXT):
        assert secret not in client_log


@contextlib.asynccontextmanager
async def _stalled_service(
    says_first: tuple[str, ...] = (),
) -> AsyncIterator[tuple[int, asyncio.Event]]:
    """A service that reads the first frame, then stops reading; yields port, opened."""
    released = asyncio.Event()
    opened = asyncio.Event()

    async def stalls(connection: ServerConnection) -> None:
        await connection.recv()
        for message in says_first:
            await connection.send(message)
        opened.set()
        connection.transport.pause_reading()
        await released.wait()

    async with serve(stalls, LOOPBACK, 0, close_timeout=0.2) as server:
        try:
            yield server.sockets[0].getsockname()[1], opened
        finally:
            released.set()


def test_a_service_that_ignores_the_close_does_not_swallow_the_callers_timeout() -> (
    None
):
    async def scenario() -> None:
        async with _stalled_service() as (port, opened):
            async with asyncio.timeout(1):
                async with _session(port):
                    await opened.wait()

    with pytest.raises(TimeoutError):
        asyncio.run(scenario())


@pytest.mark.parametrize(
    ("says_first", "end", "outcome", "deadline_seconds", "opening_seconds"),
    [
        pytest.param((), _leaves, Closed(), 1, None, id="leave"),
        pytest.param((), _passes_the_limit, OverCap(), 1, None, id="over-limit"),
        pytest.param(
            ("not json",), _takes_every_event, Malformed(), 1, None, id="nonsense"
        ),
        pytest.param(
            (), _leaves, Closed(), 30, OPENING_WITH_MARGIN, id="leave-opening-bound"
        ),
        pytest.param(
            (),
            _passes_the_limit,
            OverCap(),
            30,
            OPENING_WITH_MARGIN,
            id="over-limit-opening-bound",
        ),
    ],
)
def test_the_session_bounds_free_a_caller_leaving_a_service_ignoring_the_close(
    says_first: tuple[str, ...],
    end: Callable[[HearingSession], Awaitable[None]],
    outcome: HearingOutcome,
    deadline_seconds: float,
    opening_seconds: float | None,
) -> None:
    async def scenario() -> tuple[HearingOutcome | None, float]:
        async with _stalled_service(says_first) as (port, opened):
            session = _session(port, deadline_seconds, opening_seconds=opening_seconds)
            async with session:
                await opened.wait()
                started = time.monotonic()
                await end(session)
            return session.outcome, time.monotonic() - started

    ended, elapsed = asyncio.run(scenario())

    assert ended == outcome
    assert elapsed < CLOSE_TIMEOUT_SECONDS / 2


def test_a_caller_cancelled_while_leaving_a_stalled_service_is_not_held_by_it() -> None:
    async def scenario() -> float:
        async with _stalled_service() as (port, opened):
            started = time.monotonic()
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(0.3):
                    async with _session(port, 30):
                        await opened.wait()
            return time.monotonic() - started

    assert asyncio.run(scenario()) < CLOSE_TIMEOUT_SECONDS / 2


def test_a_send_after_the_held_limit_is_passed_never_reaches_the_peer() -> None:
    async def scenario() -> tuple[HearingOutcome | None, list[bytes]]:
        received: list[bytes] = []
        setup_received = asyncio.Event()
        closed = asyncio.Event()

        async def records(connection: ServerConnection) -> None:
            try:
                async for message in connection:
                    audio = json.loads(message).get("audio")
                    if audio is None:
                        setup_received.set()
                    else:
                        received.append(base64.b64decode(audio))
            finally:
                closed.set()

        async with serve(records, LOOPBACK, 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with _session(port, held_limit_bytes=2) as session:
                async with asyncio.timeout(5):
                    await setup_received.wait()
                for pcm in (b"a", b"bcd", b"e"):
                    session.send(pcm)
                async with asyncio.timeout(5):
                    async for _ in session.events():
                        pass
                outcome = session.outcome
            async with asyncio.timeout(5):
                await closed.wait()
        return outcome, received

    outcome, received = asyncio.run(scenario())

    assert outcome == OverCap()
    assert received == [b"a"]


def test_a_peer_that_stops_reading_does_not_hold_the_session_past_its_deadline() -> (
    None
):
    async def scenario() -> tuple[HearingOutcome | None, float]:
        async with _stalled_service() as (port, _):
            started = time.monotonic()
            outcome = await _outcome_at(port, 1)
            return outcome, time.monotonic() - started

    outcome, elapsed = asyncio.run(scenario())

    assert outcome == PastDeadline()
    assert elapsed < CLOSE_TIMEOUT_SECONDS / 2


_HEARD = TranscriptDelta(item_id="i", delta=HEARD_TEXT).model_dump_json()


@pytest.mark.parametrize(
    ("says", "end", "outcome"),
    [
        pytest.param([], _leaves, Closed(), id="leave-after-open"),
        pytest.param([], _passes_the_limit, OverCap(), id="over-limit-left-at-once"),
        pytest.param([_HEARD, _HEARD], _leaves, Closed(), id="two-events-untaken"),
        pytest.param(["not json"], _takes_every_event, Malformed(), id="nonsense"),
    ],
)
def test_every_orderly_end_closes_the_connection_with_the_normal_code(
    says: list[str],
    end: Callable[[HearingSession], Awaitable[None]],
    outcome: HearingOutcome,
) -> None:
    async def scenario() -> tuple[HearingOutcome | None, list[int | None]]:
        codes: list[int | None] = []
        opened = asyncio.Event()

        async def records(connection: ServerConnection) -> None:
            await connection.recv()
            for message in says:
                await connection.send(message)
            opened.set()
            await connection.wait_closed()
            codes.append(connection.close_code)

        async with serve(records, LOOPBACK, 0) as server:
            session = _session(server.sockets[0].getsockname()[1])
            async with session:
                await opened.wait()
                await end(session)
        return session.outcome, codes

    assert asyncio.run(scenario()) == (outcome, [CLOSE_NORMAL])


@pytest.mark.parametrize(
    "end",
    [
        pytest.param(_leaves, id="leave"),
        pytest.param(_passes_the_limit, id="over-limit"),
    ],
)
@pytest.mark.parametrize("yields_before_cancel", range(31))
def test_a_cancel_at_any_moment_after_an_orderly_end_returns_at_once(
    end: Callable[[HearingSession], Awaitable[None]], yields_before_cancel: int
) -> None:
    async def hearing(port: int, ended: asyncio.Event) -> None:
        async with _session(port, 30) as session:
            await ended.wait()
            await end(session)

    async def scenario() -> float:
        async with _stalled_service() as (port, opened):
            ended = asyncio.Event()
            caller = asyncio.create_task(hearing(port, ended))
            await opened.wait()
            ended.set()
            for _ in range(yields_before_cancel):
                await asyncio.sleep(0)
            started = time.monotonic()
            caller.cancel()
            await asyncio.wait({caller}, timeout=CLOSE_TIMEOUT_SECONDS / 2)
            elapsed = time.monotonic() - started
            assert caller.cancelled()
        assert asyncio.all_tasks() == {asyncio.current_task()}
        return elapsed

    assert asyncio.run(scenario()) < AT_ONCE_SECONDS


@pytest.mark.parametrize(
    ("end", "outcome"),
    [
        pytest.param(_passes_the_limit, OverCap(), id="over-limit"),
        pytest.param(_leaves, Closed(), id="leave"),
    ],
)
def test_an_end_before_the_connection_opens_ends_the_session_at_once(
    end: Callable[[HearingSession], Awaitable[None]], outcome: HearingOutcome
) -> None:
    async def scenario() -> tuple[HearingOutcome | None, float]:
        async with await asyncio.start_server(_silent, LOOPBACK, 0) as server:
            session = _session(server.sockets[0].getsockname()[1], 30)
            started = time.monotonic()
            async with session:
                await end(session)
            return session.outcome, time.monotonic() - started

    ended, elapsed = asyncio.run(scenario())

    assert ended == outcome
    assert elapsed < AT_ONCE_SECONDS


def test_a_caller_timeout_during_a_stalled_opening_leaves_no_task_behind() -> None:
    async def scenario() -> float:
        peers: set[asyncio.Task[object] | None] = set()

        async def silent(reader: asyncio.StreamReader, writer: object) -> None:
            peers.add(asyncio.current_task())
            await _silent(reader, writer)

        async with await asyncio.start_server(silent, LOOPBACK, 0) as server:
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                async with (
                    asyncio.timeout(0.2),
                    _session(server.sockets[0].getsockname()[1], 30) as session,
                ):
                    await _takes_every_event(session)
            elapsed = time.monotonic() - started
            assert asyncio.all_tasks() - peers == {asyncio.current_task()}
        return elapsed

    assert asyncio.run(scenario()) < AT_ONCE_SECONDS


def _fails_the_writer(session: HearingSession) -> None:
    # A str is the one input through the public surface that the writer cannot encode.
    session.send(cast("bytes", "x"))


@pytest.mark.parametrize(
    "says_first", [(), (_HEARD, _HEARD)], ids=["quiet", "reader-blocked-on-full-queue"]
)
def test_a_failing_writer_ends_the_session_at_once_and_is_raised_from_leaving(
    says_first: tuple[str, ...],
) -> None:
    async def scenario() -> tuple[bool, HearingOutcome | None]:
        events_ran_out = False

        async def hearing(session: HearingSession, opened: asyncio.Event) -> None:
            nonlocal events_ran_out
            async with session:
                await opened.wait()
                _fails_the_writer(session)
                # The session ends while the caller has taken no event yet.
                await asyncio.wait({session._runner})  # type: ignore[arg-type]  # noqa: SLF001
                async with asyncio.timeout(AT_ONCE_SECONDS):
                    await _takes_every_event(session)
                events_ran_out = True

        async with _stalled_service(says_first) as (port, opened):
            session = _session(port)
            with pytest.raises(TypeError):
                await hearing(session, opened)
        assert asyncio.all_tasks() == {asyncio.current_task()}
        return events_ran_out, session.outcome

    assert asyncio.run(scenario()) == (True, None)


@pytest.mark.parametrize("yields_before_cancel", range(9))
def test_an_unexpected_failure_beats_a_cancel_landing_with_it(
    yields_before_cancel: int,
) -> None:
    async def hearing(
        session: HearingSession, opened: asyncio.Event, failed: asyncio.Event
    ) -> None:
        async with session:
            await opened.wait()
            _fails_the_writer(session)
            failed.set()
            await asyncio.Event().wait()

    async def scenario() -> HearingOutcome | None:
        async with _stalled_service() as (port, opened):
            session = _session(port)
            failed = asyncio.Event()
            caller = asyncio.create_task(hearing(session, opened, failed))
            await failed.wait()
            for _ in range(yields_before_cancel):
                await asyncio.sleep(0)
            caller.cancel()
            await asyncio.wait({caller}, timeout=CLOSE_TIMEOUT_SECONDS / 2)
            with pytest.raises(TypeError):
                caller.result()
        assert asyncio.all_tasks() == {asyncio.current_task()}
        return session.outcome

    assert asyncio.run(scenario()) is None


def _chain(error: BaseException | None) -> Iterator[BaseException]:
    while error is not None:
        yield error
        error = error.__cause__ or error.__context__


async def _fails_when_cancelled() -> None:
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        message = "the caller's own code failed"
        raise RuntimeError(message) from None


async def _a_task_group_whose_child_fails_when_cancelled() -> None:
    async with asyncio.TaskGroup() as group:
        group.create_task(_fails_when_cancelled())


@pytest.mark.parametrize(
    ("callers_code", "turned_into"),
    [
        pytest.param(_fails_when_cancelled, RuntimeError, id="absorbed"),
        pytest.param(
            _a_task_group_whose_child_fails_when_cancelled,
            ExceptionGroup,
            id="task-group",
        ),
    ],
)
def test_a_leave_whose_cancel_the_callers_code_turned_into_another_error_aborts(
    callers_code: Callable[[], Awaitable[None]], turned_into: type[BaseException]
) -> None:
    async def hearing(port: int, opened: asyncio.Event) -> None:
        async with asyncio.timeout(0.3), _session(port, 30):
            await opened.wait()
            await callers_code()

    async def scenario() -> tuple[BaseException, float]:
        async with _stalled_service() as (port, opened):
            started = time.monotonic()
            with pytest.raises(TimeoutError) as raised:
                await hearing(port, opened)
            elapsed = time.monotonic() - started
        assert asyncio.all_tasks() == {asyncio.current_task()}
        return raised.value, elapsed

    timed_out, elapsed = asyncio.run(scenario())

    assert elapsed < CLOSE_TIMEOUT_SECONDS / 2
    assert any(isinstance(error, turned_into) for error in _chain(timed_out))


@pytest.mark.parametrize(
    "base_url",
    [
        "http://host/path",
        "http://127.0.0.1:1#fragment",
        "http://127.0.0.1:1?query=1",
        "http://127.0.0.1:1?",
        "http://127.0.0.1:1#",
    ],
)
def test_a_base_url_that_is_not_an_origin_raises_value_error(base_url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        HearingSession(
            base_url,
            workload_token=WORKLOAD_TOKEN,
            person_token=PERSON_TOKEN,
            language=HearingLanguage.DE,
            deadline_seconds=1,
            held_limit_bytes=1,
        )


def _session_created() -> str:
    setting = TranscriptionSetting(language=HearingLanguage.DE)
    session = TranscriptionSession(
        type="transcription",
        audio=SessionAudio(input=SessionInput(transcription=setting)),
    )
    return SessionCreated(session=session).model_dump_json()


@pytest.mark.parametrize(
    "stall",
    [
        pytest.param("handshake", id="handshake-never-answered"),
        pytest.param("first-event", id="no-event-after-the-handshake"),
    ],
)
def test_a_stalled_opening_ends_with_unreachable_at_the_opening_bound(
    stall: str,
) -> None:
    async def scenario() -> tuple[HearingOutcome | None, float]:
        async with contextlib.AsyncExitStack() as stack:
            if stall == "handshake":
                server = await stack.enter_async_context(
                    await asyncio.start_server(_silent, LOOPBACK, 0)
                )
                port = server.sockets[0].getsockname()[1]
            else:
                port, _ = await stack.enter_async_context(_stalled_service())
            started = time.monotonic()
            ended = await _outcome_at(port, 4, OPENING_SHORT)
            return ended, time.monotonic() - started

    ended, elapsed = asyncio.run(scenario())

    assert ended == Unreachable()
    assert elapsed < AT_ONCE_SECONDS * 2


@pytest.mark.parametrize(
    "first_event",
    [
        pytest.param(_session_created(), id="session-created"),
        pytest.param(_HEARD, id="a-delta-first"),
    ],
)
def test_a_service_that_opened_in_time_may_stay_silent_past_the_opening_bound(
    first_event: str,
) -> None:
    async def scenario() -> HearingOutcome | None:
        async def says_first_then_waits(connection: ServerConnection) -> None:
            await connection.recv()
            await connection.send(first_event)
            await connection.wait_closed()

        async with serve(says_first_then_waits, LOOPBACK, 0) as server:
            port = server.sockets[0].getsockname()[1]
            session = _session(port, 10, opening_seconds=OPENING_WITH_MARGIN)
            async with session:
                async with asyncio.timeout(OPENING_WITH_MARGIN):
                    await anext(session.events())
                # The one wait past the bound: it must not end the open session.
                await asyncio.sleep(OPENING_WITH_MARGIN + 0.3)
                assert session.outcome is None
            return session.outcome

    assert asyncio.run(scenario()) == Closed()


def test_a_caller_timeout_during_the_wait_for_the_first_event_leaves_no_task() -> None:
    async def hearing(port: int, opened: asyncio.Event) -> None:
        async with (
            asyncio.timeout(0.2),
            _session(port, 30, opening_seconds=10) as session,
        ):
            await opened.wait()
            await _takes_every_event(session)

    async def scenario() -> float:
        async with _stalled_service() as (port, opened):
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                await hearing(port, opened)
            elapsed = time.monotonic() - started
        assert asyncio.all_tasks() == {asyncio.current_task()}
        return elapsed

    assert asyncio.run(scenario()) < AT_ONCE_SECONDS


def test_frames_sent_before_a_stalled_opening_ends_reach_the_peer_only_before_it() -> (
    None
):
    async def scenario() -> tuple[HearingOutcome | None, list[bytes], list[bytes]]:
        received: list[bytes] = []
        flushed = asyncio.Event()
        closed = asyncio.Event()

        async def records(connection: ServerConnection) -> None:
            try:
                async for message in connection:
                    audio = json.loads(message).get("audio")
                    if audio is not None:
                        received.append(base64.b64decode(audio))
                        if len(received) == 2:  # noqa: PLR2004 - the two frames sent
                            flushed.set()
            finally:
                closed.set()

        async with serve(records, LOOPBACK, 0) as server:
            port = server.sockets[0].getsockname()[1]
            session = _session(
                port, 30, held_limit_bytes=10, opening_seconds=OPENING_WITH_MARGIN
            )
            async with session:
                session.send(b"a")
                session.send(b"b")
                async with asyncio.timeout(OPENING_WITH_MARGIN):
                    await flushed.wait()
                async with asyncio.timeout(OPENING_WITH_MARGIN * 2):
                    await _takes_every_event(session)
                session.send(b"c")
            async with asyncio.timeout(5):
                await closed.wait()
            return session.outcome, received[:2], received[2:]

    outcome, before, after = asyncio.run(scenario())

    assert outcome == Unreachable()
    assert before == [b"a", b"b"]
    assert after == []


@pytest.mark.parametrize("opening_seconds", [0, -1])
def test_an_opening_bound_at_or_below_zero_raises_value_error(
    opening_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="opening_seconds"):
        _session(1, opening_seconds=opening_seconds)
