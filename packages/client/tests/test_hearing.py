"""The hearing session against scripted peers the real route cannot be made to play."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.frames import CloseCode

from speech_client.hearing import HearingSession, Refused
from speech_client.outcomes import Busy, Cut, Malformed, OverCap, PastDeadline
from speech_client.wire import (
    BusyWait,
    HearingLanguage,
    RefusalCode,
    TranscriptDelta,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from websockets.asyncio.server import ServerConnection

    from speech_client.hearing import HearingOutcome

LOOPBACK = "127.0.0.1"
WORKLOAD_TOKEN = "workload-secret"  # noqa: S105 - a scenario value
PERSON_TOKEN = "person-secret"  # noqa: S105 - a scenario value
HEARD_TEXT = "heard-secret-words"
STALL_SECONDS = 1.2
CLOSE_TIMEOUT_SECONDS = 10
CLOSE_NORMAL = CloseCode.NORMAL_CLOSURE


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


def _session(
    port: int, deadline_seconds: float = 5, held_limit_bytes: int = 1
) -> HearingSession:
    return HearingSession(
        f"http://{LOOPBACK}:{port}",
        workload_token=WORKLOAD_TOKEN,
        person_token=PERSON_TOKEN,
        language=HearingLanguage.DE,
        deadline_seconds=deadline_seconds,
        held_limit_bytes=held_limit_bytes,
    )


async def _outcome_at(port: int, deadline_seconds: float = 5) -> HearingOutcome | None:
    async with _session(port, deadline_seconds) as session:
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


def test_a_stalled_opening_waits_for_the_deadline_not_for_the_sockets_own() -> None:
    async def scenario() -> float:
        async def silent(reader: asyncio.StreamReader, _: object) -> None:
            await reader.read()

        async with await asyncio.start_server(silent, LOOPBACK, 0) as server:
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
async def _stalled_service() -> AsyncIterator[tuple[int, asyncio.Event]]:
    """A service that reads the first frame, then stops reading; yields port, opened."""
    released = asyncio.Event()
    opened = asyncio.Event()

    async def stalls(connection: ServerConnection) -> None:
        await connection.recv()
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


def test_the_session_deadline_frees_a_caller_leaving_a_service_ignoring_the_close() -> (
    None
):
    async def scenario() -> float:
        async with _stalled_service() as (port, opened):
            async with _session(port, 1):
                await opened.wait()
                started = time.monotonic()
            return time.monotonic() - started

    assert asyncio.run(scenario()) < CLOSE_TIMEOUT_SECONDS / 2


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


def test_leaving_the_context_closes_the_connection_with_the_normal_code() -> None:
    async def scenario() -> int | None:
        codes: list[int | None] = []
        opened = asyncio.Event()

        async def records(connection: ServerConnection) -> None:
            await connection.recv()
            opened.set()
            await connection.wait_closed()
            codes.append(connection.close_code)

        async with (
            serve(records, LOOPBACK, 0) as server,
            _session(server.sockets[0].getsockname()[1]),
        ):
            await opened.wait()
        return codes[0]

    assert asyncio.run(scenario()) == CLOSE_NORMAL


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
