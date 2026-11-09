"""The hearing session's outcomes that the real service never produces."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from speech_runtime_client.hearing import HearingSession, Refused
from speech_runtime_client.outcomes import Cut, Malformed, PastDeadline
from speech_runtime_client.speak import Busy
from speech_runtime_client.wire import (
    BusyWait,
    HearingLanguage,
    RefusalCode,
    TranscriptDelta,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from websockets.asyncio.server import ServerConnection

    from speech_runtime_client.hearing import HearingOutcome

LOOPBACK = "127.0.0.1"
WORKLOAD_TOKEN = "workload-secret"  # noqa: S105 - a scenario value
PERSON_TOKEN = "person-secret"  # noqa: S105 - a scenario value
HEARD_TEXT = "heard-secret-words"
ANSWERED_WITHIN_SECONDS = 3
STALL_SECONDS = 1.2


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


def _session(port: int, deadline_seconds: float = 5) -> HearingSession:
    return HearingSession(
        f"http://{LOOPBACK}:{port}",
        workload_token=WORKLOAD_TOKEN,
        person_token=PERSON_TOKEN,
        language=HearingLanguage.DE,
        deadline_seconds=deadline_seconds,
        held_limit_bytes=1,
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
    started = time.monotonic()

    ended = asyncio.run(_outcome_against_raw(_upgrade_answered(answer)))

    assert ended == outcome
    assert time.monotonic() - started < ANSWERED_WITHIN_SECONDS


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
            await asyncio.sleep(0.2)
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


def test_a_service_that_ignores_the_close_does_not_swallow_the_callers_timeout() -> (
    None
):
    async def scenario() -> None:
        released = asyncio.Event()

        async def stalls(connection: ServerConnection) -> None:
            connection.transport.pause_reading()
            await released.wait()

        async with serve(stalls, LOOPBACK, 0, close_timeout=0.2) as server:
            try:
                async with asyncio.timeout(1):
                    async with _session(server.sockets[0].getsockname()[1]):
                        await asyncio.sleep(0.3)
            finally:
                released.set()

    with pytest.raises(TimeoutError):
        asyncio.run(scenario())
