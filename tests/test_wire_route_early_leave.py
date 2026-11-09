"""A client that closes right after the upgrade leaves no crash in the log."""

from __future__ import annotations

import asyncio
import base64
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from speech_client.hearing import HearingSession
from speech_client.wire import (
    REALTIME_INTENT,
    REALTIME_INTENT_PARAM,
    REALTIME_PATH,
    HearingLanguage,
    SessionCreated,
)
from speech_service.config import Settings
from speech_service.control import create_control_app
from tests.support.hear_support import FakeHearing, hear_runtime
from tests.support.identity_support import valid_callers, valid_headers, valid_pair
from tests.support.tcp_server_support import serving_asgi

if TYPE_CHECKING:
    from collections.abc import Iterator

    from starlette.types import ASGIApp, Message, Receive, Scope, Send

WAIT_SECONDS = 10
UVICORN_SEND_BEFORE_ACCEPT = (
    "Expected ASGI message 'websocket.accept', 'websocket.close' "
    "or 'websocket.http.response.start' but got 'websocket.send'."
)


class _Failures(logging.Handler):
    """The records at error level or above, with the exceptions they carry."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _failures_logged() -> Iterator[_Failures]:
    failures = _Failures()
    root = logging.getLogger()
    uvicorn_errors = logging.getLogger("uvicorn.error")
    root.addHandler(failures)
    uvicorn_errors.addHandler(failures)
    try:
        yield failures
    finally:
        root.removeHandler(failures)
        uvicorn_errors.removeHandler(failures)


def _upgrade_request(base_url: str) -> bytes:
    """An admitted upgrade request."""
    headers = {
        "Host": urlsplit(base_url).netloc,
        "Upgrade": "websocket",
        "Connection": "Upgrade",
        "Sec-WebSocket-Key": base64.b64encode(bytes(16)).decode(),
        "Sec-WebSocket-Version": "13",
        **valid_headers(),
    }
    lines = [
        f"GET {REALTIME_PATH}?{REALTIME_INTENT_PARAM}={REALTIME_INTENT} HTTP/1.1",
        *(f"{name}: {value}" for name, value in headers.items()),
    ]
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


class _ServerThatSawTheClientLeave:
    """The first upgrade as uvicorn serves one whose transport closed before accept.

    uvicorn drops the accept of a closing transport and then rejects the first
    send; the leave is not ordered against the accept on a real socket, so this
    holds the accept until uvicorn has queued the disconnect.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self._leave_seen = False
        self.first_upgrade_ended = asyncio.Event()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "websocket" or self._leave_seen:
            await self._app(scope, receive, send)
            return
        self._leave_seen = True
        queued: list[Message] = []

        async def replaying_receive() -> Message:
            return queued.pop() if queued else await receive()

        async def closing_send(message: Message) -> None:
            if message["type"] == "websocket.accept":
                queued.append(await receive())
            elif message["type"] == "websocket.send":
                raise RuntimeError(UVICORN_SEND_BEFORE_ACCEPT)

        try:
            await self._app(scope, replaying_receive, closing_send)
        finally:
            self.first_upgrade_ended.set()


async def _leave_right_after_the_upgrade(base_url: str) -> None:
    host, port = urlsplit(base_url).netloc.split(":")
    _, writer = await asyncio.open_connection(host, int(port))
    writer.write(_upgrade_request(base_url))
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def _second_hearing_first_event(base_url: str) -> object:
    pair = valid_pair()
    async with HearingSession(
        base_url,
        workload_token=pair.workload,
        person_token=pair.person,
        language=HearingLanguage.EN,
        deadline_seconds=WAIT_SECONDS,
        held_limit_bytes=1024 * 1024,
    ) as session:
        return await asyncio.wait_for(anext(session.events()), WAIT_SECONDS)


def test_a_client_that_leaves_right_after_the_upgrade_causes_no_crash_log() -> None:
    async def scenario() -> tuple[list[logging.LogRecord], object]:
        app = create_control_app(
            Settings(), hear_runtime(FakeHearing()), valid_callers()
        )
        server = _ServerThatSawTheClientLeave(app)
        async with serving_asgi(server) as base_url:
            with _failures_logged() as failures:
                await _leave_right_after_the_upgrade(base_url)
                await asyncio.wait_for(server.first_upgrade_ended.wait(), WAIT_SECONDS)
            return failures.records, await _second_hearing_first_event(base_url)

    crashes, first_event = asyncio.run(scenario())

    assert [record.getMessage() for record in crashes] == []
    assert isinstance(first_event, SessionCreated)
