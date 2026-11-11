"""A client that closes right after the upgrade leaves no crash in the log."""

from __future__ import annotations

import asyncio
import base64
import logging
import socket
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
from speech_service.live_session import LiveSession
from tests.support.hear_support import FakeHearing, hear_runtime
from tests.support.identity_support import valid_callers, valid_headers, valid_pair
from tests.support.tcp_server_support import serving_asgi

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pytest
    from starlette.datastructures import Headers
    from starlette.types import ASGIApp, Receive, Scope, Send
    from webauth.service_callers import PersonCaller

    from speech_client.wire import RefusalCode
    from speech_service.callers import SpeechCallers

WAIT_SECONDS = 10


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


class _FirstUpgradeWatched:
    """The app, and an event set once its first upgrade has ended however it ended."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self.first_upgrade_ended = asyncio.Event()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await self._app(scope, receive, send)
        finally:
            if scope["type"] == "websocket":
                self.first_upgrade_ended.set()


class _AdmissionHeldOpen:
    """Real callers whose admission waits, once checked, until `go_on` is set."""

    def __init__(self, callers: SpeechCallers) -> None:
        self._callers = callers
        self.checked = asyncio.Event()
        self.go_on = asyncio.Event()

    async def admit(self, headers: Headers) -> PersonCaller | RefusalCode:
        admitted = await self._callers.admit(headers)
        self.checked.set()
        await self.go_on.wait()
        return admitted


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


async def _upgrading_client(base_url: str) -> socket.socket:
    """A loopback client that has sent an admitted upgrade request."""
    host, port = urlsplit(base_url).netloc.split(":")
    loop = asyncio.get_running_loop()
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(0)
    await loop.sock_connect(client, (host, int(port)))
    await loop.sock_sendall(client, _upgrade_request(base_url))
    return client


async def _leave_as_the_accept_is_sent(
    base_url: str, admission: _AdmissionHeldOpen
) -> None:
    """Close the client so the server's accept meets a closing transport.

    The close is in the server's socket before the next loop pass, and the
    admission goes on at the head of that pass; uvicorn reads the close behind
    it and queues the disconnect one pass later, so the route accepts and sends
    while the transport is closing and its disconnect is not queued yet.
    """
    client = await _upgrading_client(base_url)
    await admission.checked.wait()
    client.close()
    asyncio.get_running_loop().call_soon(admission.go_on.set)


async def _second_hearing_first_event(base_url: str) -> object:
    pair = valid_pair()
    async with HearingSession(
        base_url,
        workload_token=pair.workload,
        person_token=pair.person,
        languages=(HearingLanguage.EN,),
        deadline_seconds=WAIT_SECONDS,
        held_limit_bytes=1024 * 1024,
    ) as session:
        return await asyncio.wait_for(anext(session.events()), WAIT_SECONDS)


def test_a_client_that_leaves_as_the_accept_is_sent_causes_no_crash_log() -> None:
    async def scenario() -> tuple[list[logging.LogRecord], object]:
        admission = _AdmissionHeldOpen(valid_callers())
        app = create_control_app(Settings(), hear_runtime(FakeHearing()), admission)
        watched = _FirstUpgradeWatched(app)
        async with serving_asgi(watched) as base_url:
            with _failures_logged() as failures:
                await _leave_as_the_accept_is_sent(base_url, admission)
                await asyncio.wait_for(watched.first_upgrade_ended.wait(), WAIT_SECONDS)
            return failures.records, await _second_hearing_first_event(base_url)

    crashes, first_event = asyncio.run(scenario())

    assert [record.getMessage() for record in crashes] == []
    assert isinstance(first_event, SessionCreated)


class _StartError(RuntimeError):
    """A failure of the service's own code while it starts the session."""


def test_a_session_start_failure_with_the_client_still_there_is_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_start(_session: LiveSession) -> None:
        raise _StartError

    monkeypatch.setattr(LiveSession, "start", failing_start)

    async def scenario() -> list[logging.LogRecord]:
        app = create_control_app(
            Settings(), hear_runtime(FakeHearing()), valid_callers()
        )
        watched = _FirstUpgradeWatched(app)
        async with serving_asgi(watched) as base_url:
            with _failures_logged() as failures:
                client = await _upgrading_client(base_url)
                await asyncio.wait_for(watched.first_upgrade_ended.wait(), WAIT_SECONDS)
                client.close()
            return failures.records

    crashes = asyncio.run(scenario())

    assert [type(record.exc_info[1]) for record in crashes if record.exc_info] == [
        _StartError
    ]
