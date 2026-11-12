"""The client's relay between a browser socket and the real service over TCP."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI, WebSocket
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from speech_client.hearing import HearingSession
from speech_client.outcomes import Busy
from speech_client.relay import (
    AppEnded,
    BrowserLeft,
    HopClose,
    RelayOutcome,
    RuleBroken,
    relay,
)
from speech_client.wire import (
    Language,
    SessionCreated,
    TranscriptCompleted,
    TranscriptLanguage,
)
from tests.support.hear_support import HEARD_TEXT, FakeHearing, hear_runtime
from tests.support.identity_support import valid_pair
from tests.support.live_support import INPUT_RATE, SPEECH_LEVEL
from tests.support.tcp_server_support import serving_asgi, serving_over_tcp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from speech_client.wire import ServerEvent

WAIT_SECONDS = 10
FRAME_LIMIT_BYTES = 64 * 1024
END_BOUND_SECONDS = 2


def _tone(seconds: float, level: int = SPEECH_LEVEL) -> bytes:
    return struct.pack("<h", level) * int(seconds * INPUT_RATE)


@dataclass
class _Hop:
    """What the app side of one relayed hearing saw."""

    events: list[ServerEvent] = field(default_factory=list)
    outcome: asyncio.Future[RelayOutcome] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    app_ended: asyncio.Event = field(default_factory=asyncio.Event)
    completed: asyncio.Event = field(default_factory=asyncio.Event)
    first_event: asyncio.Event = field(default_factory=asyncio.Event)
    tokens: tuple[str, str] = ("", "")


def _tokens() -> tuple[str, str]:
    pair = valid_pair()
    return pair.workload, pair.person


def _session(service_url: str, tokens: tuple[str, str]) -> HearingSession:
    return HearingSession(
        service_url,
        workload_token=tokens[0],
        person_token=tokens[1],
        languages=(Language.EN,),
        deadline_seconds=WAIT_SECONDS,
        held_limit_bytes=1024 * 1024,
    )


def _app_host(service_url: str, hop: _Hop) -> FastAPI:
    """A minimal app: a stand-in login check, then the relay."""
    host = FastAPI()

    @host.websocket("/hear")
    async def hear(browser: WebSocket) -> None:
        await browser.accept()
        hop.tokens = _tokens()

        async def take(event: ServerEvent) -> None:
            hop.events.append(event)
            hop.first_event.set()
            if isinstance(event, TranscriptCompleted):
                hop.completed.set()

        hop.outcome.set_result(
            await relay(
                browser,
                _session(service_url, hop.tokens),
                app_ended=hop.app_ended,
                on_event=take,
                frame_limit_bytes=FRAME_LIMIT_BYTES,
            )
        )

    return host


@asynccontextmanager
async def _serving(app: FastAPI) -> AsyncIterator[str]:
    async with serving_asgi(app) as base_url:
        yield f"ws{base_url.removeprefix('http')}/hear"


@asynccontextmanager
async def _browser_at_service() -> AsyncIterator[tuple[_Hop, str, str]]:
    """The real service, an app relaying to it, and the app's hop."""
    async with serving_over_tcp(hear_runtime(FakeHearing())) as service:
        hop = _Hop()
        async with _serving(_app_host(service.base_url, hop)) as url:
            yield hop, url, service.base_url


async def _received(browser: ClientConnection) -> list[str | bytes]:
    """Every message the browser gets until its socket closes."""
    received: list[str | bytes] = []
    with contextlib.suppress(ConnectionClosed):
        async for message in browser:
            received.append(message)  # noqa: PERF401 - kept when the close raises
    return received


async def _close_code(browser: ClientConnection) -> int:
    receiving = asyncio.wait_for(browser.recv(), WAIT_SECONDS)
    with pytest.raises(ConnectionClosed) as closed:
        await receiving
    assert closed.value.rcvd is not None
    return closed.value.rcvd.code


async def _a_new_session_opens(service_url: str) -> bool:
    """Whether the service takes a session now: its one place is free."""
    async with _session(service_url, _tokens()) as session:
        first = await asyncio.wait_for(anext(session.events()), WAIT_SECONDS)
    return isinstance(first, SessionCreated)


def test_frames_reach_the_service_and_its_events_reach_the_app_not_the_browser() -> (
    None
):
    async def scenario() -> tuple[list[ServerEvent], list[str | bytes], RelayOutcome]:
        async with _browser_at_service() as (hop, url, _), connect(url) as browser:
            await browser.send(_tone(0.5))
            await browser.send(_tone(1.0, level=0))
            await asyncio.wait_for(hop.completed.wait(), WAIT_SECONDS)
            await browser.close()
            received = await _received(browser)
            return (
                hop.events,
                received,
                await asyncio.wait_for(hop.outcome, WAIT_SECONDS),
            )

    events, received, outcome = asyncio.run(scenario())

    assert isinstance(events[0], SessionCreated)
    assert events[-1].transcript == HEARD_TEXT
    assert events[-1].languages == (TranscriptLanguage(code=Language.EN),)
    assert received == []
    assert outcome == BrowserLeft()


def test_a_browser_that_leaves_ends_the_session_and_frees_the_service() -> None:
    async def scenario() -> tuple[RelayOutcome, bool]:
        async with _browser_at_service() as (hop, url, service_url):
            async with connect(url) as browser:
                await browser.send(_tone(0.1))
                await asyncio.wait_for(hop.first_event.wait(), WAIT_SECONDS)
            outcome = await asyncio.wait_for(hop.outcome, WAIT_SECONDS)
            return outcome, await _a_new_session_opens(service_url)

    outcome, freed = asyncio.run(scenario())

    assert outcome == BrowserLeft()
    assert freed


def test_the_apps_end_signal_ends_the_session_and_the_hop_within_its_bound() -> None:
    async def scenario() -> tuple[RelayOutcome, int, bool]:
        async with _browser_at_service() as (hop, url, service_url):
            async with connect(url) as browser:
                await browser.send(_tone(0.1))
                await asyncio.wait_for(hop.first_event.wait(), WAIT_SECONDS)
                hop.app_ended.set()
                outcome = await asyncio.wait_for(hop.outcome, END_BOUND_SECONDS)
                code = await asyncio.wait_for(_close_code(browser), END_BOUND_SECONDS)
            return outcome, code, await _a_new_session_opens(service_url)

    outcome, code, freed = asyncio.run(scenario())

    assert outcome == AppEnded()
    assert code == HopClose.NORMAL
    assert freed


def test_a_text_frame_ends_the_hop_with_its_close_code() -> None:
    async def scenario() -> tuple[RelayOutcome, int]:
        async with _browser_at_service() as (hop, url, _), connect(url) as browser:
            await browser.send("not audio")
            return await asyncio.wait_for(hop.outcome, WAIT_SECONDS), await _close_code(
                browser
            )

    outcome, code = asyncio.run(scenario())

    assert outcome == RuleBroken(HopClose.TEXT_FRAME)
    assert code == HopClose.TEXT_FRAME


def test_an_oversized_frame_ends_the_hop_with_its_close_code() -> None:
    async def scenario() -> tuple[RelayOutcome, int]:
        async with _browser_at_service() as (hop, url, _), connect(url) as browser:
            await browser.send(b"\0" * (FRAME_LIMIT_BYTES + 1))
            return await asyncio.wait_for(hop.outcome, WAIT_SECONDS), await _close_code(
                browser
            )

    outcome, code = asyncio.run(scenario())

    assert outcome == RuleBroken(HopClose.FRAME_TOO_LARGE)
    assert code == HopClose.FRAME_TOO_LARGE


def test_a_busy_service_reaches_the_app_as_busy_and_the_browser_as_its_code() -> None:
    async def scenario() -> tuple[RelayOutcome, int]:
        async with (
            _browser_at_service() as (hop, url, service_url),
            _session(service_url, _tokens()) as taking,
        ):
            await asyncio.wait_for(anext(taking.events()), WAIT_SECONDS)
            async with connect(url) as browser:
                return (
                    await asyncio.wait_for(hop.outcome, WAIT_SECONDS),
                    await _close_code(browser),
                )

    outcome, code = asyncio.run(scenario())

    assert isinstance(outcome, Busy)
    assert code == HopClose.BUSY


def test_no_token_and_no_heard_text_reaches_a_log_or_the_browser(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    async def scenario() -> tuple[_Hop, list[str | bytes]]:
        async with _browser_at_service() as (hop, url, _), connect(url) as browser:
            await browser.send(_tone(0.5))
            await browser.send(_tone(1.0, level=0))
            await asyncio.wait_for(hop.completed.wait(), WAIT_SECONDS)
            hop.app_ended.set()
            received = await _received(browser)
            await asyncio.wait_for(hop.outcome, WAIT_SECONDS)
            return hop, received

    hop, received = asyncio.run(scenario())
    logged = " ".join(record.getMessage() for record in caplog.records)

    assert received == []
    assert not any(token in logged for token in hop.tokens)
    assert HEARD_TEXT not in logged
