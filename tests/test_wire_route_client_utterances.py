"""The client's finished-utterance helper against the real service."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI, WebSocket
from websockets.asyncio.client import connect

from speech_client.relay import HOP_FRAME_BYTES, AppEnded, HopClose, RuleBroken
from speech_client.utterances import Utterance, hear_utterances
from speech_client.wire import Language
from tests.support.hear_support import FakeHearing, hear_runtime
from tests.support.tcp_server_support import serving_asgi, serving_over_tcp
from tests.test_wire_route_client_relay import (
    _a_new_session_opens,
    _close_code,
    _tokens,
    _tone,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from websockets.asyncio.client import ClientConnection

    from speech_client.utterances import UtteranceOutcome

WAIT_SECONDS = 10
HELD_LIMIT_BYTES = 1024 * 1024
END_BOUND_SECONDS = 2


@dataclass
class _Hop:
    """What the app side of one hearing saw."""

    utterances: list[Utterance] = field(default_factory=list)
    outcome: asyncio.Future[UtteranceOutcome] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    ended: asyncio.Event = field(default_factory=asyncio.Event)
    heard: asyncio.Event = field(default_factory=asyncio.Event)


def _app_host(service_url: str, hop: _Hop) -> FastAPI:
    """A minimal app: the helper on a route, hearing both languages."""
    host = FastAPI()

    @host.websocket("/hear")
    async def hear(browser: WebSocket) -> None:
        await browser.accept()
        workload_token, person_token = _tokens()

        async def take(utterance: Utterance) -> None:
            hop.utterances.append(utterance)
            hop.heard.set()

        hop.outcome.set_result(
            await hear_utterances(
                browser,
                service_url,
                workload_token=workload_token,
                person_token=person_token,
                languages=tuple(Language),
                deadline_seconds=WAIT_SECONDS,
                held_limit_bytes=HELD_LIMIT_BYTES,
                ended=hop.ended,
                on_utterance=take,
            )
        )

    return host


async def _speak_then_fall_silent(browser: ClientConnection) -> None:
    """Send half a second of speech and a second of silence in hop frames."""
    audio = _tone(0.5) + _tone(1.0, level=0)
    for start in range(0, len(audio), HOP_FRAME_BYTES):
        await browser.send(audio[start : start + HOP_FRAME_BYTES])


@asynccontextmanager
async def _browser_at_service(
    engine: FakeHearing,
) -> AsyncIterator[tuple[_Hop, str, str]]:
    """The real service, an app hearing through it, and the app's hop."""
    async with serving_over_tcp(hear_runtime(engine)) as service:
        hop = _Hop()
        async with serving_asgi(_app_host(service.base_url, hop)) as base_url:
            yield hop, f"ws{base_url.removeprefix('http')}/hear", service.base_url


@pytest.mark.parametrize(
    ("language", "text"),
    [
        pytest.param(Language.DE, "Guten Tag zusammen", id="german"),
        pytest.param(Language.EN, "Good day everyone", id="english"),
    ],
)
def test_a_spoken_utterance_reaches_the_app_with_its_language_and_positions(
    language: Language, text: str
) -> None:
    engine = FakeHearing(text)
    engine.heard_in = language

    async def scenario() -> tuple[list[Utterance], UtteranceOutcome, int]:
        async with (
            _browser_at_service(engine) as (hop, url, _),
            connect(url) as browser,
        ):
            await _speak_then_fall_silent(browser)
            await asyncio.wait_for(hop.heard.wait(), WAIT_SECONDS)
            hop.ended.set()
            outcome = await asyncio.wait_for(hop.outcome, END_BOUND_SECONDS)
            return hop.utterances, outcome, await _close_code(browser)

    utterances, outcome, code = asyncio.run(scenario())

    (utterance,) = utterances
    assert (utterance.text, utterance.language) == (text, language)
    assert utterance.audio_start_ms is not None
    assert utterance.audio_end_ms is not None
    assert utterance.audio_start_ms <= utterance.audio_end_ms
    assert outcome == AppEnded()
    assert code == HopClose.NORMAL


def test_the_apps_end_signal_ends_the_hearing_and_frees_the_service() -> None:
    async def scenario() -> tuple[list[Utterance], UtteranceOutcome, int, bool]:
        async with _browser_at_service(FakeHearing()) as (hop, url, service_url):
            async with connect(url) as browser:
                await browser.send(_tone(0.1))
                hop.ended.set()
                outcome = await asyncio.wait_for(hop.outcome, END_BOUND_SECONDS)
                code = await asyncio.wait_for(_close_code(browser), END_BOUND_SECONDS)
            freed = await _a_new_session_opens(service_url)
            return hop.utterances, outcome, code, freed

    utterances, outcome, code, freed = asyncio.run(scenario())

    assert utterances == []
    assert outcome == AppEnded()
    assert code == HopClose.NORMAL
    assert freed


def test_a_frame_over_the_hop_frame_limit_ends_the_hearing_with_its_close_code() -> (
    None
):
    async def scenario() -> tuple[UtteranceOutcome, int]:
        async with (
            _browser_at_service(FakeHearing()) as (hop, url, _),
            connect(url) as browser,
        ):
            await browser.send(b"\0" * (HOP_FRAME_BYTES + 1))
            outcome = await asyncio.wait_for(hop.outcome, WAIT_SECONDS)
            return outcome, await _close_code(browser)

    outcome, code = asyncio.run(scenario())

    assert outcome == RuleBroken(HopClose.FRAME_TOO_LARGE)
    assert code == HopClose.FRAME_TOO_LARGE
