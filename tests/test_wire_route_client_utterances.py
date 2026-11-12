"""The client's finished-utterance helper against the real service."""

from __future__ import annotations

import asyncio
import struct
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI, WebSocket
from websockets.asyncio.client import connect

from speech_client.hearing import HearingSession
from speech_client.relay import HOP_FRAME_BYTES, AppEnded, HopClose
from speech_client.utterances import Utterance, hear_utterances
from speech_client.wire import Language, SessionCreated
from tests.support.hear_support import FakeHearing, hear_runtime
from tests.support.identity_support import valid_pair
from tests.support.live_support import INPUT_RATE, SPEECH_LEVEL
from tests.support.tcp_server_support import close_code, serving_asgi, serving_over_tcp

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
        pair = valid_pair()

        async def take(utterance: Utterance) -> None:
            hop.utterances.append(utterance)
            hop.heard.set()

        hop.outcome.set_result(
            await hear_utterances(
                browser,
                service_url,
                workload_token=pair.workload,
                person_token=pair.person,
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
    audio = struct.pack("<h", SPEECH_LEVEL) * (INPUT_RATE // 2) + bytes(INPUT_RATE * 2)
    for start in range(0, len(audio), HOP_FRAME_BYTES):
        await browser.send(audio[start : start + HOP_FRAME_BYTES])


async def _service_takes_a_session(service_url: str) -> bool:
    """Whether the service's one place is free: a new session is opened."""
    pair = valid_pair()
    async with HearingSession(
        service_url,
        workload_token=pair.workload,
        person_token=pair.person,
        languages=(Language.EN,),
        deadline_seconds=WAIT_SECONDS,
        held_limit_bytes=1,
    ) as session:
        first = await asyncio.wait_for(anext(session.events()), WAIT_SECONDS)
    return isinstance(first, SessionCreated)


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
            return hop.utterances, outcome, await close_code(browser)

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
                await browser.send(bytes(HOP_FRAME_BYTES))
                hop.ended.set()
                outcome = await asyncio.wait_for(hop.outcome, END_BOUND_SECONDS)
                code = await asyncio.wait_for(close_code(browser), END_BOUND_SECONDS)
            freed = await _service_takes_a_session(service_url)
            return hop.utterances, outcome, code, freed

    utterances, outcome, code, freed = asyncio.run(scenario())

    assert utterances == []
    assert outcome == AppEnded()
    assert code == HopClose.NORMAL
    assert freed
