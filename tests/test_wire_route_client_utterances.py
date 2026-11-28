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
from speech_client.moments import SustainMoments
from speech_client.realtime_wire import SessionCreated
from speech_client.relay import HOP_FRAME_BYTES, AppEnded, HopClose
from speech_client.utterances import (
    HearingCallbacks,
    Partial,
    SpeechStart,
    SpeechStop,
    SustainedFrom,
    SustainedVoice,
    TurnEnd,
    Utterance,
    hear_utterances,
)
from speech_client.wire import Language
from tests.support.elapsed_support import well_before
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
# The moment's cases count under 200 ms, seven 32 ms windows, and send frames of a
# size that is not a whole number of milliseconds of the browser's 24 kHz audio.
SUSTAINED_MS = 200
SEVEN_WINDOWS_MS = 224
ODD_FRAME_BYTES = 1002
VOICE_FRAMES = 30


@dataclass
class _Hop:
    """What the app side of one hearing saw."""

    utterances: list[Utterance] = field(default_factory=list)
    calls: list[
        SpeechStart
        | SpeechStop
        | SustainedVoice
        | SustainedFrom
        | Partial
        | Utterance
        | TurnEnd
    ] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    turn_ends: asyncio.Queue[TurnEnd] = field(default_factory=asyncio.Queue)
    outcome: asyncio.Future[UtteranceOutcome] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    ended: asyncio.Event = field(default_factory=asyncio.Event)
    heard: asyncio.Event = field(default_factory=asyncio.Event)


def _app_host(
    service_url: str,
    hop: _Hop,
    *,
    sustained: bool,
    moments: SustainMoments | None = None,
) -> FastAPI:
    """A minimal app: the helper on a route, hearing both languages."""
    host = FastAPI()

    @host.websocket("/hear")
    async def hear(browser: WebSocket) -> None:
        await browser.accept()
        pair = valid_pair()

        async def take(utterance: Utterance) -> None:
            hop.utterances.append(utterance)
            hop.calls.append(utterance)
            hop.heard.set()

        async def take_start(start: SpeechStart) -> None:
            hop.calls.append(start)
            hop.started.set()

        async def note(
            call: SpeechStop | SustainedVoice | SustainedFrom | Partial,
        ) -> None:
            hop.calls.append(call)

        async def take_turn_end(turn: TurnEnd) -> None:
            hop.calls.append(turn)
            hop.turn_ends.put_nowait(turn)

        hop.outcome.set_result(
            await hear_utterances(
                browser,
                service_url,
                token_source=pair.tokens,
                languages=tuple(Language),
                deadline_seconds=WAIT_SECONDS,
                held_limit_bytes=HELD_LIMIT_BYTES,
                sustained_speech_ms=SUSTAINED_MS if moments else None,
                moments=moments,
                ended=hop.ended,
                callbacks=HearingCallbacks(
                    on_utterance=take,
                    on_partial=note,
                    on_speech_started=take_start,
                    on_speech_stopped=note,
                    on_turn_ended=take_turn_end,
                    on_speech_sustained=note if sustained else None,
                    on_speech_sustained_from=note if moments else None,
                ),
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
        token_source=pair.tokens,
        languages=(Language.EN,),
        deadline_seconds=WAIT_SECONDS,
        held_limit_bytes=1,
    ) as session:
        first = await asyncio.wait_for(anext(session.events()), WAIT_SECONDS)
    return isinstance(first, SessionCreated)


@asynccontextmanager
async def _browser_at_service(
    engine: FakeHearing,
    *,
    sustained: bool = False,
    moments: SustainMoments | None = None,
) -> AsyncIterator[tuple[_Hop, str, str]]:
    """The real service, an app hearing through it, and the app's hop."""
    async with serving_over_tcp(hear_runtime(engine)) as service:
        hop = _Hop()
        async with serving_asgi(
            _app_host(service.base_url, hop, sustained=sustained, moments=moments)
        ) as base_url:
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
            outcome = await asyncio.wait_for(hop.outcome, well_before(WAIT_SECONDS))
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
                outcome = await asyncio.wait_for(hop.outcome, well_before(WAIT_SECONDS))
                code = await asyncio.wait_for(
                    close_code(browser), well_before(WAIT_SECONDS)
                )
            freed = await _service_takes_a_session(service_url)
            return hop.utterances, outcome, code, freed

    utterances, outcome, code, freed = asyncio.run(scenario())

    assert utterances == []
    assert outcome == AppEnded()
    assert code == HopClose.NORMAL
    assert freed


def test_speech_started_reaches_the_app_before_the_items_partials_and_utterance() -> (
    None
):
    async def scenario() -> list[
        SpeechStart | SpeechStop | Partial | Utterance | TurnEnd
    ]:
        async with (
            _browser_at_service(FakeHearing("Good day everyone")) as (hop, url, _),
            connect(url) as browser,
        ):
            await _speak_then_fall_silent(browser)
            await asyncio.wait_for(hop.heard.wait(), WAIT_SECONDS)
            hop.ended.set()
            await asyncio.wait_for(hop.outcome, well_before(WAIT_SECONDS))
            return hop.calls

    calls = asyncio.run(scenario())

    starts = [call for call in calls if isinstance(call, SpeechStart)]
    assert len(starts) == 1
    assert calls[0] == starts[0]
    partials = [i for i, call in enumerate(calls) if isinstance(call, Partial)]
    assert all(calls.index(starts[0]) < i for i in partials)
    assert isinstance(calls[-2], Utterance)


def test_sustained_speech_reaches_the_app_once_before_the_utterance() -> None:
    async def scenario() -> list[
        SpeechStart | SpeechStop | SustainedVoice | Partial | Utterance | TurnEnd
    ]:
        async with (
            _browser_at_service(FakeHearing("Good day everyone"), sustained=True) as (
                hop,
                url,
                _,
            ),
            connect(url) as browser,
        ):
            await _speak_then_fall_silent(browser)
            await asyncio.wait_for(hop.heard.wait(), WAIT_SECONDS)
            hop.ended.set()
            await asyncio.wait_for(hop.outcome, well_before(WAIT_SECONDS))
            return hop.calls

    calls = asyncio.run(scenario())

    (voice,) = [call for call in calls if isinstance(call, SustainedVoice)]
    utterance_at = next(
        i for i, call in enumerate(calls) if isinstance(call, Utterance)
    )
    assert calls.index(voice) < utterance_at


def test_turn_ended_reaches_the_app_for_a_turn_with_words_and_for_an_empty_one() -> (
    None
):
    engine = FakeHearing("Good day everyone")

    async def scenario() -> list[
        SpeechStart | SpeechStop | Partial | Utterance | TurnEnd
    ]:
        async with (
            _browser_at_service(engine) as (hop, url, _),
            connect(url) as browser,
        ):
            await _speak_then_fall_silent(browser)
            await asyncio.wait_for(hop.turn_ends.get(), WAIT_SECONDS)
            engine.text = ""
            await _speak_then_fall_silent(browser)
            await asyncio.wait_for(hop.turn_ends.get(), WAIT_SECONDS)
            hop.ended.set()
            await asyncio.wait_for(hop.outcome, well_before(WAIT_SECONDS))
            return hop.calls

    calls = asyncio.run(scenario())

    assert [call for call in calls if isinstance(call, TurnEnd)] == [
        TurnEnd(heard_words=True),
        TurnEnd(heard_words=False),
    ]
    assert len([call for call in calls if isinstance(call, Utterance)]) == 1
    utterance_at = next(
        i for i, call in enumerate(calls) if isinstance(call, Utterance)
    )
    assert calls[utterance_at + 1] == TurnEnd(heard_words=True)


def test_speech_stopped_reaches_the_app_with_the_audio_end_before_the_utterance() -> (
    None
):
    async def scenario() -> tuple[
        list[SpeechStart | SpeechStop | Partial | Utterance | TurnEnd],
        list[Utterance],
    ]:
        async with (
            _browser_at_service(FakeHearing("Good day everyone")) as (hop, url, _),
            connect(url) as browser,
        ):
            await _speak_then_fall_silent(browser)
            await asyncio.wait_for(hop.turn_ends.get(), WAIT_SECONDS)
            hop.ended.set()
            await asyncio.wait_for(hop.outcome, well_before(WAIT_SECONDS))
            return hop.calls, hop.utterances

    calls, utterances = asyncio.run(scenario())

    stops = [call for call in calls if isinstance(call, SpeechStop)]
    assert len(stops) == 1
    assert stops[0].audio_end_ms == utterances[0].audio_end_ms
    stop_at = calls.index(stops[0])
    assert stop_at < calls.index(utterances[0])
    assert stop_at < next(i for i, c in enumerate(calls) if isinstance(c, TurnEnd))


async def _send_voice_frames(browser: ClientConnection) -> None:
    voice = struct.pack("<h", SPEECH_LEVEL) * (ODD_FRAME_BYTES // 2)
    for _ in range(VOICE_FRAMES):
        await browser.send(voice)


@pytest.mark.parametrize(
    "namings", [1, 2], ids=["one-moment", "an-equal-second-moment"]
)
def test_a_moment_gets_its_event_once_counted_from_the_position_now_returned(
    namings: int,
) -> None:
    """WIRE-176 to WIRE-178: the library's position is a moment the service accepts."""
    moments = SustainMoments()

    async def scenario() -> tuple[list[int | None], list[object]]:
        async with (
            _browser_at_service(
                FakeHearing("Good day everyone"), sustained=True, moments=moments
            ) as (hop, url, _),
            connect(url) as browser,
        ):
            await _send_voice_frames(browser)
            await asyncio.wait_for(hop.started.wait(), WAIT_SECONDS)
            positions = [moments.now() for _ in range(namings)]
            await _send_voice_frames(browser)
            await _speak_then_fall_silent(browser)
            await asyncio.wait_for(hop.heard.wait(), WAIT_SECONDS)
            hop.ended.set()
            await asyncio.wait_for(hop.outcome, well_before(WAIT_SECONDS))
            return positions, hop.calls

    positions, calls = asyncio.run(scenario())

    (position,) = set(positions)
    assert position is not None
    (heard,) = [call for call in calls if isinstance(call, SustainedFrom)]
    (voice,) = [call for call in calls if isinstance(call, SustainedVoice)]
    assert heard.item_id == voice.item_id
    # The first window that begins at or after the position starts within 32 ms of it.
    assert SEVEN_WINDOWS_MS <= heard.audio_end_ms - position < SEVEN_WINDOWS_MS + 32
    assert calls.index(voice) < calls.index(heard)
