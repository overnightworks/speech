"""The client's live hearing session against the real service over TCP."""

from __future__ import annotations

import asyncio
import struct
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from speech_client.hearing import (
    Closed,
    HearingSession,
    Refused,
)
from speech_client.outcomes import OverCap, PastDeadline, Unreachable
from speech_client.wire import (
    HearingLanguage,
    RefusalCode,
    SessionCreated,
    SessionUpdated,
    TranscriptCompleted,
    TranscriptLanguage,
)
from tests.support.hear_support import HEARD_TEXT, FakeHearing, hear_runtime
from tests.support.identity_support import valid_pair
from tests.support.live_support import INPUT_RATE, SPEECH_LEVEL
from tests.support.tcp_server_support import free_loopback_port, serving_over_tcp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from speech_client.hearing import HearingOutcome
    from speech_client.wire import ServerEvent

NOT_A_TOKEN = "no"  # noqa: S105 - a scenario value
DEADLINE_SECONDS = 10.0
HELD_LIMIT_BYTES = 1024 * 1024


def _tone(seconds: float, level: int = SPEECH_LEVEL) -> bytes:
    return struct.pack("<h", level) * int(seconds * INPUT_RATE)


def _session(
    base_url: str,
    *,
    workload_token: str | None = None,
    deadline_seconds: float = DEADLINE_SECONDS,
    held_limit_bytes: int = HELD_LIMIT_BYTES,
) -> HearingSession:
    pair = valid_pair()
    return HearingSession(
        base_url,
        workload_token=workload_token or pair.workload,
        person_token=pair.person,
        languages=(HearingLanguage.EN,),
        deadline_seconds=deadline_seconds,
        held_limit_bytes=held_limit_bytes,
    )


@asynccontextmanager
async def _serving_session(**options: str | float) -> AsyncIterator[HearingSession]:
    """A session open against the real service over TCP."""
    async with (
        serving_over_tcp(hear_runtime(FakeHearing())) as server,
        _session(server.base_url, **options) as session,
    ):
        yield session


async def _until_completed(session: HearingSession) -> list[ServerEvent]:
    seen: list[ServerEvent] = []
    async for event in session.events():
        seen.append(event)
        if isinstance(event, TranscriptCompleted):
            break
    return seen


async def _ended(session: HearingSession) -> HearingOutcome | None:
    async for _ in session.events():
        pass
    return session.outcome


def test_audio_sent_before_the_connection_is_open_is_heard_in_one_turn() -> None:
    """WIRE-48, WIRE-65: held audio reaches the service and a turn comes back."""

    async def scenario() -> list[ServerEvent]:
        async with _serving_session() as session:
            session.send(_tone(0.5))
            session.send(_tone(1.0, level=0))
            return await asyncio.wait_for(_until_completed(session), 10)

    events = asyncio.run(scenario())

    assert isinstance(events[0], SessionCreated)
    assert events[-1].transcript == HEARD_TEXT


def test_a_session_of_one_language_sends_the_set_and_hears_that_language() -> None:
    """WIRE-57, WIRE-136: the client always sends `languages`; completed names one."""

    async def scenario() -> list[ServerEvent]:
        async with _serving_session() as session:
            session.send(_tone(0.5))
            session.send(_tone(1.0, level=0))
            return await asyncio.wait_for(_until_completed(session), 10)

    events = asyncio.run(scenario())

    (updated,) = [event for event in events if isinstance(event, SessionUpdated)]
    setting = updated.session.audio.input.transcription
    assert (setting.language, setting.languages) == (None, (HearingLanguage.EN,))
    assert isinstance(events[-1], TranscriptCompleted)
    assert events[-1].languages == (TranscriptLanguage(code=HearingLanguage.EN),)


def test_leaving_the_context_ends_the_session_closed() -> None:
    async def scenario() -> HearingOutcome | None:
        async with _serving_session() as session:
            await anext(session.events())
        return session.outcome

    assert asyncio.run(scenario()) == Closed()


def test_audio_exactly_the_held_limit_is_held_and_the_session_goes_on() -> None:
    async def scenario() -> ServerEvent:
        async with _serving_session(held_limit_bytes=4) as session:
            session.send(b"\0\0\0\0")
            return await asyncio.wait_for(anext(session.events()), 10)

    assert isinstance(asyncio.run(scenario()), SessionCreated)


def test_audio_one_byte_over_the_held_limit_ends_the_session_over_the_cap() -> None:
    async def scenario() -> HearingOutcome | None:
        async with _serving_session(held_limit_bytes=4) as session:
            session.send(b"\0\0\0\0")
            session.send(b"\0")
            return await asyncio.wait_for(_ended(session), 10)

    assert asyncio.run(scenario()) == OverCap()


def test_a_refused_upgrade_gives_the_status_and_the_code() -> None:
    """WIRE-51: the HTTP refusal before the session is the outcome."""

    async def scenario() -> HearingOutcome | None:
        async with _serving_session(workload_token=NOT_A_TOKEN) as session:
            return await asyncio.wait_for(_ended(session), 10)

    outcome = asyncio.run(scenario())

    assert isinstance(outcome, Refused)
    assert outcome.code == RefusalCode.UNAUTHORIZED


def test_a_deadline_that_passes_in_an_idle_session_ends_it_past_the_deadline() -> None:
    async def scenario() -> HearingOutcome | None:
        async with _serving_session(deadline_seconds=0.3) as session:
            return await asyncio.wait_for(_ended(session), 10)

    assert asyncio.run(scenario()) == PastDeadline()


def test_a_port_nothing_listens_on_is_unreachable() -> None:
    async def scenario() -> HearingOutcome | None:
        base_url = f"http://127.0.0.1:{free_loopback_port()}"
        async with _session(base_url) as session:
            return await asyncio.wait_for(_ended(session), 10)

    assert asyncio.run(scenario()) == Unreachable()
