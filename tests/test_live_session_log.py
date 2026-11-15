"""A live turn writes a line per step in one trace, and no line holds what was heard."""

import asyncio
import json
from typing import TYPE_CHECKING

import pytest
from logline.testing import configured

from speech_client.wire import RefusalCode
from speech_service.voice_activity import WINDOW_SECONDS
from tests.support.clock_support import FakeClock
from tests.support.hear_support import HEARD_TEXT, FakeHearing, loaded
from tests.support.live_support import (
    WAIT_SECONDS,
    append,
    audio,
    commit_a_turn,
    one_question,
    started,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


# Half a second of speech touches 16 windows.
SPEECH_WINDOWS = 16
WAIT = "turn_end_wait.ended"
HEARING = "hearing.ended"
Line = dict[str, object]


def _lines(capsys: pytest.CaptureFixture[str], *events: str) -> list[Line]:
    written = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    return [line for line in written if line["event"] in events]


def _drive(
    scenario: Callable[[], Awaitable[None]], capsys: pytest.CaptureFixture[str]
) -> list[Line]:
    with configured("speech"):
        asyncio.run(scenario())
    return _lines(capsys, WAIT, HEARING)


async def _one_question_heard(engine: FakeHearing, *, turns: int = 1) -> None:
    session, _ = await started(loaded(engine, FakeClock()))
    for _ in range(turns):
        await one_question(session)
        await session.settled()


def test_a_turn_ended_by_silence_writes_its_wait_and_its_hearing_in_one_trace(
    capsys: pytest.CaptureFixture[str],
) -> None:
    wait, hearing = _drive(lambda: _one_question_heard(FakeHearing()), capsys)

    assert (wait["event"], wait["outcome"], wait["ended_by"]) == (WAIT, "ok", "silence")
    assert (hearing["event"], hearing["outcome"]) == (HEARING, "ok")
    assert wait["item_id"] == hearing["item_id"]
    assert wait["trace_id"] is not None
    assert wait["trace_id"] == hearing["trace_id"]
    assert all(
        isinstance(line["duration_ms"], float) and line["duration_ms"] >= 0
        for line in (wait, hearing)
    )


def test_the_hearing_line_tells_when_the_person_stopped_and_what_silence_ended_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _wait, hearing = _drive(lambda: _one_question_heard(FakeHearing()), capsys)

    assert isinstance(hearing["audio_end_ms"], int)
    assert isinstance(hearing["silence_ms"], int)
    assert hearing["silence_ms"] > 0
    stopped_ms = hearing["audio_end_ms"] - hearing["silence_ms"]
    assert stopped_ms == pytest.approx(SPEECH_WINDOWS * WINDOW_SECONDS * 1000, abs=40)


def test_a_turn_the_client_ended_names_that_and_no_silence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def scenario() -> None:
        session, _ = await started(loaded(FakeHearing(), FakeClock()))
        await commit_a_turn(session)
        await session.settled()

    wait, hearing = _drive(scenario, capsys)

    assert wait["ended_by"] == "client"
    assert hearing["silence_ms"] == 0


def test_every_turn_has_its_own_trace(capsys: pytest.CaptureFixture[str]) -> None:
    lines = _drive(lambda: _one_question_heard(FakeHearing(), turns=2), capsys)

    first_wait, first_hearing, second_wait, second_hearing = lines
    assert first_wait["trace_id"] == first_hearing["trace_id"]
    assert second_wait["trace_id"] == second_hearing["trace_id"]
    assert first_wait["trace_id"] != second_wait["trace_id"]


def test_a_turn_writes_no_line_per_chunk(capsys: pytest.CaptureFixture[str]) -> None:
    async def scenario() -> None:
        session, _ = await started(loaded(FakeHearing(), FakeClock()))
        for _ in range(20):
            await session.on_message(append(audio(0.05)))
        await session.on_message(append(audio(0.8, 0)))
        await session.settled()

    with configured("speech"):
        asyncio.run(scenario())

    written = [line["event"] for line in _lines(capsys, WAIT, HEARING)]
    assert written == [WAIT, HEARING]


def test_a_failed_decode_ends_the_hearing_as_failed_with_the_error_type_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def scenario() -> None:
        engine = FakeHearing()
        engine.fail_transcription = True
        session, socket = await started(loaded(engine, FakeClock()))
        await commit_a_turn(session)
        await socket.until_closed()

    _wait, hearing = _drive(scenario, capsys)

    assert hearing["outcome"] == "failed"
    assert hearing["level"] == "error"
    assert isinstance(hearing["error_type"], str)
    assert "exception" not in hearing


def test_a_hearing_the_client_leaves_ends_as_cancelled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def scenario() -> None:
        engine = FakeHearing(hold=True)
        session, _ = await started(loaded(engine, FakeClock()))
        await one_question(session)
        assert await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        await session.on_disconnect()
        engine.release.set()

    _wait, hearing = _drive(scenario, capsys)

    assert hearing["outcome"] == "cancelled"


def test_a_refused_session_writes_the_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def scenario() -> None:
        session, socket = await started(loaded(FakeHearing(), FakeClock()))
        await session.on_message("not an event")
        await socket.until_closed()

    with configured("speech"):
        asyncio.run(scenario())

    (refused,) = _lines(capsys, "session.refused")
    assert refused["reason"] == RefusalCode.INVALID_REQUEST.value


def test_no_line_holds_the_heard_text(capsys: pytest.CaptureFixture[str]) -> None:
    async def scenario() -> None:
        session, _ = await started(loaded(FakeHearing(), FakeClock()))
        await one_question(session)
        await session.settled()

    with configured("speech"):
        asyncio.run(scenario())

    assert HEARD_TEXT not in capsys.readouterr().out


def test_the_decode_of_a_turn_is_given_its_voiced_seconds() -> None:
    async def scenario() -> FakeHearing:
        engine = FakeHearing()
        session, _ = await started(loaded(engine, FakeClock()))
        await one_question(session)
        await session.settled()
        return engine

    engine = asyncio.run(scenario())

    assert engine.heard_voiced[-1] == pytest.approx(SPEECH_WINDOWS * WINDOW_SECONDS)
