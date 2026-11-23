"""A session that asks for sounds gets one event per sound; no other session does."""

import asyncio
import json
from typing import TYPE_CHECKING

import pytest
from logline.testing import configured
from starlette.status import WS_1012_SERVICE_RESTART

from tests.support.clock_support import FakeClock
from tests.support.hear_support import FakeHearing, loaded
from tests.support.live_support import (
    ASK_FOR_SOUNDS,
    COMPLETED,
    SOUND_DETECTED,
    FakeSocket,
    append,
    audio,
    end_is,
    one_question,
    started,
    update,
)
from tests.support.sound_support import (
    WAIT_SECONDS,
    FakeSoundModel,
    loaded_taggers,
)

if TYPE_CHECKING:
    from pathlib import Path

    from speech_service.hearing import Hearing
    from speech_service.live_session import LiveSession

# Two seconds and a little more: the first hop is due, the second is not.
FIRST_HOP = audio(2.2, 0)
LAUGH = {"Laughter": 0.4}
SECOND_HOP_START_MS = 500


async def _asking(
    tmp_path: Path,
    *,
    ask: bool = True,
    hold: bool = False,
) -> tuple[LiveSession, FakeSocket, FakeSoundModel, Hearing]:
    taggers, model = loaded_taggers(tmp_path)
    hearing = loaded(FakeHearing(), FakeClock(), sounds=taggers)
    model.windows.clear()
    if hold:
        model.hold_runs()
    session, socket = await started(hearing)
    if ask:
        await session.on_message(update(sound_detection=ASK_FOR_SOUNDS))
    return session, socket, model, hearing


async def _in_flight(model: FakeSoundModel) -> None:
    await asyncio.to_thread(model.entered.wait, WAIT_SECONDS)


def _sound_events(socket: FakeSocket) -> list[dict[str, object]]:
    return socket.of_type(SOUND_DETECTED)


def test_a_session_that_asks_is_echoed_the_ask_and_gets_one_event_per_sound(
    tmp_path: Path,
) -> None:
    """WIRE-163, WIRE-167: a sound is one event with its window's bounds."""

    async def scenario() -> FakeSocket:
        session, socket, model, hearing = await _asking(tmp_path)
        model.script(LAUGH)
        await session.on_message(append(FIRST_HOP))
        await session.settled()
        hearing.close()
        return socket

    socket = asyncio.run(scenario())

    (echo,) = socket.of_type("session.updated")
    assert echo["session"]["audio"]["input"]["sound_detection"] == ASK_FOR_SOUNDS
    (heard,) = _sound_events(socket)
    assert heard["sound"] == "Laughter"
    assert (heard["audio_start_ms"], heard["audio_end_ms"]) == (0, 2000)
    assert "item_id" not in heard
    assert heard["event_id"]


def test_a_session_that_does_not_ask_is_scored_nowhere_and_echoed_no_field(
    tmp_path: Path,
) -> None:
    """WIRE-164, WIRE-165: a session that does not ask is untouched."""

    async def scenario() -> tuple[FakeSocket, FakeSoundModel]:
        session, socket, model, hearing = await _asking(tmp_path, ask=False)
        model.script(LAUGH)
        await session.on_message(update("en"))
        await session.on_message(append(FIRST_HOP))
        await one_question(session)
        await session.settled()
        hearing.close()
        return socket, model

    socket, model = asyncio.run(scenario())

    assert "sound_detection" not in json.dumps(socket.of_type("session.updated"))
    assert _sound_events(socket) == []
    assert model.windows == []
    assert COMPLETED in socket.types()


@pytest.mark.parametrize("ask", [True, False], ids=["asked", "not-asked"])
def test_a_hearing_without_sounds_echoes_no_field_hears_on_and_sends_no_event(
    *, ask: bool
) -> None:
    """WIRE-165: without sounds an ask is echoed without its field."""

    async def scenario() -> FakeSocket:
        session, socket = await started(loaded(FakeHearing(), FakeClock()))
        sound_detection = ASK_FOR_SOUNDS if ask else None
        await session.on_message(update(sound_detection=sound_detection))
        await session.on_message(append(FIRST_HOP))
        await one_question(session)
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    assert "sound_detection" not in json.dumps(socket.of_type("session.updated"))
    assert _sound_events(socket) == []
    assert COMPLETED in socket.types()


@pytest.mark.parametrize(
    "ask",
    [{"type": "other"}, {}],
    ids=["other-type", "no-type"],
)
def test_an_ask_of_another_kind_ends_the_session_invalid(
    tmp_path: Path, ask: dict[str, str]
) -> None:
    """WIRE-58, WIRE-164: an ask of another kind is refused."""

    async def scenario() -> FakeSocket:
        session, socket, _, hearing = await _asking(tmp_path, ask=False)
        await session.on_message(update(sound_detection=ask))
        hearing.close()
        return socket

    socket = asyncio.run(scenario())

    assert end_is(socket, "invalid_request", failed=0)
    assert socket.of_type("error")[0]["error"]["param"].endswith("sound_detection.type")


def test_a_run_that_raises_ends_the_session_unavailable_with_no_sound_after(
    tmp_path: Path,
) -> None:
    """WIRE-168: a failed score ends the session unavailable."""

    async def scenario() -> FakeSocket:
        session, socket, model, hearing = await _asking(tmp_path)
        model.failure = RuntimeError("boom")
        await session.on_message(append(audio(2.2)))
        await session.settled()
        await session.on_message(append(audio(1)))
        await session.settled()
        hearing.close()
        return socket

    socket = asyncio.run(scenario())

    assert end_is(socket, "unavailable", failed=1)
    assert _sound_events(socket) == []


def test_a_hop_that_falls_due_in_flight_is_scored_when_the_run_returns(
    tmp_path: Path,
) -> None:
    """WIRE-167: the latest hop that fell due while one was in flight is scored."""

    async def scenario() -> int:
        session, _, model, hearing = await _asking(tmp_path, hold=True)
        await session.on_message(append(FIRST_HOP))
        await _in_flight(model)
        await session.on_message(append(audio(0.6)))
        model.release.set()
        await session.settled()
        hearing.close()
        return len(model.windows)

    assert asyncio.run(scenario()) > 1


def test_a_refused_submit_ends_the_session_unavailable(tmp_path: Path) -> None:
    """WIRE-168: a refused submit ends the session unavailable."""

    async def scenario() -> FakeSocket:
        taggers, _ = loaded_taggers(tmp_path)
        session, socket = await started(
            loaded(FakeHearing(), FakeClock(), sounds=taggers)
        )
        await session.on_message(update(sound_detection=ASK_FOR_SOUNDS))
        taggers.close()
        await session.on_message(append(audio(2.2)))
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    assert end_is(socket, "unavailable", failed=1)


def test_a_hop_in_flight_when_the_session_ends_by_error_sends_nothing(
    tmp_path: Path,
) -> None:
    """WIRE-168: no sound event follows an error."""

    async def scenario() -> FakeSocket:
        session, socket, model, hearing = await _asking(tmp_path, hold=True)
        model.script(LAUGH)
        await session.on_message(append(FIRST_HOP))
        await _in_flight(model)
        await session.on_message("not json")
        model.release.set()
        await session.settled()
        hearing.close()
        return socket

    socket = asyncio.run(scenario())

    assert end_is(socket, "invalid_request", failed=0)
    assert _sound_events(socket) == []


def test_a_hop_in_flight_when_the_client_leaves_sends_nothing(tmp_path: Path) -> None:
    """WIRE-168: a client that leaves gets nothing."""

    async def scenario() -> FakeSocket:
        session, socket, model, hearing = await _asking(tmp_path, hold=True)
        model.script(LAUGH)
        await session.on_message(append(FIRST_HOP))
        await _in_flight(model)
        await session.on_disconnect()
        model.release.set()
        hearing.close()
        return socket

    socket = asyncio.run(scenario())

    assert socket.types() == ["session.created", "session.updated"]


def test_sounds_are_sent_during_a_drain_and_the_close_is_the_turns(
    tmp_path: Path,
) -> None:
    """WIRE-146, WIRE-168: sounds are sent during a drain until its close."""

    async def scenario() -> FakeSocket:
        session, socket, model, hearing = await _asking(tmp_path)
        model.script(LAUGH)
        await session.on_message(append(audio(0.5)))
        hearing.begin_drain()
        await session.on_message(append(audio(2.2, 0)))
        await session.settled()
        await session.on_tick()
        hearing.close()
        return socket

    socket = asyncio.run(scenario())

    assert [event["sound"] for event in _sound_events(socket)] == ["Laughter"]
    assert COMPLETED in socket.types()
    assert "error" not in socket.types()
    assert socket.close_code == WS_1012_SERVICE_RESTART


def test_a_hop_in_flight_at_the_drains_close_sends_nothing(tmp_path: Path) -> None:
    """WIRE-146, WIRE-168: sounds are sent during a drain until its close."""

    async def scenario() -> FakeSocket:
        session, socket, model, hearing = await _asking(tmp_path, hold=True)
        model.script(LAUGH)
        await session.on_message(append(audio(0.5)))
        hearing.begin_drain()
        await session.on_message(append(audio(2.2, 0)))
        await _in_flight(model)
        await session.on_tick()
        while not socket.closed:
            await session.on_tick()
            await asyncio.sleep(0)
        model.release.set()
        await session.settled()
        hearing.close()
        return socket

    socket = asyncio.run(scenario())

    assert socket.close_code == WS_1012_SERVICE_RESTART
    assert _sound_events(socket) == []


def test_each_sound_event_is_one_log_line_and_a_quiet_hop_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """WIRE-167: each sound event is one log line, and a quiet window none."""

    async def scenario() -> None:
        session, _, model, hearing = await _asking(tmp_path)
        model.script({}, LAUGH)
        await session.on_message(append(FIRST_HOP))
        await session.settled()
        await session.on_message(append(audio(0.5, 0)))
        await session.settled()
        hearing.close()

    with configured("speech"):
        asyncio.run(scenario())

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (detected,) = [line for line in lines if line["event"] == "sound.detected"]
    assert detected["sound"] == "Laughter"
    assert detected["audio_start_ms"] == SECOND_HOP_START_MS
    assert detected["score"] == pytest.approx(0.4)


def test_a_failed_run_is_logged_by_its_type_and_never_by_its_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """WIRE-168: a failed score ends the session unavailable."""

    async def scenario() -> None:
        session, _, model, hearing = await _asking(tmp_path)
        model.failure = RuntimeError("heard something private")
        await session.on_message(append(audio(2.2)))
        await session.settled()
        hearing.close()

    with configured("speech"):
        asyncio.run(scenario())

    out = capsys.readouterr().out
    (failed,) = [
        json.loads(line)
        for line in out.splitlines()
        if json.loads(line)["event"] == "sound.failed"
    ]
    assert failed["error_type"] == "RuntimeError"
    assert "heard something private" not in out
