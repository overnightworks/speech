"""A session that asks for sustained speech gets one event per item once voice lasts."""

import asyncio
import json
from itertools import groupby
from typing import TYPE_CHECKING

import pytest
from starlette.status import WS_1012_SERVICE_RESTART

from speech_service.hearing import DRAIN_TURN_SECONDS
from speech_service.hearing_session import DEFAULT_SUSTAINED_SPEECH_MS
from tests.support.clock_support import FakeClock
from tests.support.detector_support import (
    FOUR_TENTHS_MS,
    THRESHOLD_WINDOWS,
    end_of_window,
)
from tests.support.hear_support import FakeHearing, loaded
from tests.support.live_support import (
    COMPLETED,
    SPEECH_LEVEL,
    SPEECH_STOPPED,
    SPEECH_SUSTAINED,
    WAIT_SECONDS,
    FakeSocket,
    append,
    commit,
    end_is,
    started,
    update,
    windows,
)

STARTED = "input_audio_buffer.speech_started"
COMMITTED = "input_audio_buffer.committed"
FOUR_TENTHS = {"duration_ms": FOUR_TENTHS_MS}
SILENCE_WINDOWS = 16
PARAM = "session.audio.input.sustained_speech.duration_ms"
ITEMS = 4
LEVELS = {"v": SPEECH_LEVEL, ".": 0}

if TYPE_CHECKING:
    from speech_service.hearing import Hearing
    from speech_service.live_session import LiveSession


def _voice(pattern: str) -> str:
    """The audio of `pattern`, one window a character: `v` voiced, `.` silent.

    A silent window follows: the resampler holds back the last samples it is fed.
    """
    return "".join(
        windows(len(list(run)), LEVELS[character])
        for character, run in groupby(pattern + ".")
    )


async def _asking(
    ask: object = FOUR_TENTHS, engine: FakeHearing | None = None
) -> tuple[LiveSession, FakeSocket, Hearing, FakeClock]:
    clock = FakeClock()
    hearing = loaded(engine or FakeHearing(), clock)
    session, socket = await started(hearing)
    await session.on_message(update(sustained_speech=ask))
    return session, socket, hearing, clock


async def _end_by(
    ending: str, session: LiveSession, hearing: Hearing, clock: FakeClock
) -> None:
    """End the open item by `ending`; an unknown ending raises KeyError."""

    async def drain_cut() -> None:
        hearing.begin_drain()
        clock.jump(DRAIN_TURN_SECONDS)
        await session.on_tick()

    endings = {
        "silence": lambda: session.on_message(append(_voice("." * SILENCE_WINDOWS))),
        "commit": lambda: session.on_message(commit()),
        "drain-cut": drain_cut,
    }
    await endings[ending]()


def _sustained(socket: FakeSocket) -> list[dict[str, object]]:
    return socket.of_type(SPEECH_SUSTAINED)


@pytest.mark.parametrize(
    ("ask", "applied"),
    [
        pytest.param({}, DEFAULT_SUSTAINED_SPEECH_MS, id="default"),
        pytest.param({"duration_ms": 250}, 250, id="as-asked"),
        pytest.param({"duration_ms": 30000}, 30000, id="the-turn-cap"),
    ],
)
def test_the_echo_carries_the_applied_duration_and_the_creation_none(
    ask: dict[str, int], applied: int
) -> None:
    """WIRE-172, WIRE-174: `session.updated` echoes the duration the session applies."""

    async def scenario() -> FakeSocket:
        _, socket, _, _ = await _asking(ask)
        return socket

    socket = asyncio.run(scenario())

    (echo,) = socket.of_type("session.updated")
    assert echo["session"]["audio"]["input"]["sustained_speech"] == {
        "duration_ms": applied
    }
    assert "sustained_speech" not in json.dumps(socket.of_type("session.created"))


def test_a_session_that_does_not_ask_sees_no_field_and_no_event() -> None:
    """WIRE-172: without the ask every event and echo is as it was."""

    async def scenario() -> FakeSocket:
        hearing = loaded(FakeHearing(), FakeClock())
        session, socket = await started(hearing)
        await session.on_message(update())
        await session.on_message(append(_voice("v" * 40 + "." * SILENCE_WINDOWS)))
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    assert "sustained" not in json.dumps(socket.events)
    assert socket.types() == [
        "session.created",
        "session.updated",
        STARTED,
        SPEECH_STOPPED,
        COMMITTED,
        "conversation.item.created",
        COMPLETED,
    ]


@pytest.mark.parametrize(
    "ask",
    [
        pytest.param({"duration_ms": "400"}, id="string"),
        pytest.param({"duration_ms": 30001}, id="above-the-turn-cap"),
    ],
)
def test_a_bad_ask_ends_the_session_invalid_naming_the_field(ask: object) -> None:
    """WIRE-58, WIRE-172: a bad ask ends the session invalid_request with its param."""

    async def scenario() -> FakeSocket:
        _, socket, _, _ = await _asking(ask)
        return socket

    socket = asyncio.run(scenario())

    assert end_is(socket, "invalid_request", failed=0)
    assert socket.of_type("error")[0]["error"]["param"] == PARAM
    assert socket.of_type("session.updated") == []


def test_the_event_names_its_item_and_the_end_of_the_window_of_the_count() -> None:
    """WIRE-171, WIRE-173, WIRE-175: the event names the item and the window's end."""

    async def scenario() -> FakeSocket:
        session, socket, _, _ = await _asking()
        await session.on_message(append(_voice("v" * THRESHOLD_WINDOWS)))
        return socket

    socket = asyncio.run(scenario())

    (event,) = _sustained(socket)
    (opened,) = socket.of_type(STARTED)
    assert event["item_id"] == opened["item_id"]
    assert event["audio_end_ms"] == end_of_window(THRESHOLD_WINDOWS - 1)
    assert event["event_id"]


def test_the_event_lies_after_the_start_and_before_the_stop_and_the_commit() -> None:
    """WIRE-175: the order of one item's events."""

    async def scenario() -> FakeSocket:
        session, socket, _, _ = await _asking()
        await session.on_message(append(_voice("v" * 20 + "." * SILENCE_WINDOWS)))
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    types = socket.types()
    assert types[:5] == [
        "session.created",
        "session.updated",
        STARTED,
        SPEECH_SUSTAINED,
        SPEECH_STOPPED,
    ]
    assert types.index(SPEECH_SUSTAINED) < types.index(COMMITTED)
    assert types.count(COMPLETED) == 1


def test_a_pre_roll_that_holds_the_count_has_its_event_right_after_the_start() -> None:
    """WIRE-173: the pre-roll counts, and its event follows `speech_started`."""

    async def scenario() -> FakeSocket:
        session, socket, _, _ = await _asking({"duration_ms": 100})
        await session.on_message(append(_voice("vv.vvv")))
        return socket

    socket = asyncio.run(scenario())

    (event,) = _sustained(socket)
    assert event["audio_end_ms"] == end_of_window(5)
    types = socket.types()
    assert types.index(SPEECH_SUSTAINED) == types.index(STARTED) + 1


def test_each_item_gets_its_own_event_and_a_voice_that_stopped_first_none() -> None:
    """WIRE-173, WIRE-175: the count starts afresh with every item."""

    async def scenario() -> FakeSocket:
        session, socket, _, _ = await _asking()
        short = "v" * 8 + "." * SILENCE_WINDOWS
        await session.on_message(append(_voice(short + short)))
        await session.on_message(append(_voice("v" * 20 + "." * SILENCE_WINDOWS)))
        await session.on_message(append(_voice("v" * 20 + "." * SILENCE_WINDOWS)))
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    events = _sustained(socket)
    items = [event["item_id"] for event in socket.of_type(STARTED)]
    assert len(items) == ITEMS
    assert [event["item_id"] for event in events] == items[2:]
    assert events[0]["audio_end_ms"] < events[1]["audio_end_ms"]


@pytest.mark.parametrize("ending", ["silence", "commit", "drain-cut"])
def test_an_item_that_ends_before_the_count_has_no_event_and_ends_once(
    ending: str,
) -> None:
    """WIRE-80, WIRE-173: no event for an item that ended first, which still ends."""

    async def scenario() -> FakeSocket:
        session, socket, hearing, clock = await _asking()
        await session.on_message(append(_voice("v" * (THRESHOLD_WINDOWS - 1))))
        await _end_by(ending, session, hearing, clock)
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    assert _sustained(socket) == []
    assert socket.types().count(STARTED) == 1
    assert socket.types().count(COMPLETED) == 1


@pytest.mark.parametrize("ending", ["commit", "drain-cut"])
def test_an_item_that_ends_after_the_event_keeps_it_and_ends_once(
    ending: str,
) -> None:
    """WIRE-175: the event stands when the item is ended by the client or a drain."""

    async def scenario() -> FakeSocket:
        session, socket, hearing, clock = await _asking()
        await session.on_message(append(_voice("v" * (THRESHOLD_WINDOWS + 2))))
        await _end_by(ending, session, hearing, clock)
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    assert len(_sustained(socket)) == 1
    types = socket.types()
    assert types.index(STARTED) < types.index(SPEECH_SUSTAINED) < types.index(COMMITTED)
    assert types.count(COMPLETED) == 1


def test_the_event_is_sent_to_a_draining_session_until_its_close() -> None:
    """WIRE-175: a drain sends the event while the open item runs, then closes 1012."""

    async def scenario() -> tuple[FakeSocket, list[object]]:
        session, socket, hearing, clock = await _asking()
        await session.on_message(append(_voice("v" * 8)))
        hearing.begin_drain()
        await session.on_message(append(_voice("v" * 8)))
        clock.jump(DRAIN_TURN_SECONDS)
        await session.on_tick()
        await session.settled()
        await session.on_tick()
        closed_with = socket.types()
        await session.on_message(append(_voice("v" * (2 * THRESHOLD_WINDOWS))))
        return socket, closed_with

    socket, closed_with = asyncio.run(scenario())

    assert socket.types() == closed_with
    assert len(_sustained(socket)) == 1
    assert socket.close_code == WS_1012_SERVICE_RESTART
    assert socket.types().index(SPEECH_SUSTAINED) < socket.types().index(COMMITTED)


def test_nothing_follows_an_error_not_even_a_sustained_voice() -> None:
    """WIRE-175: after `error` the session sends no event."""

    async def scenario() -> FakeSocket:
        session, socket, _, _ = await _asking()
        await session.on_message(append(_voice("v" * (THRESHOLD_WINDOWS - 1))))
        await session.on_message(append("not base64!"))
        await session.on_message(append(_voice("v" * 20)))
        return socket

    socket = asyncio.run(scenario())

    assert socket.types()[-1] == "error"
    assert _sustained(socket) == []


def test_the_event_is_sent_while_a_decode_is_held() -> None:
    """WIRE-175: the event waits for no decode."""
    voiced_before_the_decode = 15

    async def scenario() -> tuple[list[object], list[object]]:
        engine = FakeHearing(hold=True)
        session, socket, _, _ = await _asking({"duration_ms": 800}, engine)
        await session.on_message(append(_voice("v" * voiced_before_the_decode)))
        await session.on_tick()
        assert await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        before = _sustained(socket)
        await session.on_message(append(_voice("v" * voiced_before_the_decode)))
        during = _sustained(socket)
        engine.release.set()
        await session.settled()
        return before, during

    before, during = asyncio.run(scenario())

    assert before == []
    assert len(during) == 1
