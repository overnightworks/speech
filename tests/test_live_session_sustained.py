"""A session that asks for sustained speech gets one event per item once voice lasts."""

import asyncio
import json
from itertools import groupby
from typing import TYPE_CHECKING

import pytest
from starlette.status import WS_1012_SERVICE_RESTART

from speech_service.hearing import DRAIN_TURN_SECONDS
from speech_service.hearing_session import DEFAULT_SUSTAINED_SPEECH_MS
from speech_service.voice_activity import WINDOW_SECONDS
from tests.support.clock_support import FakeClock
from tests.support.hear_support import FakeHearing, loaded
from tests.support.live_support import (
    COMPLETED,
    SPEECH_STOPPED,
    SPEECH_SUSTAINED,
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
FOUR_TENTHS = {"duration_ms": 400}
# The ruled arithmetic: 0.4 s is 13 windows of 32 ms, and a pause of 5 windows is
# longer than the 0.15 s bound where 4 are not.
THRESHOLD_WINDOWS = 13
SILENCE_WINDOWS = 16
PARAM = "session.audio.input.sustained_speech.duration_ms"
ITEMS = 4
LEVELS = {"v": 5000, ".": 0}

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


def _end_of_window(index: int) -> int:
    return round((index + 1) * WINDOW_SECONDS * 1000)


async def _asking(
    ask: object = FOUR_TENTHS, engine: FakeHearing | None = None
) -> tuple[LiveSession, FakeSocket, Hearing]:
    clock = FakeClock()
    hearing = loaded(engine or FakeHearing(), clock)
    session, socket = await started(hearing)
    await session.on_message(update(sustained_speech=ask))
    return session, socket, hearing


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
        _, socket, _ = await _asking(ask)
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
    ("ask", "param"),
    [
        pytest.param(None, "session.audio.input.sustained_speech", id="null-ask"),
        pytest.param({"duration_ms": None}, PARAM, id="null-duration"),
        pytest.param({"duration_ms": "400"}, PARAM, id="string"),
        pytest.param({"duration_ms": 400.5}, PARAM, id="float"),
        pytest.param({"duration_ms": True}, PARAM, id="bool"),
        pytest.param({"duration_ms": 0}, PARAM, id="zero"),
        pytest.param({"duration_ms": -1}, PARAM, id="negative"),
        pytest.param({"duration_ms": 30001}, PARAM, id="above-the-turn-cap"),
        pytest.param(
            {"duration_ms": 400, "pause_ms": 100},
            "session.audio.input.sustained_speech.pause_ms",
            id="unknown-key",
        ),
    ],
)
def test_every_bad_value_ends_the_session_invalid_naming_the_field(
    ask: object, param: str
) -> None:
    """WIRE-58, WIRE-172: a bad ask ends the session invalid_request with its param."""

    async def scenario() -> FakeSocket:
        _, socket, _ = await _asking(ask)
        return socket

    socket = asyncio.run(scenario())

    assert end_is(socket, "invalid_request", failed=0)
    assert socket.of_type("error")[0]["error"]["param"] == param
    assert socket.of_type("session.updated") == []


@pytest.mark.parametrize(
    ("pattern", "reached_at"),
    [
        pytest.param(
            "v" * (THRESHOLD_WINDOWS - 1), None, id="shorter-than-the-threshold"
        ),
        pytest.param(
            "v" * THRESHOLD_WINDOWS, THRESHOLD_WINDOWS - 1, id="exactly-the-threshold"
        ),
        pytest.param("v" * 40, THRESHOLD_WINDOWS - 1, id="longer-never-a-second"),
        pytest.param(
            "v" * 8 + "." * 4 + "v" * 8, 8 + 4 + 5 - 1, id="a-short-pause-counts-on"
        ),
        pytest.param("v" * 8 + "." * 5 + "v" * 8, None, id="a-long-pause-restarts"),
        pytest.param(
            "v" * 8 + "." * 5 + "v" * THRESHOLD_WINDOWS,
            8 + 5 + THRESHOLD_WINDOWS - 1,
            id="a-voice-after-a-long-pause-reaches-it-alone",
        ),
    ],
)
def test_one_event_comes_with_the_item_at_the_window_where_the_count_is_reached(
    pattern: str, reached_at: int | None
) -> None:
    """WIRE-171, WIRE-173, WIRE-175: the event names the item and the window's end."""

    async def scenario() -> FakeSocket:
        session, socket, _ = await _asking()
        await session.on_message(append(_voice(pattern)))
        return socket

    socket = asyncio.run(scenario())

    events = _sustained(socket)
    if reached_at is None:
        assert events == []
        return
    (event,) = events
    (opened,) = socket.of_type(STARTED)
    assert event["item_id"] == opened["item_id"]
    assert event["audio_end_ms"] == _end_of_window(reached_at)
    assert event["event_id"]


def test_the_event_lies_after_the_start_and_before_the_stop_and_the_commit() -> None:
    """WIRE-175: the order of one item's events."""

    async def scenario() -> FakeSocket:
        session, socket, _ = await _asking()
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


@pytest.mark.parametrize(
    ("pattern", "duration_ms", "reached_at"),
    [
        pytest.param("vv.vvv", 100, 5, id="a-pre-roll-that-already-holds-the-count"),
        pytest.param("vv.....vvv", 100, None, id="a-long-pause-in-the-pre-roll"),
        pytest.param("vv.....vvvv", 100, 10, id="the-count-since-that-pause-goes-on"),
    ],
)
def test_the_pre_roll_counts_under_the_same_rule_and_its_event_follows_the_start(
    pattern: str, duration_ms: int, reached_at: int | None
) -> None:
    """WIRE-173: pre-roll windows count since the last pause longer than the bound."""

    async def scenario() -> FakeSocket:
        session, socket, _ = await _asking({"duration_ms": duration_ms})
        await session.on_message(append(_voice(pattern)))
        return socket

    socket = asyncio.run(scenario())

    events = _sustained(socket)
    if reached_at is None:
        assert events == []
        return
    assert [event["audio_end_ms"] for event in events] == [_end_of_window(reached_at)]
    types = socket.types()
    assert types.index(SPEECH_SUSTAINED) == types.index(STARTED) + 1


def test_each_item_gets_its_own_event_and_a_voice_that_stopped_first_none() -> None:
    """WIRE-173, WIRE-175: the count starts afresh with every item."""

    async def scenario() -> FakeSocket:
        session, socket, _ = await _asking()
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
        clock = FakeClock()
        hearing = loaded(FakeHearing(), clock)
        session, socket = await started(hearing)
        await session.on_message(update(sustained_speech=FOUR_TENTHS))
        await session.on_message(append(_voice("v" * (THRESHOLD_WINDOWS - 1))))
        if ending == "silence":
            await session.on_message(append(_voice("." * SILENCE_WINDOWS)))
        elif ending == "commit":
            await session.on_message(commit())
        else:
            hearing.begin_drain()
            clock.jump(DRAIN_TURN_SECONDS)
            await session.on_tick()
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    assert _sustained(socket) == []
    assert socket.types().count(STARTED) == 1
    assert socket.types().count(COMPLETED) == 1


@pytest.mark.parametrize("ending", ["commit", "drain-cut"])
def test_an_item_that_ends_after_the_event_keeps_it_and_ends_as_it_does_today(
    ending: str,
) -> None:
    """WIRE-175: the event stands when the item is ended by the client or a drain."""

    async def scenario() -> FakeSocket:
        clock = FakeClock()
        hearing = loaded(FakeHearing(), clock)
        session, socket = await started(hearing)
        await session.on_message(update(sustained_speech=FOUR_TENTHS))
        await session.on_message(append(_voice("v" * (THRESHOLD_WINDOWS + 2))))
        if ending == "commit":
            await session.on_message(commit())
        else:
            hearing.begin_drain()
            clock.jump(DRAIN_TURN_SECONDS)
            await session.on_tick()
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    assert len(_sustained(socket)) == 1
    types = socket.types()
    assert types.index(STARTED) < types.index(SPEECH_SUSTAINED) < types.index(COMMITTED)
    assert types.count(COMPLETED) == 1


def test_the_event_is_sent_to_a_draining_session_until_its_close() -> None:
    """WIRE-175: a drain sends the event while the open item runs, then closes 1012."""

    async def scenario() -> FakeSocket:
        clock = FakeClock()
        hearing = loaded(FakeHearing(), clock)
        session, socket = await started(hearing)
        await session.on_message(update(sustained_speech=FOUR_TENTHS))
        await session.on_message(append(_voice("v" * 8)))
        hearing.begin_drain()
        await session.on_message(append(_voice("v" * 8)))
        clock.jump(DRAIN_TURN_SECONDS)
        await session.on_tick()
        await session.settled()
        await session.on_tick()
        return socket

    socket = asyncio.run(scenario())

    assert len(_sustained(socket)) == 1
    assert socket.close_code == WS_1012_SERVICE_RESTART
    assert socket.types().index(SPEECH_SUSTAINED) < socket.types().index(COMMITTED)


def test_nothing_follows_an_error_not_even_a_sustained_voice() -> None:
    """WIRE-175: after `error` the session sends no event."""

    async def scenario() -> FakeSocket:
        session, socket, _ = await _asking()
        await session.on_message(append(_voice("v" * (THRESHOLD_WINDOWS - 1))))
        await session.on_message(append("not base64!"))
        await session.on_message(append(_voice("v" * 20)))
        return socket

    socket = asyncio.run(scenario())

    assert socket.types()[-1] == "error"
    assert _sustained(socket) == []


def test_the_event_is_sent_while_a_decode_is_held() -> None:
    """WIRE-175: the event waits for no decode."""

    async def scenario() -> int:
        engine = FakeHearing(hold=True)
        session, socket, _ = await _asking(engine=engine)
        await session.on_message(append(_voice("v" * 30)))
        await session.on_tick()
        sent_while_held = len(_sustained(socket))
        engine.release.set()
        await session.settled()
        return sent_while_held

    assert asyncio.run(scenario()) == 1
