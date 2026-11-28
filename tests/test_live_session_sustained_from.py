"""A session that asked for sustained speech names a moment and hears from it."""

import asyncio

import pytest
from starlette.status import WS_1012_SERVICE_RESTART

from speech_client.realtime_wire import (
    AudioCommitted,
    ErrorEvent,
    SpeechStarted,
    SpeechStopped,
    SpeechSustained,
    SpeechSustainedFrom,
)
from speech_service.hearing import DRAIN_TURN_SECONDS
from tests.support.clock_support import FakeClock
from tests.support.detector_support import end_of_window, start_of_window
from tests.support.hear_support import FakeHearing, loaded
from tests.support.live_support import (
    FakeSocket,
    append,
    end_is,
    samples,
    started,
    sustain_from,
    update,
    windows,
)
from tests.support.sustained_support import asking, voice

# Seven windows of voice are the duration the cases count under.
SEVEN_WINDOWS = {"duration_ms": 200}
SILENCE_WINDOWS = 16
FROM_PARAM = "from_ms"
EVENT_ID = "m1"


def _moments(socket: FakeSocket) -> list[SpeechSustainedFrom]:
    return socket.of_type(SpeechSustainedFrom)


def _refusal(socket: FakeSocket, failed: int = 0) -> tuple[str | None, str | None]:
    """The `param` and the `event_id` of the one `error` that ended the session."""
    assert end_is(socket, "invalid_request", failed=failed)
    (error,) = socket.of_type(ErrorEvent)
    return error.error.param, error.error.event_id


def test_the_moments_event_follows_the_items_own_and_names_its_item_and_moment() -> (
    None
):
    """WIRE-177: the event carries the item, the moment as named, the window's end."""

    async def scenario() -> FakeSocket:
        session, socket, _, _ = await asking(SEVEN_WINDOWS)
        await session.on_message(sustain_from(from_ms=start_of_window(4)))
        await session.on_message(append(voice("v" * 20 + "." * SILENCE_WINDOWS)))
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    (event,) = _moments(socket)
    (opened,) = socket.of_type(SpeechStarted)
    (sustained,) = socket.of_type(SpeechSustained)
    assert event.item_id == opened.item_id
    assert event.from_ms == start_of_window(4)
    assert event.audio_end_ms == end_of_window(10)
    assert sustained.audio_end_ms == end_of_window(6)
    types = socket.types()
    assert (
        types.index(SpeechStarted)
        < types.index(SpeechSustained)
        < types.index(SpeechSustainedFrom)
        < types.index(SpeechStopped)
        < types.index(AudioCommitted)
    )


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"from_ms": 0}, id="a-valid-moment"),
        pytest.param({"from_ms": -1}, id="a-negative-position"),
        pytest.param({"from_ms": "0"}, id="a-text-position"),
        pytest.param({}, id="no-position"),
        pytest.param({"from_ms": 0, "extra": 1}, id="a-key-the-service-does-not-know"),
    ],
)
@pytest.mark.parametrize("update_sent", [True, False], ids=["updated", "no-update"])
def test_a_session_that_did_not_ask_refuses_the_type_whatever_its_fields(
    fields: dict[str, object], *, update_sent: bool
) -> None:
    """WIRE-61, WIRE-176: the ask is judged before the fields."""

    async def scenario() -> FakeSocket:
        session, socket = await started(loaded(FakeHearing(), FakeClock()))
        if update_sent:
            await session.on_message(update())
        await session.on_message(sustain_from(**fields, event_id=EVENT_ID))
        return socket

    assert _refusal(asyncio.run(scenario())) == ("type", EVENT_ID)


@pytest.mark.parametrize(
    ("fields", "param"),
    [
        pytest.param({}, FROM_PARAM, id="no-position"),
        pytest.param({"from_ms": 320, "extra": 1}, "extra", id="an-unknown-key"),
    ],
)
def test_a_malformed_moment_ends_the_session_naming_the_field(
    fields: dict[str, object], param: str
) -> None:
    """WIRE-176, WIRE-82: the parser's refusal, field and id, ends the session."""

    async def scenario() -> FakeSocket:
        session, socket, _, _ = await asking(SEVEN_WINDOWS)
        await session.on_message(sustain_from(**fields, event_id=EVENT_ID))
        return socket

    assert _refusal(asyncio.run(scenario())) == (param, EVENT_ID)


@pytest.mark.parametrize(
    ("appended", "heard_ms"),
    [
        pytest.param(windows(10, 0), 320, id="whole-windows"),
        pytest.param(
            samples(1001, 0), 41, id="a-fraction-of-a-millisecond-rounds-down"
        ),
    ],
)
@pytest.mark.parametrize(
    ("behind_by", "refused"),
    [
        pytest.param(1, True, id="behind-by-one-millisecond"),
        pytest.param(0, False, id="at-the-audio-appended"),
    ],
)
def test_a_moment_below_the_whole_milliseconds_appended_ends_the_session(
    appended: str, heard_ms: int, behind_by: int, *, refused: bool
) -> None:
    """WIRE-176: behind counts the appended bytes, rounded down, not the resampler."""

    async def scenario() -> FakeSocket:
        session, socket, _, _ = await asking(SEVEN_WINDOWS)
        await session.on_message(append(appended))
        await session.on_message(sustain_from(from_ms=heard_ms - behind_by))
        return socket

    socket = asyncio.run(scenario())

    if refused:
        assert _refusal(socket) == (FROM_PARAM, None)
    else:
        assert not socket.of_type(ErrorEvent)


def test_an_equal_moment_after_audio_is_behind_and_ends_the_session() -> None:
    """WIRE-178: an equal moment is behind once whole milliseconds of audio pass it."""

    async def scenario() -> FakeSocket:
        session, socket, _, _ = await asking(SEVEN_WINDOWS)
        await session.on_message(sustain_from(from_ms=0))
        await session.on_message(append(windows(20)))
        await session.on_message(sustain_from(from_ms=0))
        return socket

    socket = asyncio.run(scenario())

    assert _refusal(socket, failed=1) == (FROM_PARAM, None)


def test_the_event_is_sent_to_a_draining_session_until_its_close() -> None:
    """WIRE-175: a drain sends the moment's event while the open item runs."""

    async def scenario() -> tuple[FakeSocket, int | None]:
        session, socket, hearing, clock = await asking(SEVEN_WINDOWS)
        await session.on_message(sustain_from(from_ms=start_of_window(2)))
        await session.on_message(append(voice("v" * 4)))
        hearing.begin_drain()
        await session.on_message(append(voice("v" * 8)))
        clock.jump(DRAIN_TURN_SECONDS)
        await session.on_tick()
        await session.settled()
        await session.on_tick()
        return socket, socket.close_code

    socket, close_code = asyncio.run(scenario())

    assert len(_moments(socket)) == 1
    assert close_code == WS_1012_SERVICE_RESTART
    assert socket.types().index(SpeechSustainedFrom) < socket.types().index(
        AudioCommitted
    )


def test_nothing_follows_an_error_not_even_a_moments_event() -> None:
    """WIRE-175: after `error` the session sends no event."""

    async def scenario() -> FakeSocket:
        session, socket, _, _ = await asking(SEVEN_WINDOWS)
        await session.on_message(sustain_from(from_ms=start_of_window(2)))
        await session.on_message(append(voice("v" * 6)))
        await session.on_message(append("not base64!"))
        await session.on_message(append(voice("v" * 20)))
        return socket

    socket = asyncio.run(scenario())

    assert socket.types()[-1] is ErrorEvent
    assert _moments(socket) == []
