"""The relay's hop rules against a scripted browser and a scripted session."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Self

import pytest
from websockets.asyncio.server import serve

from speech_client.hearing import OPENING_SECONDS, Closed, HearingSession, Refused
from speech_client.outcomes import (
    Busy,
    Cut,
    Malformed,
    OverCap,
    PastDeadline,
    Unreachable,
)
from speech_client.relay import (
    AppEnded,
    BrowserLeft,
    HopClose,
    RuleBroken,
    relay,
)
from speech_client.wire import (
    BusyWait,
    DurationUsage,
    Language,
    RefusalCode,
    TranscriptCompleted,
    TranscriptDelta,
    TranscriptLanguage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

    from websockets.asyncio.server import ServerConnection

    from speech_client.hearing import HearingOutcome
    from speech_client.relay import RelayOutcome
    from speech_client.wire import ServerEvent

FRAME_LIMIT = 8
# Well below the socket close timeout (10 s) and the session deadline (30 s).
CLOSE_BOUND_SECONDS = 2
# An opening bound a stalled service passes at once, far below the deadline.
STALLED_OPENING_SECONDS = 0.3
SECRET_WORDS = "heard-secret-words"  # noqa: S105 - a scenario value
DISCONNECT = {"type": "websocket.disconnect", "code": 1001}


def _binary(size: int) -> dict[str, object]:
    return {"type": "websocket.receive", "bytes": b"\0" * size, "text": None}


def _text(text: str = "hello") -> dict[str, object]:
    return {"type": "websocket.receive", "bytes": None, "text": text}


class _Browser:
    """A browser socket that plays its messages, then waits for its peer.

    `ends_app` is set as the last message arrives: the app ends the hearing in
    the very pass the browser's frame is received.
    """

    def __init__(
        self,
        messages: list[Mapping[str, object]],
        ends_app: asyncio.Event | None = None,
    ) -> None:
        self._messages = list(messages)
        self._ends_app = ends_app
        self.closed_with: list[int] = []

    async def receive(self) -> Mapping[str, object]:
        if self._messages:
            if self._ends_app is not None and len(self._messages) == 1:
                self._ends_app.set()
            return self._messages.pop(0)
        await asyncio.Event().wait()
        raise AssertionError

    async def close(self, code: int) -> None:
        self.closed_with.append(code)


class _Session:
    """A hearing session that gives its events, then its outcome, or never ends.

    `ends_app` is set as the last event is yielded: the app ends the hearing in
    the very pass the session's event arrives.
    """

    def __init__(
        self,
        events: list[ServerEvent] | None = None,
        ends: HearingOutcome | None = None,
        ends_app: asyncio.Event | None = None,
    ) -> None:
        self._events = events or []
        self._ends = ends
        self._ends_app = ends_app
        self.outcome: HearingOutcome | None = None
        self.sent: list[bytes] = []
        self.left = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        self.left = True

    def send(self, pcm: bytes) -> None:
        self.sent.append(pcm)

    async def events(self) -> AsyncIterator[ServerEvent]:
        for index, event in enumerate(self._events):
            if self._ends_app is not None and index == len(self._events) - 1:
                self._ends_app.set()
            yield event
        if self._ends is None:
            await asyncio.Event().wait()
        self.outcome = self._ends


async def _no_events(_: ServerEvent) -> None:
    return None


def _collector() -> tuple[list[ServerEvent], Callable[[ServerEvent], Awaitable[None]]]:
    """The events an app handler took, and the handler that takes them."""
    seen: list[ServerEvent] = []

    async def take(event: ServerEvent) -> None:
        seen.append(event)

    return seen, take


async def _hop(
    browser: _Browser,
    session: _Session,
    *,
    app_ended: asyncio.Event | None = None,
    on_event: object = _no_events,
) -> RelayOutcome:
    return await asyncio.wait_for(
        relay(
            browser,
            session,
            app_ended=app_ended or asyncio.Event(),
            on_event=on_event,
            frame_limit_bytes=FRAME_LIMIT,
        ),
        5,
    )


@pytest.mark.parametrize(
    ("messages", "outcome", "close"),
    [
        pytest.param(
            [_binary(1), DISCONNECT], BrowserLeft(), None, id="the-browser-leaves"
        ),
        pytest.param(
            [_text()],
            RuleBroken(HopClose.TEXT_FRAME),
            HopClose.TEXT_FRAME,
            id="a-text-frame",
        ),
        pytest.param(
            [_text("")],
            RuleBroken(HopClose.TEXT_FRAME),
            HopClose.TEXT_FRAME,
            id="an-empty-text-frame",
        ),
        pytest.param(
            [_binary(FRAME_LIMIT + 1)],
            RuleBroken(HopClose.FRAME_TOO_LARGE),
            HopClose.FRAME_TOO_LARGE,
            id="a-frame-one-byte-over-the-limit",
        ),
    ],
)
def test_the_browsers_frames_end_the_hop_with_their_named_outcome(
    messages: list[Mapping[str, object]],
    outcome: RelayOutcome,
    close: HopClose | None,
) -> None:
    browser = _Browser(messages)

    ended = asyncio.run(_hop(browser, _Session()))

    assert ended == outcome
    assert browser.closed_with == ([] if close is None else [close])


def test_a_frame_exactly_the_limit_is_forwarded_unread_in_order() -> None:
    browser = _Browser([_binary(FRAME_LIMIT), _binary(0), DISCONNECT])
    session = _Session()

    asyncio.run(_hop(browser, session))

    assert session.sent == [b"\0" * FRAME_LIMIT, b""]


def test_a_rule_broken_frame_is_not_forwarded() -> None:
    browser = _Browser([_binary(1), _text()])
    session = _Session()

    asyncio.run(_hop(browser, session))

    assert session.sent == [b"\0"]


@pytest.mark.parametrize(
    ("ends", "close"),
    [
        (Closed(), HopClose.NORMAL),
        (Busy(BusyWait(1)), HopClose.BUSY),
        (OverCap(), HopClose.OVER_CAP),
        (PastDeadline(), HopClose.PAST_DEADLINE),
        (Refused(403, RefusalCode.UNAUTHORIZED), HopClose.SERVICE_FAILED),
        (Malformed(), HopClose.SERVICE_FAILED),
        (Cut(), HopClose.SERVICE_FAILED),
        (Unreachable(), HopClose.SERVICE_FAILED),
    ],
)
def test_a_session_outcome_reaches_the_app_unchanged_and_the_browser_as_its_code(
    ends: HearingOutcome, close: HopClose
) -> None:
    browser = _Browser([])

    ended = asyncio.run(_hop(browser, _Session(ends=ends)))

    assert ended == ends
    assert browser.closed_with == [close]


def test_the_apps_end_signal_ends_the_hop_and_leaves_the_session() -> None:
    browser = _Browser([])
    session = _Session()
    app_ended = asyncio.Event()
    app_ended.set()

    ended = asyncio.run(_hop(browser, session, app_ended=app_ended))

    assert ended == AppEnded()
    assert session.left
    assert browser.closed_with == [HopClose.NORMAL]


def _app_ended(*, before_the_call: bool) -> asyncio.Event:
    app_ended = asyncio.Event()
    if before_the_call:
        app_ended.set()
    return app_ended


@pytest.mark.parametrize(
    "before_the_call",
    [
        pytest.param(True, id="set-before-the-call"),
        pytest.param(False, id="set-as-the-frame-arrives"),
    ],
)
@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(_binary(1), id="a-frame"),
        pytest.param(_text(), id="a-text-frame"),
        pytest.param(_binary(FRAME_LIMIT + 1), id="a-frame-over-the-limit"),
    ],
)
def test_a_frame_received_after_the_apps_end_is_not_handed_to_the_session(
    frame: Mapping[str, object], *, before_the_call: bool
) -> None:
    app_ended = _app_ended(before_the_call=before_the_call)
    browser = _Browser([frame], None if before_the_call else app_ended)
    session = _Session()

    ended = asyncio.run(_hop(browser, session, app_ended=app_ended))

    assert ended == AppEnded()
    assert session.sent == []
    assert browser.closed_with == [HopClose.NORMAL]


def test_a_frame_received_before_the_apps_end_is_still_handed_to_the_session() -> None:
    app_ended = asyncio.Event()
    browser = _Browser([_binary(1), _binary(2)], app_ended)
    session = _Session()

    ended = asyncio.run(_hop(browser, session, app_ended=app_ended))

    assert ended == AppEnded()
    assert session.sent == [b"\0"]


@pytest.mark.parametrize(
    "before_the_call",
    [
        pytest.param(True, id="set-before-the-call"),
        pytest.param(False, id="set-as-the-event-arrives"),
    ],
)
@pytest.mark.parametrize(
    "event",
    [
        pytest.param(TranscriptDelta(item_id="i", delta="a"), id="a-delta"),
        pytest.param(
            TranscriptCompleted(
                item_id="i",
                transcript=SECRET_WORDS,
                usage=DurationUsage(seconds=1),
                languages=(TranscriptLanguage(code=Language.EN),),
            ),
            id="a-transcript-finishing-late",
        ),
    ],
)
def test_an_event_yielded_after_the_apps_end_is_not_handed_to_the_app(
    event: ServerEvent, *, before_the_call: bool
) -> None:
    app_ended = _app_ended(before_the_call=before_the_call)
    session = _Session([event], ends_app=None if before_the_call else app_ended)
    browser = _Browser([])
    seen, take = _collector()

    async def scenario() -> tuple[RelayOutcome, set[asyncio.Task[object]]]:
        ended = await _hop(browser, session, app_ended=app_ended, on_event=take)
        return ended, asyncio.all_tasks() - {asyncio.current_task()}

    ended, tasks_left = asyncio.run(scenario())

    assert ended == AppEnded()
    assert seen == []
    assert browser.closed_with == [HopClose.NORMAL]
    assert not tasks_left


def test_an_event_yielded_before_the_apps_end_is_still_handed_to_the_app() -> None:
    first = TranscriptDelta(item_id="i", delta="a")
    app_ended = asyncio.Event()
    session = _Session(
        [first, TranscriptDelta(item_id="i", delta="b")], ends_app=app_ended
    )
    seen, take = _collector()
    ended = asyncio.run(_hop(_Browser([]), session, app_ended=app_ended, on_event=take))

    assert ended == AppEnded()
    assert seen == [first]


def test_a_disconnect_in_the_pass_of_the_apps_end_is_the_browser_leaving() -> None:
    app_ended = asyncio.Event()
    browser = _Browser([DISCONNECT], app_ended)

    ended = asyncio.run(_hop(browser, _Session(), app_ended=app_ended))

    assert ended == BrowserLeft()
    assert browser.closed_with == []


def test_the_session_events_go_to_the_app_in_order() -> None:
    events: list[ServerEvent] = [
        TranscriptDelta(item_id="i", delta="a"),
        TranscriptDelta(item_id="i", delta="b"),
    ]
    seen, take = _collector()
    asyncio.run(_hop(_Browser([]), _Session(events, ends=Closed()), on_event=take))

    assert seen == events


def test_a_failure_of_the_apps_event_handling_is_raised_and_leaves_nothing_behind() -> (
    None
):
    session = _Session([TranscriptDelta(item_id="i", delta="a")])

    async def refuse(_: ServerEvent) -> None:
        message = "the app failed"
        raise RuntimeError(message)

    async def scenario() -> set[asyncio.Task[object]]:
        with pytest.raises(RuntimeError, match="the app failed"):
            await _hop(_Browser([]), session, on_event=refuse)
        return asyncio.all_tasks() - {asyncio.current_task()}

    assert not asyncio.run(scenario())
    assert session.left


def test_a_finished_hop_leaves_no_task_behind() -> None:
    async def scenario() -> set[asyncio.Task[object]]:
        await _hop(_Browser([_text()]), _Session())
        return asyncio.all_tasks() - {asyncio.current_task()}

    assert not asyncio.run(scenario())


def test_no_log_and_no_outcome_carries_heard_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[ServerEvent] = [TranscriptDelta(item_id="i", delta=SECRET_WORDS)]
    caplog.set_level(logging.DEBUG)

    ended = asyncio.run(_hop(_Browser([_text(SECRET_WORDS)]), _Session(events)))

    assert SECRET_WORDS not in caplog.text
    assert SECRET_WORDS not in repr(ended)


def test_the_private_hop_codes_lie_in_the_private_range() -> None:
    private = [code for code in HopClose if code is not HopClose.NORMAL]

    assert private
    assert all(4000 <= code <= 4999 for code in private)  # noqa: PLR2004 - RFC 6455


def test_the_relay_needs_no_starlette_to_be_imported() -> None:
    probe = (
        "import sys, speech_client.relay;"
        "sys.exit('starlette' in sys.modules or 'fastapi' in sys.modules)"
    )

    done = subprocess.run(  # noqa: S603 - a fixed probe of this interpreter
        [sys.executable, "-I", "-c", probe], check=False
    )

    assert done.returncode == 0


async def _never_answers(reader: asyncio.StreamReader, _: object) -> None:
    await reader.read()


@contextlib.asynccontextmanager
async def _stalled_service() -> AsyncIterator[tuple[str, asyncio.Event]]:
    """A service that reads the first frame, then stops reading and never pings."""
    opened = asyncio.Event()
    released = asyncio.Event()

    async def stalls(connection: ServerConnection) -> None:
        await connection.recv()
        opened.set()
        connection.transport.pause_reading()
        await released.wait()

    async with serve(
        stalls, "127.0.0.1", 0, close_timeout=0.2, ping_interval=None
    ) as server:
        try:
            yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", opened
        finally:
            released.set()


def _stalled_hop(
    service_url: str,
    app_ended: asyncio.Event,
    browser: _Browser | None = None,
    opening_seconds: float = OPENING_SECONDS,
) -> Awaitable[RelayOutcome]:
    session = HearingSession(
        service_url,
        workload_token="workload",  # noqa: S106 - a scenario value
        person_token="person",  # noqa: S106 - a scenario value
        languages=(Language.EN,),
        deadline_seconds=30,
        held_limit_bytes=1,
        opening_seconds=opening_seconds,
    )
    return relay(
        browser or _Browser([]),
        session,
        app_ended=app_ended,
        on_event=_no_events,
        frame_limit_bytes=FRAME_LIMIT,
    )


def test_a_hop_cut_while_the_service_stalls_leaves_nothing_to_wait_for() -> None:
    async def scenario() -> None:
        async with _stalled_service() as (service_url, opened):
            app_ended = asyncio.Event()
            hop = asyncio.create_task(_stalled_hop(service_url, app_ended))
            await opened.wait()
            app_ended.set()
            await asyncio.wait({hop}, timeout=0.1)
            hop.cancel()
            await asyncio.wait({hop}, timeout=1)
            assert hop.done()
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())


def test_a_caller_timeout_before_the_hop_ends_aborts_a_stalled_service() -> None:
    async def scenario() -> float:
        async with _stalled_service() as (service_url, opened):
            hop = asyncio.create_task(_stalled_hop(service_url, asyncio.Event()))
            await opened.wait()
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.2):
                    await hop
            elapsed = time.monotonic() - started
        assert asyncio.all_tasks() == {asyncio.current_task()}
        return elapsed

    assert asyncio.run(scenario()) < CLOSE_BOUND_SECONDS


@pytest.mark.parametrize("yields_before_cancel", range(31))
def test_a_cancel_at_any_moment_after_the_apps_end_signal_returns_at_once(
    yields_before_cancel: int,
) -> None:
    async def scenario() -> float:
        async with _stalled_service() as (service_url, opened):
            app_ended = asyncio.Event()
            hop = asyncio.create_task(_stalled_hop(service_url, app_ended))
            await opened.wait()
            app_ended.set()
            for _ in range(yields_before_cancel):
                await asyncio.sleep(0)
            started = time.monotonic()
            hop.cancel()
            await asyncio.wait({hop}, timeout=CLOSE_BOUND_SECONDS)
            elapsed = time.monotonic() - started
            assert hop.cancelled()
        assert asyncio.all_tasks() == {asyncio.current_task()}
        return elapsed

    assert asyncio.run(scenario()) < CLOSE_BOUND_SECONDS / 2


@pytest.mark.parametrize("stall", ["handshake", "first-event"])
def test_a_service_that_does_not_open_ends_the_hop_with_the_service_failed_close(
    stall: str,
) -> None:
    async def scenario() -> tuple[RelayOutcome, list[int], float]:
        browser = _Browser([])
        async with contextlib.AsyncExitStack() as stack:
            if stall == "handshake":
                server = await stack.enter_async_context(
                    await asyncio.start_server(_never_answers, "127.0.0.1", 0)
                )
                url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            else:
                url, _ = await stack.enter_async_context(_stalled_service())
            started = time.monotonic()
            outcome = await asyncio.wait_for(
                _stalled_hop(url, asyncio.Event(), browser, STALLED_OPENING_SECONDS),
                CLOSE_BOUND_SECONDS * 2,
            )
            return outcome, browser.closed_with, time.monotonic() - started

    outcome, closed_with, elapsed = asyncio.run(scenario())

    assert outcome == Unreachable()
    assert closed_with == [HopClose.SERVICE_FAILED]
    assert elapsed < CLOSE_BOUND_SECONDS
