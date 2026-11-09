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

from speech_client.hearing import Closed, HearingSession, Refused
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
    HearingLanguage,
    RefusalCode,
    TranscriptDelta,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from websockets.asyncio.server import ServerConnection

    from speech_client.hearing import HearingOutcome
    from speech_client.relay import RelayOutcome
    from speech_client.wire import ServerEvent

FRAME_LIMIT = 8
CLOSE_TIMEOUT_SECONDS = 10
SECRET_WORDS = "heard-secret-words"  # noqa: S105 - a scenario value
DISCONNECT = {"type": "websocket.disconnect", "code": 1001}


def _binary(size: int) -> dict[str, object]:
    return {"type": "websocket.receive", "bytes": b"\0" * size, "text": None}


def _text(text: str = "hello") -> dict[str, object]:
    return {"type": "websocket.receive", "bytes": None, "text": text}


class _Browser:
    """A browser socket that plays its messages, then waits for its peer."""

    def __init__(self, messages: list[Mapping[str, object]]) -> None:
        self._messages = list(messages)
        self.closed_with: list[int] = []

    async def receive(self) -> Mapping[str, object]:
        if self._messages:
            return self._messages.pop(0)
        await asyncio.Event().wait()
        raise AssertionError

    async def close(self, code: int) -> None:
        self.closed_with.append(code)


class _Session:
    """A hearing session that gives its events, then its outcome, or never ends."""

    def __init__(
        self,
        events: list[ServerEvent] | None = None,
        ends: HearingOutcome | None = None,
    ) -> None:
        self._events = events or []
        self._ends = ends
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
        for event in self._events:
            yield event
        if self._ends is None:
            await asyncio.Event().wait()
        self.outcome = self._ends


async def _no_events(_: ServerEvent) -> None:
    return None


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


def test_the_session_events_go_to_the_app_in_order() -> None:
    events: list[ServerEvent] = [
        TranscriptDelta(item_id="i", delta="a"),
        TranscriptDelta(item_id="i", delta="b"),
    ]
    seen: list[ServerEvent] = []

    async def take(event: ServerEvent) -> None:
        seen.append(event)

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


def test_the_hop_logs_nothing_and_no_outcome_names_heard_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[ServerEvent] = [TranscriptDelta(item_id="i", delta=SECRET_WORDS)]
    caplog.set_level(logging.DEBUG)

    ended = asyncio.run(_hop(_Browser([_text(SECRET_WORDS)]), _Session(events)))

    assert not [r for r in caplog.records if r.name.startswith("speech_client")]
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


def test_a_hop_cut_while_the_service_stalls_leaves_nothing_to_wait_for() -> None:
    async def scenario() -> None:
        async with _stalled_service() as (service_url, opened):
            session = HearingSession(
                service_url,
                workload_token="workload",  # noqa: S106 - a scenario value
                person_token="person",  # noqa: S106 - a scenario value
                language=HearingLanguage.EN,
                deadline_seconds=30,
                held_limit_bytes=1,
            )
            app_ended = asyncio.Event()
            hop = asyncio.create_task(
                relay(
                    _Browser([]),
                    session,
                    app_ended=app_ended,
                    on_event=_no_events,
                    frame_limit_bytes=FRAME_LIMIT,
                )
            )
            await opened.wait()
            app_ended.set()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(hop, 0.1)

    started = time.monotonic()
    asyncio.run(scenario())

    assert time.monotonic() - started < CLOSE_TIMEOUT_SECONDS / 2
