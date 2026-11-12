"""The relay of a browser's live audio to the service, for an app's server.

`relay` forwards the binary frames of a browser socket to a `HearingSession` and
hands the session's typed events to the app, never to the browser. The tokens
stay in the session on the app's server; the relay forwards the browser no token
and no event, only the close code that names how the hop ended. What the app's
route writes on its own browser socket is the app's. The app keeps the route,
admission (accepting the socket included), its switches, the token choice and
when a hearing ends: it sets `app_ended` to end it.

Frames from the browser are untrusted. A binary frame is forwarded unread; the
session holds it up to its own byte limit. A text frame, or a frame over
`frame_limit_bytes`, ends the hop with its named close code. Nothing here
measures loudness or detects speech: everything is streamed, and the service's
own detector decides what speech is.

A browser sends 100 ms of audio a frame: `HOP_FRAME_SAMPLES` samples of the
wire's `PcmFormat`, `HOP_FRAME_BYTES` bytes; the last frame of a hop may be
shorter. An app may pass `HOP_FRAME_BYTES` as `frame_limit_bytes`.

`relay` returns one `RelayOutcome`, a closed union a caller matches with
`assert_never`: the session's own `HearingOutcome`, or a hop end (`BrowserLeft`,
`AppEnded`, `RuleBroken`). Unless the browser left, the relay closes the browser
socket with the `HopClose` of that end. `app_ended` ends the browser hop at once,
and the end is hard: a frame the relay receives, or an event the session yields,
after `app_ended` is set, also when it was set before the call, reaches neither
the session nor `on_event`, and the hop returns `AppEnded`. A browser that
disconnects in the same pass still ends the hop as `BrowserLeft`. Two limits
stay: frames the relay accepted before the end and the session still holds may be
written to the service until the session is left (at most the session's held
limit), and an `on_event` call already running when the end is set is cancelled.
`relay` itself returns when the session has closed: within the socket's close
timeout, or at the session's deadline when the service no longer reads. A cancel
of `relay`, a caller's timeout included, aborts the service socket at once,
without a close handshake, whenever it lands, and leaves no task behind.
`relay` enters and leaves `session` itself: pass it unentered. A failure of
`on_event` is raised, and no task the relay started outlives it. The relay logs
nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Final, Protocol, assert_never

from speech_client.hearing import Closed, HearingOutcome, HearingSession, Refused
from speech_client.outcomes import (
    Busy,
    Cut,
    Malformed,
    OverCap,
    PastDeadline,
    Unreachable,
)
from speech_client.wire import PcmFormat, SessionInput

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from speech_client.wire import ServerEvent


HOP_FRAME_SAMPLES: Final = SessionInput().format.rate // 10
HOP_FRAME_BYTES: Final = HOP_FRAME_SAMPLES * PcmFormat.sample_bytes * PcmFormat.channels


class HopClose(IntEnum):
    """The close code of the hop's browser socket, one per way the hop ends.

    A code of the hop's own is 4000 plus the HTTP status nearest its meaning.
    """

    NORMAL = 1000
    TEXT_FRAME = 4400  # Bad Request
    FRAME_TOO_LARGE = 4413  # Content Too Large
    OVER_CAP = 4507  # Insufficient Storage
    BUSY = 4429  # Too Many Requests
    PAST_DEADLINE = 4504  # Gateway Timeout
    SERVICE_FAILED = 4502  # Bad Gateway


class BrowserSocket(Protocol):
    """What the relay calls on the browser's socket; FastAPI's WebSocket fits."""

    async def receive(self) -> Mapping[str, object]:
        """The next ASGI message of the socket."""
        ...

    async def close(self, code: int) -> None:
        """Close the socket with `code`."""
        ...


@dataclass(frozen=True, slots=True)
class BrowserLeft:
    """The browser disconnected."""


@dataclass(frozen=True, slots=True)
class AppEnded:
    """The app set its end signal."""


@dataclass(frozen=True, slots=True)
class RuleBroken:
    """A frame broke the hop's rules; `close` names which rule."""

    close: HopClose


type RelayOutcome = HearingOutcome | BrowserLeft | AppEnded | RuleBroken


def _close_for(outcome: RelayOutcome) -> HopClose | None:  # noqa: PLR0911 - one per member
    match outcome:
        case BrowserLeft():
            return None
        case Closed() | AppEnded():
            return HopClose.NORMAL
        case RuleBroken(close):
            return close
        case OverCap():
            return HopClose.OVER_CAP
        case Busy():
            return HopClose.BUSY
        case PastDeadline():
            return HopClose.PAST_DEADLINE
        case Refused() | Malformed() | Cut() | Unreachable():
            return HopClose.SERVICE_FAILED
        case _:
            assert_never(outcome)


async def _forward(
    browser: BrowserSocket,
    session: HearingSession,
    frame_limit_bytes: int,
    app_ended: asyncio.Event,
) -> RelayOutcome:
    while True:
        message = await browser.receive()
        if message["type"] == "websocket.disconnect":
            return BrowserLeft()
        if app_ended.is_set():
            return AppEnded()
        frame = message.get("bytes")
        if not isinstance(frame, bytes):
            return RuleBroken(HopClose.TEXT_FRAME)
        if len(frame) > frame_limit_bytes:
            return RuleBroken(HopClose.FRAME_TOO_LARGE)
        session.send(frame)


async def _deliver(
    session: HearingSession,
    on_event: Callable[[ServerEvent], Awaitable[None]],
    app_ended: asyncio.Event,
) -> RelayOutcome:
    async with contextlib.aclosing(session.events()) as events:
        async for event in events:
            if app_ended.is_set():
                return AppEnded()
            await on_event(event)
    # After an unexpected failure the outcome stays `None`; leaving the session
    # then raises that failure in place of this assertion.
    assert session.outcome is not None  # noqa: S101 - events ran out
    return session.outcome


async def _signalled(app_ended: asyncio.Event) -> RelayOutcome:
    await app_ended.wait()
    return AppEnded()


async def relay(
    browser: BrowserSocket,
    session: HearingSession,
    *,
    app_ended: asyncio.Event,
    on_event: Callable[[ServerEvent], Awaitable[None]],
    frame_limit_bytes: int,
) -> RelayOutcome:
    """Relay `browser` to `session` until the first end; see the module."""
    async with session:
        ends = [
            asyncio.create_task(
                _forward(browser, session, frame_limit_bytes, app_ended)
            ),
            asyncio.create_task(_deliver(session, on_event, app_ended)),
            asyncio.create_task(_signalled(app_ended)),
        ]
        try:
            await asyncio.wait(ends, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for end in ends:
                end.cancel()
            await asyncio.gather(*ends, return_exceptions=True)
        # `result` raises a failure of the app's event handling.
        outcome, *_ = (end.result() for end in ends if not end.cancelled())
        code = _close_for(outcome)
        if code is not None:
            await browser.close(code)
    return outcome
