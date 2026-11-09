"""The relay of a browser's live audio to the service, for an app's server.

`relay` forwards the binary frames of a browser socket to a `HearingSession` and
hands the session's typed events to the app, never to the browser. The tokens
stay in the session on the app's server; the browser sees no token and no event,
only the close code that names how the hop ended. The app keeps the route,
admission (accepting the socket included), its switches, the token choice and
when a hearing ends: it sets `app_ended` to end it.

Frames from the browser are untrusted. A binary frame is forwarded unread; the
session holds it up to its own byte limit. A text frame, or a frame over
`frame_limit_bytes`, ends the hop with its named close code. Nothing here
measures loudness or detects speech: everything is streamed, and the service's
own detector decides what speech is.

`relay` returns one `RelayOutcome`, a closed union a caller matches with
`assert_never`: the session's own `HearingOutcome`, or a hop end (`BrowserLeft`,
`AppEnded`, `RuleBroken`). Unless the browser left, the relay closes the browser
socket with the `HopClose` of that end. `app_ended` takes effect at once; the
session's polite close of a stalled service is cut after the socket's own close
timeout of 10 s at the most. A failure of `on_event` is raised, and no task the
relay started outlives it. The relay logs nothing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol, assert_never

from speech_client.hearing import Closed, HearingOutcome, HearingSession, Refused
from speech_client.outcomes import (
    Busy,
    Cut,
    Malformed,
    OverCap,
    PastDeadline,
    Unreachable,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from speech_client.wire import ServerEvent


class HopClose(IntEnum):
    """The close code of the hop's browser socket, one per way the hop ends."""

    NORMAL = 1000
    TEXT_FRAME = 4400
    FRAME_TOO_LARGE = 4401
    OVER_CAP = 4402
    BUSY = 4403
    PAST_DEADLINE = 4404
    SERVICE_FAILED = 4405


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
    browser: BrowserSocket, session: HearingSession, frame_limit_bytes: int
) -> RelayOutcome:
    while True:
        message = await browser.receive()
        if message["type"] == "websocket.disconnect":
            return BrowserLeft()
        frame = message.get("bytes")
        if not isinstance(frame, bytes):
            return RuleBroken(HopClose.TEXT_FRAME)
        if len(frame) > frame_limit_bytes:
            return RuleBroken(HopClose.FRAME_TOO_LARGE)
        session.send(frame)


async def _deliver(
    session: HearingSession, on_event: Callable[[ServerEvent], Awaitable[None]]
) -> RelayOutcome:
    async for event in session.events():
        await on_event(event)
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
            asyncio.create_task(_forward(browser, session, frame_limit_bytes)),
            asyncio.create_task(_deliver(session, on_event)),
            asyncio.create_task(_signalled(app_ended)),
        ]
        try:
            await asyncio.wait(ends, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for end in ends:
                end.cancel()
            await asyncio.gather(*ends, return_exceptions=True)
        # `result` raises a failure of the app's event handling.
        outcome = [end.result() for end in ends if not end.cancelled()][0]
        code = _close_for(outcome)
        if code is not None:
            await browser.close(code)
    return outcome
