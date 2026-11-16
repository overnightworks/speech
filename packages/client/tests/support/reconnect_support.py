"""The scripted scenarios of the hearing reconnect tests share: peers, sources, runs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from packages.client.tests.support.hearing_support import (
    fixed_tokens,
    session_to,
)
from packages.client.tests.support.scripted_hearing_support import (
    Refusal,
    Visit,
    opens,
    scripted,
    takes_audio,
)
from speech_client.hearing import (
    ReconnectBound,
    Reconnected,
    SessionTokens,
)
from speech_client.realtime_wire import PcmFormat, ServerEvent, SessionInput
from speech_client.wire import RefusalCode

if TYPE_CHECKING:
    from collections.abc import Callable

    from websockets.asyncio.server import ServerConnection

    from packages.client.tests.support.scripted_hearing_support import Script
    from speech_client.hearing import HearingOutcome, HearingSession, TokenSource


BOUND = ReconnectBound(attempts=3, pause_seconds=0.05, within_seconds=3)
CHUNK_MS = 100
CHUNK = bytes(SessionInput().format.rate * PcmFormat.sample_bytes * CHUNK_MS // 1000)
WHOLE_RUN_SECONDS = 15
UNAVAILABLE = Refusal(RefusalCode.UNAVAILABLE)


class Tokens:
    """A token source that numbers its calls and holds the calls named in `gated`."""

    def __init__(self, *gated: int) -> None:
        """Hold the calls numbered `gated` until they are released."""
        self.calls = 0
        self._reached = {call: asyncio.Event() for call in gated}
        self._released = {call: asyncio.Event() for call in gated}

    async def __call__(self) -> SessionTokens:
        """The tokens of this call, numbered from 1."""
        self.calls += 1
        call = self.calls
        if call in self._reached:
            self._reached[call].set()
            await self._released[call].wait()
        return SessionTokens(f"workload-{call}", f"person-{call}")

    async def reached(self, call: int) -> None:
        """Wait until the held call `call` has been made."""
        async with asyncio.timeout(WHOLE_RUN_SECONDS):
            await self._reached[call].wait()

    def release(self, call: int) -> None:
        """Let the held call `call` return its tokens."""
        self._released[call].set()


async def hear(  # noqa: PLR0913 - one keyword per thing a scenario varies
    port: int,
    *,
    token_source: TokenSource = fixed_tokens,
    deadline_seconds: float = WHOLE_RUN_SECONDS,
    opening_seconds: float = 1,
    held_limit_bytes: int = 1,
    before: Callable[[HearingSession], None] = lambda _: None,
    traceparent: str | None = None,
) -> tuple[list[ServerEvent | Reconnected], HearingOutcome | None]:
    session = session_to(
        port,
        deadline_seconds,
        held_limit_bytes,
        opening_seconds,
        token_source,
        traceparent=traceparent,
    )
    async with session:
        before(session)
        async with asyncio.timeout(WHOLE_RUN_SECONDS):
            events = [event async for event in session.events()]
    return events, session.outcome


def outcome_of(*scripts: Script, **inputs: object) -> HearingOutcome | None:
    async def scenario() -> HearingOutcome | None:
        async with scripted(*scripts) as (port, _):
            return (await hear(port, **inputs))[1]

    return asyncio.run(scenario())


def visits_and_outcome(
    *scripts: Script, **inputs: object
) -> tuple[int, HearingOutcome | None]:
    async def scenario() -> tuple[int, HearingOutcome | None]:
        async with scripted(*scripts) as (port, visits):
            outcome = (await hear(port, **inputs))[1]
            return len(visits), outcome

    return asyncio.run(scenario())


def skip_ahead(seconds: float) -> None:
    """Let the loop's next pass see its clock `seconds` on, and only that pass.

    Every timer within `seconds` is due in that pass, together with the socket data
    already waiting, which fixes how a bound and an event of one pass interleave.
    """
    loop = asyncio.get_running_loop()
    clock = loop.time
    loop.time = lambda: clock() + seconds  # type: ignore[method-assign]
    loop.call_soon(lambda: delattr(loop, "time"))


def jumpable_clock() -> Callable[[float], None]:
    """Make the running loop's clock movable; the returned function moves it on.

    Call it before the session starts: the session reads the clock it finds then.
    """
    loop = asyncio.get_running_loop()
    clock = loop.time
    moved = 0.0

    def jump(seconds: float) -> None:
        nonlocal moved
        moved += seconds

    loop.time = lambda: clock() + moved  # type: ignore[method-assign]
    return jump


async def hear_through(session: HearingSession) -> None:
    async with session:
        async with asyncio.timeout(WHOLE_RUN_SECONDS):
            async for _ in session.events():
                pass


async def collected(session: HearingSession) -> list[ServerEvent | Reconnected]:
    return [event async for event in session.events()]


def reads(
    chunks: int, *, closes_with: int = 1000, says: tuple[str, ...] = ()
) -> Script:
    """Open, say `says`, take `chunks` chunks of audio, then close with a code."""

    async def play(connection: ServerConnection, visit: Visit) -> None:
        await opens(connection, visit)
        for event in says:
            await connection.send(event)
        await takes_audio(connection, visit, chunks)
        await connection.close(closes_with)

    return play
