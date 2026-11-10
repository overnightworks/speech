"""The scripted peers and session builders the hearing tests share."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from websockets.asyncio.server import serve

from speech_client.hearing import OPENING_SECONDS, HearingSession
from speech_client.wire import HearingLanguage, TranscriptDelta

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from websockets.asyncio.server import ServerConnection

    from speech_client.hearing import HearingOutcome

LOOPBACK = "127.0.0.1"
WORKLOAD_TOKEN = "workload-secret"  # noqa: S105 - a scenario value
PERSON_TOKEN = "person-secret"  # noqa: S105 - a scenario value
HEARD_TEXT = "heard-secret-words"
# Well below the socket close timeout (10 s) and the session deadline (30 s).
AT_ONCE_SECONDS = 1
OPENING_MARGIN = 2  # met in time, below the close timeout (10 s)


async def silent_peer(reader: asyncio.StreamReader, _: object) -> None:
    await reader.read()


async def takes_every_event(session: HearingSession) -> None:
    async for _ in session.events():
        pass


def session_to(
    port: int,
    deadline_seconds: float = 5,
    held_limit_bytes: int = 1,
    opening_seconds: float = OPENING_SECONDS,
) -> HearingSession:
    return HearingSession(
        f"http://{LOOPBACK}:{port}",
        workload_token=WORKLOAD_TOKEN,
        person_token=PERSON_TOKEN,
        language=HearingLanguage.DE,
        deadline_seconds=deadline_seconds,
        held_limit_bytes=held_limit_bytes,
        opening_seconds=opening_seconds,
    )


async def outcome_at(
    port: int, deadline_seconds: float = 5, opening_seconds: float = OPENING_SECONDS
) -> HearingOutcome | None:
    async with session_to(
        port, deadline_seconds, opening_seconds=opening_seconds
    ) as session:
        async with asyncio.timeout(deadline_seconds + 5):
            async for _ in session.events():
                pass
        return session.outcome


@contextlib.asynccontextmanager
async def stalled_service(
    says_first: tuple[str, ...] = (),
) -> AsyncIterator[tuple[int, asyncio.Event]]:
    """A service that reads the first frame, then stops reading; yields port, opened."""
    released = asyncio.Event()
    opened = asyncio.Event()

    async def stalls(connection: ServerConnection) -> None:
        await connection.recv()
        for message in says_first:
            await connection.send(message)
        opened.set()
        connection.transport.pause_reading()
        await released.wait()

    async with serve(stalls, LOOPBACK, 0, close_timeout=0.2) as server:
        try:
            yield server.sockets[0].getsockname()[1], opened
        finally:
            released.set()


HEARD_DELTA = TranscriptDelta(item_id="i", delta=HEARD_TEXT).model_dump_json()
