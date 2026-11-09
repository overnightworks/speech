"""The hearing session's outcomes that the real service never produces."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from websockets.asyncio.server import serve

from speech_runtime_client.hearing import HearingSession
from speech_runtime_client.outcomes import Cut, Malformed
from speech_runtime_client.wire import HearingLanguage

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection

    from speech_runtime_client.hearing import HearingOutcome

LOOPBACK = "127.0.0.1"


async def _says_nonsense(connection: ServerConnection) -> None:
    await connection.send("not json")
    await connection.wait_closed()


async def _drops_the_connection(connection: ServerConnection) -> None:
    connection.transport.abort()


async def _outcome_against(handler: object) -> HearingOutcome | None:
    async with serve(handler, LOOPBACK, 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with HearingSession(
            f"http://{LOOPBACK}:{port}",
            workload_token="w",  # noqa: S106 - a scenario value
            person_token="p",  # noqa: S106 - a scenario value
            language=HearingLanguage.DE,
            deadline_seconds=5,
            held_limit_bytes=1,
        ) as session:
            async with asyncio.timeout(5):
                async for _ in session.events():
                    pass
            return session.outcome


@pytest.mark.parametrize(
    ("handler", "outcome"),
    [(_says_nonsense, Malformed()), (_drops_the_connection, Cut())],
)
def test_a_server_that_breaks_the_wire_or_drops_ends_the_session_with_its_outcome(
    handler: object, outcome: HearingOutcome
) -> None:
    assert asyncio.run(_outcome_against(handler)) == outcome
