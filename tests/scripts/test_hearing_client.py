"""The hearing proof's client: the live upgrade, the turn and the refusal codes."""

from __future__ import annotations

import asyncio
from http import HTTPStatus
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from speech_client.hearing import Refused
from speech_client.wire import (
    DurationUsage,
    RefusalCode,
    SessionUpdated,
    TranscriptCompleted,
    TranscriptionSession,
    refusal,
)
from tests.support.identity_support import MINTING
from tests.support.script_support import script_module

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from websockets.asyncio.server import ServerConnection

    from hearing_figures import LiveTurn
    from speech_client.wire import ServerEvent

CLIENT = script_module("hearing_client")


def unavailable_body() -> bytes:
    return refusal(RefusalCode.UNAVAILABLE).model_dump_json().encode()


@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        (HTTPStatus.SERVICE_UNAVAILABLE, unavailable_body(), RefusalCode.UNAVAILABLE),
        (
            HTTPStatus.SERVICE_UNAVAILABLE,
            refusal(RefusalCode.BUSY).model_dump_json().encode(),
            RefusalCode.BUSY,
        ),
        (HTTPStatus.SERVICE_UNAVAILABLE, b"", None),
        (HTTPStatus.SERVICE_UNAVAILABLE, b"not json", None),
        (HTTPStatus.OK, unavailable_body(), None),
    ],
)
def test_only_a_503_with_the_contract_body_gives_a_refusal_code(
    status: HTTPStatus, body: bytes, code: RefusalCode | None
) -> None:
    assert CLIENT.refusal_code(status, body) == code


def test_a_refused_upgrade_gives_the_status_and_the_code_that_refused_it() -> None:
    body = unavailable_body()

    def refuse(_: ServerConnection, __: Request) -> Response:
        return Response(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Service Unavailable",
            Headers({"Content-Length": str(len(body))}),
            body,
        )

    async def scenario() -> object:
        async with serve(
            never_handles, "127.0.0.1", 0, process_request=refuse
        ) as server:
            port = server.sockets[0].getsockname()[1]
            service = SimpleNamespace(
                base_url=f"http://127.0.0.1:{port}", minting=MINTING
            )
            return await asyncio.wait_for(CLIENT.hear_upgrade(service), 10)

    outcome = asyncio.run(scenario())

    assert outcome == Refused(HTTPStatus.SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE)


async def never_handles(_: ServerConnection) -> None:
    """The handler of a service that refuses every upgrade before it runs."""


def test_an_accepted_upgrade_gives_no_outcome() -> None:
    async def accepting(connection: ServerConnection) -> None:
        updated = SessionUpdated(session=TranscriptionSession(type="transcription"))
        await connection.send(updated.model_dump_json())
        await connection.wait_closed()

    async def scenario() -> object:
        async with serve(accepting, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            service = SimpleNamespace(
                base_url=f"http://127.0.0.1:{port}", minting=MINTING
            )
            return await asyncio.wait_for(CLIENT.hear_upgrade(service), 10)

    assert asyncio.run(scenario()) is None


async def delivering(events: Sequence[ServerEvent]) -> AsyncIterator[ServerEvent]:
    """Stand in for the events of a session that delivers `events`."""
    for event in events:
        yield event


def receive_turn(events: Sequence[ServerEvent]) -> LiveTurn | None:
    return asyncio.run(CLIENT.receive_turn(delivering(events), 1.0, 0.0))


def test_a_valid_event_of_no_interest_is_ignored_until_the_turn_completes() -> None:
    updated = SessionUpdated(session=TranscriptionSession(type="transcription"))
    completed = TranscriptCompleted(
        item_id="i", transcript="Hallo", usage=DurationUsage(seconds=1.0)
    )

    turn = receive_turn([updated, completed])

    assert turn is not None
    assert turn.transcript == "Hallo"


def test_a_session_that_ends_without_a_turn_gives_no_turn() -> None:
    assert receive_turn([]) is None
