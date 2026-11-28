"""The trace an utterance session joins and the trace its utterances carry."""

from __future__ import annotations

import asyncio
import functools
from contextlib import nullcontext
from typing import TYPE_CHECKING

import pytest
from logline import current_traceparent, unit_trace
from logline.testing import configured

from packages.client.tests.support.utterances_support import (
    Recorder,
    completed,
    hear,
    hearing,
    service_saying,
    utterance_of,
)
from speech_client.wire import TRACEPARENT_HEADER

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection


@pytest.mark.parametrize(
    "trace", ["caller's", "ambient", "ambient-unconfigured", "none"]
)
def test_the_session_upgrade_carries_the_callers_traceparent_else_the_running_one(
    trace: str,
) -> None:
    """WIRE-149: the upgrade carries the caller's, else the trace entered in.

    The caller's is given inside a running trace, which it must win over; the
    running trace reaches the upgrade through the host's configured logline, so
    a host that never configured it sends none.
    """
    given = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    arrived: list[str | None] = []

    async def records(connection: ServerConnection) -> None:
        arrived.append(connection.request.headers.get(TRACEPARENT_HEADER))
        await connection.recv()

    async def scenario() -> str | None:
        peer = functools.partial(service_saying, records)
        if trace == "none":
            await hear(peer)
            return None
        with unit_trace(None):
            ambient = current_traceparent()
            await hear(peer, traceparent=given if trace == "caller's" else None)
            return ambient

    with nullcontext() if trace == "ambient-unconfigured" else configured("speech"):
        ambient = asyncio.run(scenario())

    expected = {"caller's": given, "ambient": ambient}.get(trace)
    assert arrived == [expected]


def test_an_utterance_carries_the_item_and_the_trace_of_its_transcript() -> None:
    """WIRE-150: the utterance names its item and its turn's traceparent."""
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    recorder = Recorder()

    asyncio.run(
        hear(
            hearing(
                completed("a", "one", traceparent=traceparent),
                completed("b", "two"),
            ),
            None,
            recorder,
        )
    )

    assert recorder.got == [
        utterance_of("one", item_id="a", traceparent=traceparent),
        utterance_of("two", item_id="b"),
    ]
