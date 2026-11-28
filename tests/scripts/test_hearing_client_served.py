"""The hearing proof's clients against the real service over TCP and failed turns."""

from __future__ import annotations

import asyncio
import re
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest

from hearing_figures import LIVE_INPUT_RATE
from speech_client.hearing import Refused
from speech_client.realtime_wire import (
    DurationUsage,
    ItemFailure,
    TranscriptCompleted,
    TranscriptDelta,
    TranscriptFailed,
    TranscriptLanguage,
)
from speech_client.wire import Language, RefusalCode
from tests.support.fakes_support import sine_pcm
from tests.support.hear_support import HEARD_TEXT, FakeHearing, hear_runtime
from tests.support.hearing_service_support import service_of
from tests.support.script_support import script_module

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from proof_identity import ProofIdentity
    from proof_minting import Minting
    from speech_client.realtime_wire import ServerEvent

CLIENT = script_module("hearing_client")
QUESTION = sine_pcm(0.5, LIVE_INPUT_RATE)
COMPLETED = TranscriptCompleted(
    item_id="i",
    transcript=HEARD_TEXT,
    usage=DurationUsage(seconds=1.0),
    languages=(TranscriptLanguage(code=Language.DE),),
)


def test_a_recording_is_heard_from_the_file_route(
    minting: Minting, identity: ProofIdentity
) -> None:
    async def scenario() -> tuple[float, str]:
        async with service_of(
            hear_runtime(FakeHearing()), minting, identity
        ) as service:
            return await asyncio.to_thread(
                CLIENT.hear_file, service, Language.DE, QUESTION
            )

    seconds, text = asyncio.run(scenario())

    assert (seconds > 0, text) == (True, HEARD_TEXT)


def test_a_service_without_hearing_ends_the_file_proof_naming_its_answer(
    minting: Minting, identity: ProofIdentity
) -> None:
    async def scenario() -> None:
        async with service_of(hear_runtime(None), minting, identity) as service:
            await asyncio.to_thread(CLIENT.hear_file, service, Language.DE, QUESTION)

    ended = scenario()
    answered = f"the file route answered {HTTPStatus.SERVICE_UNAVAILABLE.value}"

    with pytest.raises(SystemExit, match=answered):
        asyncio.run(ended)


def test_a_turn_is_heard_live_and_its_transcript_returned(
    minting: Minting, identity: ProofIdentity
) -> None:
    async def scenario() -> object:
        async with service_of(
            hear_runtime(FakeHearing()), minting, identity
        ) as service:
            return await CLIENT.hear_live(service, (Language.DE,), QUESTION)

    turn = asyncio.run(scenario())

    assert turn.transcript == HEARD_TEXT


def test_a_refused_live_session_ends_the_proof_naming_its_refusal(
    minting: Minting, identity: ProofIdentity
) -> None:
    refusal = Refused(int(HTTPStatus.SERVICE_UNAVAILABLE), RefusalCode.UNAVAILABLE)

    async def scenario() -> None:
        async with service_of(hear_runtime(None), minting, identity) as service:
            await CLIENT.hear_live(service, (Language.DE,), QUESTION)

    ended = scenario()

    with pytest.raises(SystemExit, match=re.escape(f"completed: {refusal}")):
        asyncio.run(ended)


async def delivering(events: Sequence[ServerEvent]) -> AsyncIterator[ServerEvent]:
    """Stand in for the events of a session that delivers `events`."""
    for event in events:
        yield event


def test_the_deltas_before_the_final_are_kept_in_order() -> None:
    first = TranscriptDelta(item_id="i", delta="Guten")
    second = TranscriptDelta(item_id="i", delta=" Tag")

    turn = asyncio.run(
        CLIENT.receive_turn(delivering([first, second, COMPLETED]), 1.0, 0.0)
    )

    assert turn is not None
    assert turn.deltas == ("Guten", " Tag")


def test_a_failed_turn_ends_the_proof_naming_its_event() -> None:
    failed = TranscriptFailed(
        item_id="i",
        error=ItemFailure(
            message="m", type="server_error", code=RefusalCode.UNAVAILABLE
        ),
    )

    turn = CLIENT.receive_turn(delivering([failed]), 1.0, 0.0)

    with pytest.raises(SystemExit, match=type(failed).__name__):
        asyncio.run(turn)
