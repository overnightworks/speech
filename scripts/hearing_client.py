"""The hearing stage proof's clients: a recording from a file, a turn live."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

import httpx2
from pydantic import ValidationError
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

import hearing_figures as figures
from hearing_figures import LiveTurn
from proof_identity import Minting, mint_pair
from speech_runtime_client.hearing import (
    HearingSession,
    realtime_url,
)
from speech_runtime_client.wire import (
    MODEL,
    TRANSCRIPTIONS_PATH,
    ErrorEvent,
    HearingLanguage,
    Refusal,
    TranscriptCompleted,
    TranscriptDelta,
    TranscriptFailed,
    Transcription,
    TranscriptionRequest,
    token_headers,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from prove_speech import (
        RunningSpeech,
    )
    from speech_runtime_client.wire import ServerEvent


LIVE_WAIT_SECONDS = 180.0
FILE_WAIT_SECONDS = 300.0


@dataclass(frozen=True)
class Service:
    """A running speech service and what a client needs to call it."""

    base_url: str
    minting: Minting
    running: RunningSpeech


def minted_headers(minting: Minting) -> dict[str, str]:
    """The two token headers, minted fresh, without a content type."""
    pair = mint_pair(minting)
    return token_headers(workload_token=pair.workload, person_token=pair.person)


def refusal_code(status: int, body: bytes) -> str | None:
    """The error code of a 503 answer's contract body; `None` for any other answer."""
    if status != HTTPStatus.SERVICE_UNAVAILABLE:
        return None
    try:
        return Refusal.model_validate_json(body).error.code
    except ValidationError:
        return None


@dataclass(frozen=True)
class UpgradeRefusal:
    """The HTTP answer that refused a live upgrade."""

    status: int
    code: str | None


def post_recording(
    service: Service, language: HearingLanguage, pcm: bytes
) -> httpx2.Response:
    """Post one recording to the file route and return the answer, whatever it is."""
    form = TranscriptionRequest(model=MODEL, language=language)
    with httpx2.Client(base_url=service.base_url, timeout=FILE_WAIT_SECONDS) as client:
        return client.post(
            TRANSCRIPTIONS_PATH,
            headers=minted_headers(service.minting),
            data=form.model_dump(mode="json"),
            files={"file": ("recording.wav", figures.wav_of(pcm), "audio/wav")},
        )


def hear_file(
    service: Service, language: HearingLanguage, pcm: bytes
) -> tuple[float, str]:
    """Hear a recording from a file: seconds until the answer, and its text."""
    start = time.perf_counter()
    response = post_recording(service, language, pcm)
    seconds = time.perf_counter() - start
    if response.status_code != HTTPStatus.OK:
        message = f"the file route answered {response.status_code}"
        raise SystemExit(message)
    return seconds, Transcription.model_validate_json(response.content).text


async def receive_turn(
    events: AsyncIterator[ServerEvent], question_seconds: float, started: float
) -> LiveTurn:
    """Read server events until the turn completes and return its figures."""
    delta_seconds: list[float] = []
    deltas: list[str] = []
    async for event in events:
        elapsed = time.monotonic() - started
        if isinstance(event, TranscriptDelta):
            delta_seconds.append(elapsed)
            deltas.append(event.delta)
        elif isinstance(event, TranscriptCompleted):
            return LiveTurn(
                question_seconds,
                tuple(delta_seconds),
                tuple(deltas),
                elapsed,
                event.transcript,
            )
        elif isinstance(event, TranscriptFailed | ErrorEvent):
            failure = f"the live session failed with {type(event).__name__}"
            raise SystemExit(failure)
    failure = "the live session closed before the turn was completed"
    raise SystemExit(failure)


async def _send_paced(session: HearingSession, frames: Sequence[bytes]) -> None:
    """Send each frame when its audio would have been spoken, as a microphone does."""
    started = time.monotonic()
    for index, frame in enumerate(frames, start=1):
        due = started + index * figures.FRAME_SECONDS
        await asyncio.sleep(max(0.0, due - time.monotonic()))
        session.send(frame)


async def hear_live(
    service: Service, language: HearingLanguage, pcm: bytes
) -> LiveTurn:
    """Stream one turn at speaking pace and watch for its deltas and its final."""
    question_seconds = figures.seconds_of(pcm)
    frames = figures.frames_with_closing_silence(pcm)
    pair = mint_pair(service.minting)
    async with HearingSession(
        service.base_url,
        workload_token=pair.workload,
        person_token=pair.person,
        language=language,
        deadline_seconds=question_seconds + LIVE_WAIT_SECONDS,
        held_limit_bytes=sum(map(len, frames)),
    ) as session:
        started = time.monotonic()
        sender = asyncio.create_task(_send_paced(session, frames))
        try:
            return await receive_turn(session.events(), question_seconds, started)
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender


async def upgrade_refusal(
    service: Service,
    open_connection: Callable[
        ..., contextlib.AbstractAsyncContextManager[object]
    ] = connect,
) -> UpgradeRefusal | None:
    """The answer that refuses a live upgrade, or `None` if it was accepted."""
    try:
        async with open_connection(
            realtime_url(service.base_url),
            additional_headers=minted_headers(service.minting),
        ):
            return None
    except InvalidStatus as refused:
        answer = refused.response
        return UpgradeRefusal(
            answer.status_code, refusal_code(answer.status_code, answer.body)
        )
