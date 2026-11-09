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

import hearing_figures as figures
from hearing_figures import LiveTurn
from proof_identity import Minting, mint_pair
from speech_client.hearing import HearingSession
from speech_client.wire import (
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
    from collections.abc import AsyncIterator, Sequence

    from prove_speech import (
        RunningSpeech,
    )
    from speech_client.hearing import HearingOutcome
    from speech_client.wire import ServerEvent


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
) -> LiveTurn | None:
    """Read server events until the turn completes; `None` if they end first."""
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
    return None


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
            turn = await receive_turn(session.events(), question_seconds, started)
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
    if turn is None:
        failure = (
            f"the live session ended before the turn was completed: {session.outcome}"
        )
        raise SystemExit(failure)
    return turn


async def hear_upgrade(service: Service) -> HearingOutcome | None:
    """The outcome that ends a live upgrade, or `None` if the service accepted it."""
    pair = mint_pair(service.minting)
    async with HearingSession(
        service.base_url,
        workload_token=pair.workload,
        person_token=pair.person,
        language=HearingLanguage.DE,
        deadline_seconds=LIVE_WAIT_SECONDS,
        held_limit_bytes=0,
    ) as session:
        async for _ in session.events():
            return None
        return session.outcome
