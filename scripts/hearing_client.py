"""The hearing stage proof's clients: a recording from a file, a turn live."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

import httpx2
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

import hearing_figures as figures
from hearing_figures import LiveTurn
from proof_identity import Minting, mint_pair
from speech_runtime_client.wire import (
    AUTHORIZATION_HEADER,
    MODEL,
    REALTIME_INTENT,
    REALTIME_INTENT_PARAM,
    REALTIME_PATH,
    TRANSCRIPTIONS_PATH,
    WORKLOAD_TOKEN_HEADER,
    AudioAppend,
    ErrorEvent,
    HearingLanguage,
    SessionAudio,
    SessionInput,
    SessionUpdate,
    TranscriptCompleted,
    TranscriptDelta,
    TranscriptFailed,
    Transcription,
    TranscriptionRequest,
    TranscriptionSession,
    TranscriptionSetting,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from websockets.asyncio.client import ClientConnection

    from prove_speech import (
        RunningSpeech,
    )


LIVE_WAIT_SECONDS = 180.0
FILE_WAIT_SECONDS = 300.0


@dataclass(frozen=True)
class Service:
    """A running speech service and what a client needs to call it."""

    base_url: str
    minting: Minting
    running: RunningSpeech


def token_headers(minting: Minting) -> dict[str, str]:
    """The two token headers, minted fresh, without a content type."""
    pair = mint_pair(minting)
    return {
        WORKLOAD_TOKEN_HEADER: pair.workload,
        AUTHORIZATION_HEADER: f"Bearer {pair.person}",
    }


def post_recording(
    service: Service, language: HearingLanguage, pcm: bytes
) -> httpx2.Response:
    """Post one recording to the file route and return the answer, whatever it is."""
    form = TranscriptionRequest(model=MODEL, language=language)
    with httpx2.Client(base_url=service.base_url, timeout=FILE_WAIT_SECONDS) as client:
        return client.post(
            TRANSCRIPTIONS_PATH,
            headers=token_headers(service.minting),
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


def live_url(base_url: str) -> str:
    """The Realtime transcription WebSocket of the service at `base_url`."""
    return (
        f"ws{base_url.removeprefix('http')}{REALTIME_PATH}"
        f"?{REALTIME_INTENT_PARAM}={REALTIME_INTENT}"
    )


def update_event(language: HearingLanguage) -> str:
    """The one `session.update` that sets the language of a session."""
    update = SessionUpdate(
        type="session.update",
        session=TranscriptionSession(
            type="transcription",
            audio=SessionAudio(
                input=SessionInput(
                    transcription=TranscriptionSetting(language=language)
                )
            ),
        ),
    )
    return update.model_dump_json(exclude_none=True)


async def _send_paced(
    socket: ClientConnection, frames: Sequence[bytes], started: float
) -> None:
    """Send each frame when its audio would have been spoken, as a microphone does."""
    for index, frame in enumerate(frames, start=1):
        due = started + index * figures.FRAME_SECONDS
        await asyncio.sleep(max(0.0, due - time.monotonic()))
        append = AudioAppend(
            type="input_audio_buffer.append", audio=base64.b64encode(frame).decode()
        )
        await socket.send(append.model_dump_json(exclude_none=True))


async def _receive_turn(
    socket: ClientConnection, question_seconds: float, started: float
) -> LiveTurn:
    delta_seconds: list[float] = []
    deltas: list[str] = []
    async for message in socket:
        event = figures.parse_live_event(message)
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


async def hear_live(
    service: Service, language: HearingLanguage, pcm: bytes
) -> LiveTurn:
    """Stream one turn at speaking pace and watch for its deltas and its final."""
    question_seconds = figures.seconds_of(pcm)
    frames = figures.frames_with_closing_silence(pcm)
    async with connect(
        live_url(service.base_url),
        additional_headers=token_headers(service.minting),
        open_timeout=LIVE_WAIT_SECONDS,
    ) as socket:
        await socket.send(update_event(language))
        started = time.monotonic()
        sender = asyncio.create_task(_send_paced(socket, frames, started))
        try:
            return await asyncio.wait_for(
                _receive_turn(socket, question_seconds, started),
                question_seconds + LIVE_WAIT_SECONDS,
            )
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, ConnectionClosed):
                await sender


async def upgrade_status(service: Service) -> int | None:
    """The HTTP status that refuses a live upgrade, or `None` if it was accepted."""
    try:
        async with connect(
            live_url(service.base_url),
            additional_headers=token_headers(service.minting),
        ):
            return None
    except InvalidStatus as refused:
        return refused.response.status_code
