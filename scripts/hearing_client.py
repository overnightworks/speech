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
from pydantic import ValidationError
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

import hearing_figures as figures
from hearing_figures import LiveTurn
from proof_identity import Minting, mint_pair
from speech_runtime_client.wire import (
    MODEL,
    REALTIME_INTENT,
    REALTIME_INTENT_PARAM,
    REALTIME_PATH,
    TRANSCRIPTIONS_PATH,
    AudioAppend,
    ErrorEvent,
    HearingLanguage,
    Refusal,
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
    parse_server_event,
    token_headers,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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
        event = parse_server_event(message)
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
        additional_headers=minted_headers(service.minting),
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


async def upgrade_refusal(
    service: Service,
    open_connection: Callable[
        ..., contextlib.AbstractAsyncContextManager[object]
    ] = connect,
) -> UpgradeRefusal | None:
    """The answer that refuses a live upgrade, or `None` if it was accepted."""
    try:
        async with open_connection(
            live_url(service.base_url),
            additional_headers=minted_headers(service.minting),
        ):
            return None
    except InvalidStatus as refused:
        answer = refused.response
        return UpgradeRefusal(
            answer.status_code, refusal_code(answer.status_code, answer.body)
        )
