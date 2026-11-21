"""Hear one live turn from the proof's caller container; print the outcome as JSON.

The turn arrives on standard input: a JSON object with the service's `base_url`,
the `workload_token` and `person_token`, the `languages`, the audio `frames`
(base64), and the seconds between two frames and to wait for the session. The
outcome is the final transcript, the language it names and the texts of the
partials before it, or an `error` naming the outcome that ended the session.

With `sound_detection` the session asks for sounds. With `settle_seconds` the turn
is a listening turn instead: it ends that long after the last frame was sent, and
its outcome is the `sounds` the session heard, each with its name and positions.
"""

# ruff: noqa: T201 - this script's output is the outcome the proof reads

import asyncio
import base64
import contextlib
import json
import sys

from speech_client.hearing import HearingSession, SessionTokens
from speech_client.realtime_wire import (
    SoundDetected,
    TranscriptCompleted,
    TranscriptDelta,
)
from speech_client.wire import Language


async def send_frames(
    session: HearingSession, frames: list[bytes], pace_seconds: float
) -> None:
    """Send each frame after `pace_seconds`, as a microphone would."""
    for frame in frames:
        await asyncio.sleep(pace_seconds)
        session.send(frame)


async def listen(
    session: HearingSession, sender: asyncio.Task[None], settle_seconds: float
) -> dict[str, object]:
    """Collect the sounds of the session until it has settled after the frames."""
    sounds: list[dict[str, object]] = []

    async def collect() -> None:
        sounds.extend(
            [
                {
                    "sound": event.sound.value,
                    "audio_start_ms": event.audio_start_ms,
                    "audio_end_ms": event.audio_end_ms,
                }
                async for event in session.events()
                if isinstance(event, SoundDetected)
            ]
        )

    async def settle() -> None:
        await sender
        await asyncio.sleep(settle_seconds)

    collecting = asyncio.create_task(collect())
    settling = asyncio.create_task(settle())
    await asyncio.wait({collecting, settling}, return_when=asyncio.FIRST_COMPLETED)
    ended_early = collecting.done()
    for task in (collecting, settling):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    if ended_early:
        return {"error": type(session.outcome).__name__}
    return {"sounds": sounds}


async def turn(specification: dict[str, object]) -> dict[str, object]:
    """Run the turn; a session that ends before a final is an `error`."""
    frames = [base64.b64decode(frame) for frame in specification["frames"]]
    partials: list[str] = []

    async def tokens() -> SessionTokens:
        return SessionTokens(
            specification["workload_token"], specification["person_token"]
        )

    async with HearingSession(
        specification["base_url"],
        token_source=tokens,
        languages=tuple(map(Language, specification["languages"])),
        deadline_seconds=specification["wait_seconds"],
        held_limit_bytes=sum(map(len, frames)),
        sound_detection=specification["sound_detection"],
    ) as session:
        sender = asyncio.create_task(
            send_frames(session, frames, specification["pace_seconds"])
        )
        try:
            if specification["settle_seconds"] is not None:
                return await listen(session, sender, specification["settle_seconds"])
            async for event in session.events():
                if isinstance(event, TranscriptDelta):
                    partials.append(event.delta)
                elif isinstance(event, TranscriptCompleted):
                    return {
                        "final": event.transcript,
                        "language": event.languages[0].code.value,
                        "partials": partials,
                    }
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
    return {"error": type(session.outcome).__name__}


print(json.dumps(asyncio.run(turn(json.load(sys.stdin)))))
