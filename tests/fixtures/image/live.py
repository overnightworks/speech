"""Hear one live turn from the proof's caller container; print the outcome as JSON.

The turn arrives on standard input: a JSON object with the service's `base_url`,
the `workload_token` and `person_token`, the `language`, the audio `frames`
(base64), and the seconds between two frames and to wait for the session. The
outcome is the final transcript and the texts of the partials before it, or an
`error` naming the outcome that ended the session.
"""

# ruff: noqa: T201 - this script's output is the outcome the proof reads

import asyncio
import base64
import contextlib
import json
import sys

from speech_client.hearing import HearingSession
from speech_client.wire import HearingLanguage, TranscriptCompleted, TranscriptDelta


async def send_frames(
    session: HearingSession, frames: list[bytes], pace_seconds: float
) -> None:
    """Send each frame after `pace_seconds`, as a microphone would."""
    for frame in frames:
        await asyncio.sleep(pace_seconds)
        session.send(frame)


async def turn(specification: dict[str, object]) -> dict[str, object]:
    """Run the turn; a session that ends before a final is an `error`."""
    frames = [base64.b64decode(frame) for frame in specification["frames"]]
    partials: list[str] = []
    async with HearingSession(
        specification["base_url"],
        workload_token=specification["workload_token"],
        person_token=specification["person_token"],
        language=HearingLanguage(specification["language"]),
        deadline_seconds=specification["wait_seconds"],
        held_limit_bytes=sum(map(len, frames)),
    ) as session:
        sender = asyncio.create_task(
            send_frames(session, frames, specification["pace_seconds"])
        )
        try:
            async for event in session.events():
                if isinstance(event, TranscriptDelta):
                    partials.append(event.delta)
                elif isinstance(event, TranscriptCompleted):
                    return {"final": event.transcript, "partials": partials}
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
    return {"error": type(session.outcome).__name__}


print(json.dumps(asyncio.run(turn(json.load(sys.stdin)))))
