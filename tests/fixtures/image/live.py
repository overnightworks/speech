"""Hear one live turn from the proof's caller container; print the outcome as JSON.

The turn arrives on standard input: a JSON object with `url`, `headers`, the
`session_update` text, the audio `frames` (base64) and the seconds between
two frames and to wait for the final. The outcome is the final transcript and the
texts of the partials before it, or an `error` naming what ended the turn.
"""

# ruff: noqa: T201 - this script's output is the outcome the proof reads

import asyncio
import json
import sys

from websockets.asyncio.client import connect

from speech_runtime_client.wire import AudioAppend, TranscriptCompleted, TranscriptDelta

COMPLETED = TranscriptCompleted.model_fields["type"].default
DELTA = TranscriptDelta.model_fields["type"].default


async def send_frames(socket, frames: list[str], pace_seconds: float) -> None:  # noqa: ANN001 - the library's connection type
    """Send each frame after `pace_seconds`, as a microphone would."""
    for frame in frames:
        await asyncio.sleep(pace_seconds)
        append = AudioAppend(type="input_audio_buffer.append", audio=frame)
        await socket.send(append.model_dump_json(exclude_none=True))


async def receive_turn(socket) -> dict[str, object]:  # noqa: ANN001 - the library's connection type
    """The final of the turn, with the texts of the partials before it."""
    partials: list[str] = []
    async for message in socket:
        event = json.loads(message)
        if event["type"] == DELTA:
            partials.append(event["delta"])
        elif event["type"] == COMPLETED:
            return {"final": event["transcript"], "partials": partials}
    return {"error": "ClosedBeforeFinal"}


async def turn(specification: dict[str, object]) -> dict[str, object]:
    """Run the turn; a refused or lost connection is an `error`."""
    try:
        async with connect(
            specification["url"], additional_headers=specification["headers"]
        ) as socket:
            await socket.send(specification["session_update"])
            sender = asyncio.create_task(
                send_frames(
                    socket, specification["frames"], specification["pace_seconds"]
                )
            )
            try:
                return await asyncio.wait_for(
                    receive_turn(socket), specification["wait_seconds"]
                )
            finally:
                sender.cancel()
    except Exception as error:  # noqa: BLE001 - every end of the turn is reported by type
        return {"error": type(error).__name__}


print(json.dumps(asyncio.run(turn(json.load(sys.stdin)))))
