"""The serve loop over a fake runtime, and the stop installer a scenario records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from speech_service.__main__ import StopPolicy, bind_listener, serve_on
from tests.support.identity_support import valid_callers

if TYPE_CHECKING:
    import asyncio
    import socket

    from speech_service.callers import SpeechCallers
    from speech_service.config import Settings
    from speech_service.service import Runtime


class RecordedStops:
    """A signal installer that keeps the stop events the loop hands it.

    `stop` ends the loop as its SIGTERM handler does.
    """

    def __init__(self) -> None:
        """Start with no loop installed."""
        self._events: list[asyncio.Event] = []

    def __call__(self, stopping: asyncio.Event) -> None:
        """Note the event the loop stops on."""
        self._events.append(stopping)

    def stop(self) -> None:
        """Stop the loop that installed."""
        self._events[0].set()


async def serve_runtime(
    settings: Settings,
    runtime: Runtime,
    stop: StopPolicy | None = None,
    *,
    listener: socket.socket | None = None,
    callers: SpeechCallers | None = None,
) -> int:
    """Serve `runtime` as `serve` does: on the settings' address, then close it."""
    with listener or bind_listener(settings) as bound:
        return await serve_on(
            settings,
            callers or valid_callers(),
            bound,
            runtime,
            stop or StopPolicy(),
        )
