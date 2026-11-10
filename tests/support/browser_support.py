"""The stand-in app a browser drives, under webauth's headers.

It serves its page, the built package and a recording, and relays its hop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import SecretStr
from webauth.config import SessionLifetime, WebAuthConfig, install_web_auth_config
from webauth.middleware import SecurityHeadersMiddleware
from webauth.policies import SecurityHeadersPolicy, default_content_security_policy
from webauth.proxies import TrustedProxies
from webauth.rate_limit import SingleProcessRateLimitBackend

from speech_client.hearing import HearingSession
from speech_client.relay import HOP_FRAME_BYTES, RelayOutcome, relay
from speech_client.wire import HearingLanguage, ServerEvent, TranscriptCompleted
from tests.support.identity_support import valid_pair

if TYPE_CHECKING:
    from collections.abc import Mapping

REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE_BUILD = REPOSITORY / "packages/browser/dist"
PAGE_FILES = REPOSITORY / "tests/fixtures/browser"
HOP_PATH = "/hear"
REFUSING_HOP_PATH = "/refuse"
RECORDING = "fixture.wav"
# An app that records turns the microphone on for its own origin;
# webauth's default turns it off.
MICROPHONE_ON = "camera=(), microphone=(self), geolocation=()"
SESSION_SECONDS = 30
HELD_LIMIT_BYTES = 4 * 1024 * 1024
LOOPBACK = "127.0.0.1"


@dataclass
class AppSide:
    """What the app's side of the one hop of a test saw."""

    tokens: tuple[str, str]
    frames: list[bytes | str] = field(default_factory=list)
    events: list[ServerEvent] = field(default_factory=list)
    app_ended: asyncio.Event = field(default_factory=asyncio.Event)
    completed: asyncio.Event = field(default_factory=asyncio.Event)
    outcome: asyncio.Future[RelayOutcome] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )

    @classmethod
    def new(cls) -> AppSide:
        """A side with a fresh token pair for its session."""
        pair = valid_pair()
        return cls(tokens=(pair.workload, pair.person))

    async def take(self, event: ServerEvent) -> None:
        """Keep a server event the relay hands the app."""
        self.events.append(event)
        if isinstance(event, TranscriptCompleted):
            self.completed.set()


class RecordingSocket:
    """The browser socket as the relay reads it, keeping every frame it reads."""

    def __init__(self, socket: WebSocket, frames: list[bytes | str]) -> None:
        """Record the frames of `socket` into `frames`."""
        self._socket = socket
        self._frames = frames

    async def receive(self) -> Mapping[str, object]:
        """The next message, its frame recorded."""
        message = await self._socket.receive()
        if message["type"] == "websocket.receive":
            self._frames.append(message.get("bytes") or message.get("text") or b"")
        return message

    async def close(self, code: int) -> None:
        """Close the browser socket with `code`."""
        await self._socket.close(code)


def _web_auth_config() -> WebAuthConfig:
    return WebAuthConfig(
        session_secret=SecretStr("stand-in app session secret, 32+ chars"),
        trusted_proxies=TrustedProxies(),
        rate_limits=SingleProcessRateLimitBackend(),
        allowed_hosts_exact=frozenset({LOOPBACK}),
        session_lifetime=SessionLifetime(
            idle_seconds=SESSION_SECONDS, absolute_max_age_seconds=SESSION_SECONDS
        ),
        cookie_secure=False,
    )


def stand_in_app(service_url: str, side: AppSide, recordings: Path) -> FastAPI:
    """The app: the page, the package, the recording in `recordings`, and the relay."""
    app = FastAPI()
    install_web_auth_config(app, _web_auth_config())
    app.add_middleware(
        SecurityHeadersMiddleware,
        policy=SecurityHeadersPolicy(
            content_security_policy=default_content_security_policy(),
            permissions_policy=MICROPHONE_ON,
        ),
    )

    @app.get(f"/{RECORDING}")
    async def recording() -> FileResponse:
        return FileResponse(recordings / RECORDING, media_type="audio/wav")

    @app.websocket(HOP_PATH)
    async def hear(browser: WebSocket) -> None:
        await browser.accept()
        session = HearingSession(
            service_url,
            workload_token=side.tokens[0],
            person_token=side.tokens[1],
            language=HearingLanguage.EN,
            deadline_seconds=SESSION_SECONDS,
            held_limit_bytes=HELD_LIMIT_BYTES,
        )
        side.outcome.set_result(
            await relay(
                RecordingSocket(browser, side.frames),
                session,
                app_ended=side.app_ended,
                on_event=side.take,
                frame_limit_bytes=HOP_FRAME_BYTES,
            )
        )

    @app.websocket(REFUSING_HOP_PATH)
    async def refuse(browser: WebSocket) -> None:
        await browser.close()

    app.mount("/dist", StaticFiles(directory=PACKAGE_BUILD), name="package")
    app.mount("/", StaticFiles(directory=PAGE_FILES), name="page")
    return app
