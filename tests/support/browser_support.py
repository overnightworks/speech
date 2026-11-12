"""The stand-in app a browser drives, under webauth's headers.

It serves its pages, the built package and recordings, relays its hop and answers
sounds: one it has the service speak, and test-written ones it serves whole, holds
until released, or cuts off mid-body. Besides the hop that sends nothing, it has one
whose route writes messages of its own on the browser socket, and a worklet file it
holds until released.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from playwright.async_api import async_playwright
from pydantic import SecretStr
from webauth.config import SessionLifetime, WebAuthConfig, install_web_auth_config
from webauth.middleware import SecurityHeadersMiddleware
from webauth.policies import SecurityHeadersPolicy, default_content_security_policy
from webauth.proxies import TrustedProxies
from webauth.rate_limit import SingleProcessRateLimitBackend

from speech_client.hearing import HearingSession
from speech_client.relay import HOP_FRAME_BYTES, relay
from speech_client.speak import Spoken, speak
from speech_client.wire import (
    Language,
    ResponseFormat,
    ServerEvent,
    TranscriptCompleted,
    TranscriptDelta,
    VoiceId,
)
from speech_service.pcm import wav_header
from tests.support.fakes_support import SPEAK_SAMPLE_RATE, sine_pcm
from tests.support.identity_support import valid_pair
from tests.support.page_coverage_support import PageCoverage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping

    from playwright.async_api import Page
    from starlette.types import Receive, Scope, Send

REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE_BUILD = REPOSITORY / "packages/browser/dist"
PAGE_FILES = REPOSITORY / "tests/fixtures/browser"
HOP_PATH = "/hear"
MESSAGING_HOP_PATH = "/hear-messaging"
HELD_WORKLET_PATH = "/held-worklet.js"
REFUSING_HOP_PATH = "/refuse"
VOICE_PATH = "/voice"
SOUNDS_PATH = "/sounds"
HELD_PATH = "/held"
CUT_PATH = "/cut"
CHROMIUM_FLAGS = (
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    "--autoplay-policy=document-user-activation-required",
)
RECORDING = "fixture.wav"
# An app that records turns the microphone on for its own origin;
# webauth's default turns it off.
MICROPHONE_ON = "camera=(), microphone=(self), geolocation=()"
SESSION_SECONDS = 30
HELD_LIMIT_BYTES = 4 * 1024 * 1024
LOOPBACK = "127.0.0.1"
WAV_MEDIA_TYPE = "audio/wav"
SPOKEN_TEXT = "Hello"
SPEAK_SECONDS = 10.0
SPEAK_CAP_BYTES = 1024 * 1024


@dataclass
class AppSide:
    """What the app's side of the one hop of a test saw."""

    tokens: tuple[str, str]
    frames: list[bytes | str] = field(default_factory=list)
    events: list[ServerEvent] = field(default_factory=list)
    app_ended: asyncio.Event = field(default_factory=asyncio.Event)
    completed: asyncio.Event = field(default_factory=asyncio.Event)
    requests: list[str] = field(default_factory=list)
    spoken: list[bytes] = field(default_factory=list)
    held_release: asyncio.Event = field(default_factory=asyncio.Event)
    held_aborted: asyncio.Event = field(default_factory=asyncio.Event)
    arrivals: dict[str, asyncio.Event] = field(default_factory=dict)
    # What the messaging hop's route writes on the browser socket, in order: its
    # transcripts join the test's own, and `None` ends the hop once the rest is sent.
    outbox: asyncio.Queue[str | bytes | None] = field(default_factory=asyncio.Queue)
    worklet_release: asyncio.Event = field(default_factory=asyncio.Event)

    @classmethod
    def new(cls) -> AppSide:
        """A side with a fresh token pair for its session."""
        pair = valid_pair()
        return cls(tokens=(pair.workload, pair.person))

    def record_request(self, path: str) -> None:
        """Note that the app's route `path` was asked."""
        self.requests.append(path)
        self.arrivals.setdefault(path, asyncio.Event()).set()

    async def arrival(self, path: str) -> None:
        """Wait until the app's route `path` has been asked."""
        await self.arrivals.setdefault(path, asyncio.Event()).wait()

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


class HeldResponse(Response):
    """A WAV answer whose headers leave at once and whose body waits for the test.

    The body goes out once `side.held_release` is set; a client that leaves first sets
    `side.held_aborted`.
    """

    def __init__(self, body: bytes, side: AppSide) -> None:
        """Hold `body` for `side`."""
        super().__init__(body, media_type=WAV_MEDIA_TYPE)
        self._side = side

    async def __call__(self, _scope: Scope, receive: Receive, send: Send) -> None:
        """Answer once released, or note that the client left."""
        await send(
            {
                "type": "http.response.start",
                "status": HTTPStatus.OK,
                "headers": self.raw_headers,
            }
        )
        released = asyncio.create_task(self._side.held_release.wait())
        left = asyncio.create_task(_until_disconnect(receive))
        done, pending = await asyncio.wait(
            {released, left}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if released in done:
            await send({"type": "http.response.body", "body": self.body})
        else:
            self._side.held_aborted.set()


async def _until_disconnect(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


def _install_sounds(
    app: FastAPI, service_url: str, side: AppSide, recordings: Path
) -> None:
    """The routes that answer a sound: spoken by the service, whole, held or cut off."""

    @app.get(VOICE_PATH)
    async def voice(request: Request) -> Response:
        side.record_request(request.url.path)
        outcome = await asyncio.to_thread(
            speak,
            service_url,
            SPOKEN_TEXT,
            VoiceId.THORSTEN,
            response_format=ResponseFormat.WAV,
            workload_token=side.tokens[0],
            person_token=side.tokens[1],
            deadline_seconds=SPEAK_SECONDS,
            cap_bytes=SPEAK_CAP_BYTES,
        )
        if not isinstance(outcome, Spoken):
            raise HTTPException(502)
        side.spoken.append(outcome.audio)
        return Response(outcome.audio, media_type=WAV_MEDIA_TYPE)

    @app.get(f"{SOUNDS_PATH}/{{name}}")
    async def sound(name: str, request: Request) -> FileResponse:
        side.record_request(request.url.path)
        return FileResponse(recordings / name, media_type=WAV_MEDIA_TYPE)

    @app.get(f"{HELD_PATH}/{{name}}")
    async def held(name: str, request: Request) -> Response:
        side.record_request(request.url.path)
        return HeldResponse((recordings / name).read_bytes(), side)

    @app.get(f"{CUT_PATH}/{{name}}")
    async def cut(name: str, request: Request) -> StreamingResponse:
        side.record_request(request.url.path)
        body = (recordings / name).read_bytes()

        async def first_half_then_abort() -> AsyncIterator[bytes]:
            yield body[: len(body) // 2]
            raise ConnectionAbortedError

        return StreamingResponse(first_half_then_abort(), media_type=WAV_MEDIA_TYPE)


async def _write_outbox(browser: WebSocket, side: AppSide) -> None:
    """Write the outbox on the browser socket; `None` ends the hop."""
    while (item := await side.outbox.get()) is not None:
        if isinstance(item, bytes):
            await browser.send_bytes(item)
        else:
            await browser.send_text(item)
    side.app_ended.set()


def _install_messaging(
    app: FastAPI, session_of: Callable[[], HearingSession], side: AppSide
) -> None:
    """The hop whose route writes messages of its own; a worklet file the test holds."""

    async def forward_transcript(event: ServerEvent) -> None:
        await side.take(event)
        if isinstance(event, TranscriptDelta | TranscriptCompleted):
            side.outbox.put_nowait(event.model_dump_json())

    @app.websocket(MESSAGING_HOP_PATH)
    async def hear_messaging(browser: WebSocket) -> None:
        await browser.accept()
        writing = asyncio.create_task(_write_outbox(browser, side))
        try:
            await relay(
                RecordingSocket(browser, side.frames),
                session_of(),
                app_ended=side.app_ended,
                on_event=forward_transcript,
                frame_limit_bytes=HOP_FRAME_BYTES,
            )
        finally:
            writing.cancel()

    @app.get(HELD_WORKLET_PATH)
    async def held_worklet(request: Request) -> FileResponse:
        side.record_request(request.url.path)
        await side.worklet_release.wait()
        return FileResponse(
            PACKAGE_BUILD / "capture-worklet.js", media_type="text/javascript"
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
        return FileResponse(recordings / RECORDING, media_type=WAV_MEDIA_TYPE)

    def new_session() -> HearingSession:
        return HearingSession(
            service_url,
            workload_token=side.tokens[0],
            person_token=side.tokens[1],
            languages=(Language.EN,),
            deadline_seconds=SESSION_SECONDS,
            held_limit_bytes=HELD_LIMIT_BYTES,
        )

    @app.websocket(HOP_PATH)
    async def hear(browser: WebSocket) -> None:
        await browser.accept()
        await relay(
            RecordingSocket(browser, side.frames),
            new_session(),
            app_ended=side.app_ended,
            on_event=side.take,
            frame_limit_bytes=HOP_FRAME_BYTES,
        )

    @app.websocket(REFUSING_HOP_PATH)
    async def refuse(browser: WebSocket) -> None:
        await browser.close()

    _install_messaging(app, new_session, side)
    _install_sounds(app, service_url, side, recordings)
    app.mount("/dist", StaticFiles(directory=PACKAGE_BUILD), name="package")
    app.mount("/", StaticFiles(directory=PAGE_FILES), name="page")
    return app


def write_sound(path: Path, seconds: float) -> None:
    """Write a WAV of a tone `seconds` long, in the voice's format."""
    pcm = sine_pcm(seconds, SPEAK_SAMPLE_RATE)
    path.write_bytes(wav_header(SPEAK_SAMPLE_RATE, len(pcm)) + pcm)


@asynccontextmanager
async def chromium_page() -> AsyncIterator[tuple[Page, PageCoverage]]:
    """A Chromium page under the autoplay policy, with its coverage if asked for."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(args=CHROMIUM_FLAGS)
        try:
            page = await browser.new_page()
            coverage = PageCoverage.from_environment()
            await coverage.watch(page)
            yield page, coverage
            await coverage.write()
        finally:
            await browser.close()
