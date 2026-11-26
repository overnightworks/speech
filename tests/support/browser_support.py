"""The stand-in app a browser drives, under webauth's headers.

It serves its pages, the built package and recordings, relays its hop and answers
sounds: one it has the service speak, and test-written ones it serves whole, holds
until released, or cuts off mid-body. Besides the hop that sends nothing, it has one
whose route writes messages of its own on the browser socket, and a worklet file it
holds until released.
"""

from __future__ import annotations

import asyncio
import math
import os
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from playwright.async_api import async_playwright
from pydantic import SecretStr
from webauth.admission import AdmissionConfig, NoAdmins
from webauth.config import SessionLifetime, WebAuthConfig
from webauth.oidc import OidcClientConfig
from webauth.policies import (
    CsrfPolicy,
    SecurityHeadersPolicy,
    default_content_security_policy,
)
from webauth.proxies import TrustedProxies
from webauth.rate_limit import SingleProcessRateLimitBackend
from webauth.sign_in_routes import SignInPaths
from webauth.single_process_stores import SingleProcessStores
from webauth.web_auth import (
    SignInPages,
    WebAuth,
    WebAuthSettings,
    WebAuthStores,
)

from speech_client.hearing import HearingSession, SessionTokens
from speech_client.realtime_wire import (
    ServerEvent,
    TranscriptCompleted,
    TranscriptDelta,
)
from speech_client.relay import HOP_FRAME_BYTES, relay
from speech_client.speak import Spoken, speak
from speech_client.wire import Language, ResponseFormat, VoiceId
from speech_service.config import app_group
from speech_service.pcm import BYTES_PER_SAMPLE, wav_header
from tests.support.fakes_support import SPEAK_SAMPLE_RATE, sine_pcm
from tests.support.identity_support import ISSUER, valid_pair
from tests.support.page_coverage_support import PageCoverage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping

    from playwright.async_api import Page
    from starlette.types import Receive, Scope, Send
    from webauth.ports import AccountEvent

REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE_BUILD = REPOSITORY / "packages/browser/dist"
PAGE_FILES = REPOSITORY / "tests/fixtures/browser"
HOP_PATH = "/hear"
MESSAGING_HOP_PATH = "/hear-messaging"
HELD_WORKLET_PATH = "/held-worklet.js"
REFUSING_HOP_PATH = "/refuse"
VOICE_PATH = "/voice"
SOUNDS_PATH = "/sounds"
UNSTREAMED_PATH = "/unstreamed"
HELD_PATH = "/held"
STREAMED_PATH = "/streamed"
STREAMED_LEAD_BYTES = 16 * 1024
MARKER_NAME = "marker.wav"
# The marker sound's tone, gap and short tone: the gap places its position.
MARKER_SECONDS = (2.0, 1.0, 0.5)
HEADED_PATH = "/headed"
# What the headed route sends before it holds the samples back: the whole WAV header.
HEADED_LEAD_BYTES = len(wav_header(SPEAK_SAMPLE_RATE))
CUT_PATH = "/cut"
CUT_LATE_PATH = "/cutlate"
CHROMIUM_FLAGS = (
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    "--autoplay-policy=document-user-activation-required",
)
# The factor by which the page's CPU is slowed, if the environment names one.
CPU_THROTTLE_VARIABLE = "SPEECH_BROWSER_CPU_THROTTLE"
RECORDING = "fixture.wav"
# An app that records turns the microphone on for its own origin;
# webauth's default turns it off.
MICROPHONE_ON = "camera=(), microphone=(self), geolocation=()"
SESSION_SECONDS = 30
HELD_LIMIT_BYTES = 4 * 1024 * 1024
LOOPBACK = "127.0.0.1"
WAV_MEDIA_TYPE = "audio/wav"
# A media type the player does not stream, so it reads the answer whole and decodes it.
UNSTREAMED_MEDIA_TYPE = "application/octet-stream"
SPOKEN_TEXT = "Hello"
SPEAK_SECONDS = 10.0
SPEAK_CAP_BYTES = 1024 * 1024


@dataclass
class AppSide:
    """What the app's side of the one hop of a test saw."""

    tokens: tuple[str, str]
    frames: list[bytes | str] = field(default_factory=list)
    # The texts the messaging hop's `on_text` took from the page, in order.
    texts: list[str] = field(default_factory=list)
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

    async def token_source(self) -> SessionTokens:
        """This side's tokens as the token source of its hearing session."""
        return SessionTokens(*self.tokens)

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
            frame = message.get("bytes")
            self._frames.append(frame if frame is not None else message["text"])
        return message

    async def close(self, code: int) -> None:
        """Close the browser socket with `code`."""
        await self._socket.close(code)


def _web_auth_config() -> WebAuthConfig:
    return WebAuthConfig(
        session_secret=SecretStr("stand-in app session secret, 32+ chars"),
        rate_limits=SingleProcessRateLimitBackend(),
        allowed_hosts_exact=frozenset({LOOPBACK}),
        session_lifetime=SessionLifetime(
            idle_seconds=SESSION_SECONDS, absolute_max_age_seconds=SESSION_SECONDS
        ),
        cookie_secure=False,
    )


class _NoAudit:
    """An audit sink that keeps nothing: the stand-in app's sign-in is never driven."""

    def account_event(self, event: AccountEvent) -> None:
        """Drop `event`."""


def _stand_in_auth() -> WebAuth:
    """The stand-in app's webauth host, over an issuer no test ever calls."""
    stores = SingleProcessStores(audit=_NoAudit())
    client = OidcClientConfig(
        issuer=ISSUER,
        authorization_endpoint=f"{ISSUER}/auth",
        token_endpoint=f"{ISSUER}/token",
        jwks_url=f"{ISSUER}/certs",
        end_session_endpoint=f"{ISSUER}/logout",
        client_id="stand-in",
        client_secret=SecretStr("stand-in app client secret"),
        redirect_uri=f"http://{LOOPBACK}/auth/callback",
        post_logout_redirect_uri=f"http://{LOOPBACK}/",
        leeway_seconds=0,
    )
    return WebAuth.build(
        WebAuthSettings(
            config=_web_auth_config(),
            client=client,
            sign_in_paths=SignInPaths(start="/login/oidc", callback="/auth/callback"),
            keeps_id_token=False,
            max_sessions=None,
            security_headers=SecurityHeadersPolicy(
                content_security_policy=default_content_security_policy(),
                permissions_policy=MICROPHONE_ON,
            ),
            csrf_origin=CsrfPolicy(),
        ),
        stores=WebAuthStores(
            sign_in=stores.transaction, sessions=lambda: nullcontext(stores.sessions)
        ),
        groups=AdmissionConfig(
            bound_issuer=ISSUER,
            login_group=app_group("stand-in"),
            admin_group=NoAdmins.IN_THIS_APP,
        ),
        pages=SignInPages(
            refused=lambda _request, _refusal: Response(
                status_code=HTTPStatus.FORBIDDEN
            ),
            logout_page_text=None,
            ui_language=lambda _request: None,
        ),
        front=TrustedProxies(),
    )


class HeldResponse(Response):
    """A WAV answer whose headers leave at once and whose body waits for the test.

    The first `lead` bytes of the body leave with the headers. The rest goes out once
    `side.held_release` is set; a client that leaves first sets `side.held_aborted`.
    """

    def __init__(self, body: bytes, side: AppSide, lead: int = 0) -> None:
        """Hold `body` for `side`, after sending its first `lead` bytes."""
        super().__init__(body, media_type=WAV_MEDIA_TYPE)
        self._side = side
        self._lead = lead

    async def __call__(self, _scope: Scope, receive: Receive, send: Send) -> None:
        """Answer once released, or note that the client left."""
        await send(
            {
                "type": "http.response.start",
                "status": HTTPStatus.OK,
                "headers": self.raw_headers,
            }
        )
        if self._lead:
            await send(
                {
                    "type": "http.response.body",
                    "body": self.body[: self._lead],
                    "more_body": True,
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
            await send({"type": "http.response.body", "body": self.body[self._lead :]})
        else:
            self._side.held_aborted.set()


async def _until_disconnect(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


def _install_sounds(
    app: FastAPI, service_url: str, side: AppSide, recordings: Path
) -> None:
    """The routes that answer a sound, spoken or test-written."""

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

    @app.get(f"{UNSTREAMED_PATH}/{{name}}")
    async def unstreamed(name: str, request: Request) -> FileResponse:
        side.record_request(request.url.path)
        return FileResponse(recordings / name, media_type=UNSTREAMED_MEDIA_TYPE)

    @app.get(f"{HELD_PATH}/{{name}}")
    async def held(name: str, request: Request) -> Response:
        side.record_request(request.url.path)
        return HeldResponse((recordings / name).read_bytes(), side)

    @app.get(f"{STREAMED_PATH}/{{name}}")
    async def streamed(name: str, request: Request) -> Response:
        side.record_request(request.url.path)
        return HeldResponse((recordings / name).read_bytes(), side, lead=_lead_of(name))

    @app.get(f"{HEADED_PATH}/{{name}}")
    async def headed(name: str, request: Request) -> Response:
        side.record_request(request.url.path)
        return HeldResponse(
            (recordings / name).read_bytes(), side, lead=HEADED_LEAD_BYTES
        )

    _install_cut_sounds(app, side, recordings)


def _lead_of(name: str) -> int:
    """The bytes a streamed sound sends before its body is held back.

    The marker sound's lead is its whole first tone: a flow that holds it while the
    tone plays is then not at the mercy of how late the page comes.
    """
    if name != MARKER_NAME:
        return STREAMED_LEAD_BYTES
    tone_samples = int(MARKER_SECONDS[0] * SPEAK_SAMPLE_RATE)
    return HEADED_LEAD_BYTES + tone_samples * BYTES_PER_SAMPLE


def _install_cut_sounds(app: FastAPI, side: AppSide, recordings: Path) -> None:
    """The routes that answer a sound and break the connection off."""

    @app.get(f"{CUT_PATH}/{{name}}")
    async def cut(name: str, request: Request) -> StreamingResponse:
        side.record_request(request.url.path)
        body = (recordings / name).read_bytes()

        async def first_half_then_abort() -> AsyncIterator[bytes]:
            yield body[: len(body) // 2]
            raise ConnectionAbortedError

        return StreamingResponse(first_half_then_abort(), media_type=WAV_MEDIA_TYPE)

    @app.get(f"{CUT_LATE_PATH}/{{name}}")
    async def cut_late(name: str, request: Request) -> StreamingResponse:
        side.record_request(request.url.path)
        body = (recordings / name).read_bytes()

        async def lead_then_abort_once_released() -> AsyncIterator[bytes]:
            yield body[:STREAMED_LEAD_BYTES]
            await side.held_release.wait()
            raise ConnectionAbortedError

        return StreamingResponse(
            lead_then_abort_once_released(), media_type=WAV_MEDIA_TYPE
        )


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
                on_text=side.texts.append,
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
    auth = _stand_in_auth()
    app = FastAPI()

    @app.get(f"/{RECORDING}")
    async def recording() -> FileResponse:
        return FileResponse(recordings / RECORDING, media_type=WAV_MEDIA_TYPE)

    def new_session() -> HearingSession:
        return HearingSession(
            service_url,
            token_source=side.token_source,
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
    auth.install(app, bearer=None)
    return app


def write_sound(path: Path, seconds: float) -> None:
    """Write a WAV of a tone `seconds` long, in the voice's format."""
    pcm = sine_pcm(seconds, SPEAK_SAMPLE_RATE)
    path.write_bytes(wav_header(SPEAK_SAMPLE_RATE, len(pcm)) + pcm)


def write_marker_sound(
    path: Path, tone_seconds: float, gap_seconds: float, tail_seconds: float
) -> None:
    """Write a WAV of a tone, a silent gap and a shorter tone, in the voice's format.

    The gap marks a position that a clock reading can place.
    """
    silence = bytes(int(gap_seconds * SPEAK_SAMPLE_RATE) * BYTES_PER_SAMPLE)
    pcm = (
        sine_pcm(tone_seconds, SPEAK_SAMPLE_RATE)
        + silence
        + sine_pcm(tail_seconds, SPEAK_SAMPLE_RATE)
    )
    path.write_bytes(wav_header(SPEAK_SAMPLE_RATE, len(pcm)) + pcm)


def _cpu_throttle_factor() -> float:
    """The factor the environment slows the page's CPU by; 1 when it names none."""
    factor = float(os.environ.get(CPU_THROTTLE_VARIABLE, "1"))
    if not math.isfinite(factor) or factor < 1:
        message = f"{CPU_THROTTLE_VARIABLE} must be a finite number of at least 1"
        raise ValueError(message)
    return factor


@asynccontextmanager
async def chromium_page() -> AsyncIterator[tuple[Page, PageCoverage]]:
    """A Chromium page under the autoplay policy, with its coverage if asked for.

    `CPU_THROTTLE_VARIABLE` slows the page's CPU by its factor: the flows then run as
    on a loaded machine, where the page's tasks come late and the render thread's not.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(args=CHROMIUM_FLAGS)
        try:
            page = await browser.new_page()
            if (rate := _cpu_throttle_factor()) > 1:
                session = await page.context.new_cdp_session(page)
                await session.send("Emulation.setCPUThrottlingRate", {"rate": rate})
            coverage = PageCoverage.from_environment()
            await coverage.watch(page)
            yield page, coverage
            await coverage.write()
        finally:
            await browser.close()
