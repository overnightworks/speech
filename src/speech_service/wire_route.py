"""POST /v1/audio/speech, POST /v1/audio/transcriptions and the refusals of the rest."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ValidationError
from python_multipart.exceptions import FormParserError
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from speech_client.wire import (
    API_VERSION,
    AUTHENTICATE_HEADER,
    REALTIME_INTENT,
    REALTIME_INTENT_PARAM,
    REALTIME_PATH,
    RETRY_AFTER_HEADER,
    SPEECH_PATH,
    TRANSCRIPTIONS_PATH,
    Language,
    RefusalCode,
    RequestField,
    SpeechRequest,
    Transcription,
    TranscriptionRequest,
    VoiceId,
    refusal,
)
from speech_service.audio_response import ClosingAudioResponse
from speech_service.failures import log_failure
from speech_service.live_session import LiveSession, poll_forever
from speech_service.recording import (
    RecordingTooLongError,
    UndecodableRecordingError,
    decode_recording,
)
from speech_service.service import audio_chunks

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI
    from starlette.datastructures import FormData
    from starlette.responses import Response
    from starlette.types import Message, Receive, Scope, Send
    from starlette.websockets import WebSocket

    from speech_service.admission_gate import Lease
    from speech_service.callers import SpeechCallers
    from speech_service.config import Settings
    from speech_service.hearing import HearingEngine
    from speech_service.service import Runtime, SpeakingEngine

logger = logging.getLogger(__name__)

_VERSION_PREFIX: Final = "/v{version:int}"
_BEARER_CHALLENGE: Final = "Bearer"
_LEAVE_NOTICE_SECONDS: Final = 1.0


def _refused(
    code: RefusalCode,
    param: RequestField | None = None,
    *,
    headers: dict[str, str] | None = None,
    background: BackgroundTask | None = None,
) -> JSONResponse:
    body = refusal(code, param)
    if code is RefusalCode.UNAUTHORIZED:
        headers = {**(headers or {}), AUTHENTICATE_HEADER: _BEARER_CHALLENGE}
    return JSONResponse(
        body.model_dump(mode="json"),
        status_code=body.status,
        headers=headers,
        background=background,
    )


def _refused_field(
    error: ValidationError, request: type[BaseModel]
) -> RequestField | None:
    """The `request` field the first error names; nothing else of the error leaves."""
    location = error.errors(
        include_url=False, include_context=False, include_input=False
    )[0]["loc"]
    if location and location[0] in request.model_fields:
        return location[0]
    return None


def _parse(raw: bytes) -> SpeechRequest | JSONResponse:
    try:
        return SpeechRequest.model_validate_json(raw)
    except ValidationError as error:
        return _refused(
            RefusalCode.INVALID_REQUEST, _refused_field(error, SpeechRequest)
        )


async def _read_within(request: Request, limit: int) -> bytes | None:
    """The body, or `None` once it exceeds `limit` bytes; the rest is never read."""
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            return None
    return bytes(body)


class _SpeechRoute:
    def __init__(
        self, runtime: Runtime, settings: Settings, callers: SpeechCallers
    ) -> None:
        self._runtime = runtime
        self._settings = settings
        self._callers = callers

    async def speak(self, request: Request) -> Response:
        admitted = await self._callers.admit(request.headers)
        if isinstance(admitted, RefusalCode):
            return _refused(admitted)
        accepted = self._runtime.clock.now()
        parsed = await self._checked(request)
        if isinstance(parsed, JSONResponse):
            return parsed
        engine = await self._engine(parsed.voice, accepted)
        if isinstance(engine, JSONResponse):
            return engine
        speaking, deadline = engine
        return await self._spoken(parsed, speaking, deadline)

    async def _engine(
        self, voice: VoiceId, accepted: float
    ) -> tuple[SpeakingEngine, float] | JSONResponse:
        """The engine of `voice` and the deadline of its busy wait, or the refusal.

        The busy wait counts from the accepted tokens, or from the end of the
        load the request waited for.
        """
        if voice not in self._runtime.slots:
            return _refused(RefusalCode.UNAVAILABLE, "voice")
        speaking = self._runtime.capture(voice)
        if speaking is not None:
            return speaking, accepted + self._settings.busy_wait_seconds
        speaking = await self._loaded(voice)
        if speaking is None:
            return self._unavailable()
        return speaking, self._runtime.clock.now() + self._settings.busy_wait_seconds

    async def _spoken(
        self,
        request: SpeechRequest,
        speaking: SpeakingEngine,
        deadline: float,
    ) -> Response:
        """The audio of `request` once its voice's gate is held."""
        voice = request.voice
        lease = await self._runtime.slots[voice].gate.acquire(
            deadline, self._settings.max_waiting_requests
        )
        if self._runtime.capture(voice) is not speaking:
            if lease is not None:
                lease.close()
            return self._unavailable()
        if lease is None:
            return self._busy()
        chunks = audio_chunks(
            speaking,
            request.input,
            request.spoken_language,
            request.instructions,
            request.response_format,
        )
        return ClosingAudioResponse(chunks, lease, request.response_format.media_type)

    async def _loaded(self, voice: VoiceId) -> SpeakingEngine | None:
        """The engine of `voice` once its shared load ends; `None` if none may run."""
        load = self._runtime.load_on_demand(voice)
        if load is None:
            return None
        # `wrap_future` alone would pass this request's cancellation on to the
        # load every waiter shares; the shield keeps it to this request.
        loaded = await asyncio.shield(asyncio.wrap_future(load))
        # The outcome is this request's load's, never that of a retry since.
        if loaded is None or self._runtime.capture(voice) is not loaded:
            return None
        return loaded

    async def _checked(self, request: Request) -> SpeechRequest | JSONResponse:
        raw = await _read_within(request, self._settings.max_body_bytes)
        if raw is None:
            return _refused(RefusalCode.TOO_LARGE)
        parsed = _parse(raw)
        if isinstance(parsed, JSONResponse):
            return parsed
        if len(parsed.input) > self._settings.max_input_characters:
            return _refused(RefusalCode.INVALID_REQUEST, "input")
        return parsed

    def _unavailable(self) -> JSONResponse:
        return _refused(
            RefusalCode.UNAVAILABLE,
            "voice",
            background=BackgroundTask(self._runtime.reconcile_failed_active),
        )

    def _busy(self) -> JSONResponse:
        wait = max(1, math.ceil(self._settings.busy_wait_seconds))
        return _refused(RefusalCode.BUSY, headers={RETRY_AFTER_HEADER: str(wait)})


def _replayed(request: Request, body: bytes) -> Request:
    """A request over the same scope whose body is the already read `body`."""
    messages = iter([{"type": "http.request", "body": body, "more_body": False}])

    def receive() -> Awaitable[Message]:
        return asyncio.sleep(0, next(messages, {"type": "http.disconnect"}))

    return Request(request.scope, receive)


def _heard(
    recording: bytes, language: Language, max_seconds: float
) -> Callable[[HearingEngine], str]:
    """The job that decodes `recording` and transcribes it."""

    def job(engine: HearingEngine) -> str:
        pcm = decode_recording(recording, max_seconds)
        return engine.transcribe(pcm, (language,)).text

    return job


class _TranscriptionRoute:
    def __init__(
        self, runtime: Runtime, settings: Settings, callers: SpeechCallers
    ) -> None:
        self._runtime = runtime
        self._settings = settings
        self._callers = callers

    async def transcribe(self, request: Request) -> Response:
        refused = await self._hearing_refusal(request)
        if refused is not None:
            return refused
        deadline = (
            self._runtime.hearing.gate.clock.now()
            + self._settings.hearing_busy_wait_seconds
        )
        raw = await _read_within(request, self._settings.max_recording_bytes)
        if raw is None:
            return _refused(RefusalCode.TOO_LARGE)
        lease = await self._runtime.hearing.gate.acquire(
            deadline, self._settings.hearing_max_waiting_requests
        )
        if lease is None:
            return self._busy()
        parsed = await self._parsed_within(lease, _replayed(request, raw))
        if isinstance(parsed, JSONResponse):
            return parsed
        recording, fields = parsed
        return await self._answer(lease, recording, fields.language)

    async def _hearing_refusal(self, request: Request) -> JSONResponse | None:
        admitted = await self._callers.admit(request.headers)
        if isinstance(admitted, RefusalCode):
            return _refused(admitted)
        if self._runtime.hearing.capture() is None:
            return _refused(RefusalCode.UNAVAILABLE)
        return None

    async def _parsed_within(
        self, lease: Lease, request: Request
    ) -> tuple[bytes, TranscriptionRequest] | JSONResponse:
        """The form of `request`, parsed while `lease` is held so one spool exists.

        The lease is closed on a refusal and on any error; on success `_answer`
        hands it on.
        """
        try:
            parsed = await self._parsed(request)
        except BaseException:
            lease.close()
            raise
        if isinstance(parsed, JSONResponse):
            lease.close()
        return parsed

    async def _parsed(
        self, request: Request
    ) -> tuple[bytes, TranscriptionRequest] | JSONResponse:
        try:
            async with request.form(max_files=1) as form:
                upload = form.get("file")
                if not isinstance(upload, UploadFile):
                    return _refused(RefusalCode.INVALID_REQUEST, "file")
                fields = _fields(form)
                if isinstance(fields, JSONResponse):
                    return fields
                return await upload.read(), fields
        except HTTPException, FormParserError:
            return _refused(RefusalCode.INVALID_REQUEST, "file")

    async def _answer(
        self, lease: Lease, recording: bytes, language: Language
    ) -> Response:
        try:
            job = _heard(recording, language, self._settings.max_recording_seconds)
            text = await asyncio.wrap_future(self._runtime.hearing.submit(lease, job))
        except RecordingTooLongError:
            return _refused(RefusalCode.TOO_LARGE, "file")
        except UndecodableRecordingError:
            return _refused(RefusalCode.INVALID_REQUEST, "file")
        except Exception as error:  # noqa: BLE001 - logged by type and never by message
            log_failure(logger, "hearing failed", error)
            return _refused(RefusalCode.UNAVAILABLE)
        return JSONResponse(Transcription(text=text).model_dump())

    def _busy(self) -> JSONResponse:
        wait = max(1, math.ceil(self._settings.hearing_busy_wait_seconds))
        return _refused(RefusalCode.BUSY, headers={RETRY_AFTER_HEADER: str(wait)})


def _fields(form: FormData) -> TranscriptionRequest | JSONResponse:
    text_fields = {name: value for name, value in form.items() if name != "file"}
    try:
        return TranscriptionRequest.model_validate(text_fields)
    except ValidationError as error:
        return _refused(
            RefusalCode.INVALID_REQUEST, _refused_field(error, TranscriptionRequest)
        )


class _RealtimeRoute:
    def __init__(
        self, runtime: Runtime, settings: Settings, callers: SpeechCallers
    ) -> None:
        self._runtime = runtime
        self._settings = settings
        self._callers = callers

    async def serve(self, websocket: WebSocket) -> None:
        place = await self._place(websocket)
        if isinstance(place, JSONResponse):
            await websocket.send_denial_response(place)
            return
        try:
            await websocket.accept()
            session = LiveSession(
                self._runtime.hearing,
                final_wait_seconds=self._settings.hearing_final_wait_seconds,
                send=websocket.send_text,
                close=websocket.close,
            )
            await self._converse(websocket, session)
        finally:
            place.close()

    @staticmethod
    async def _converse(websocket: WebSocket, session: LiveSession) -> None:
        """Serve `session` until the client leaves; its decodes end however it exits."""
        poller = asyncio.create_task(poll_forever(session))
        try:
            try:
                await session.start()
            except RuntimeError:
                if not await _client_has_left(websocket):
                    raise
                return
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                await session.on_message(message.get("text"))
        finally:
            poller.cancel()
            try:
                await asyncio.gather(poller, return_exceptions=True)
            finally:
                await session.on_disconnect()

    async def _place(self, websocket: WebSocket) -> JSONResponse | Lease:
        """The refusal of the upgrade, or the lease on the session's place."""
        admitted = await self._callers.admit(websocket.headers)
        if isinstance(admitted, RefusalCode):
            return _refused(admitted)
        if websocket.query_params.get(REALTIME_INTENT_PARAM) != REALTIME_INTENT:
            return _refused(RefusalCode.INVALID_REQUEST, REALTIME_INTENT_PARAM)
        hearing = self._runtime.hearing
        if hearing.capture() is None:
            return _refused(RefusalCode.UNAVAILABLE)
        place = hearing.open_session(self._settings.max_hearing_sessions)
        if place is None:
            wait = max(1, math.ceil(self._settings.hearing_busy_wait_seconds))
            return _refused(RefusalCode.BUSY, headers={RETRY_AFTER_HEADER: str(wait)})
        return place


async def _client_has_left(websocket: WebSocket) -> bool:
    """Whether the client's disconnect follows a send the server rejected.

    uvicorn drops the accept of a closing transport, rejects the next send with a
    `RuntimeError` and queues the disconnect a loop pass later, so only that
    disconnect tells the leave from a bug. A leave ends the wait at once; only a
    rejected send with the client still there waits it out before it raises.
    """
    try:
        async with asyncio.timeout(_LEAVE_NOTICE_SECONDS):
            message = await websocket.receive()
    except TimeoutError:
        return False
    return message["type"] == "websocket.disconnect"


class _NotARoute:
    """An ASGI app, so the route it serves answers every HTTP method."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["path_params"]["version"] == API_VERSION:
            response = _refused(RefusalCode.NOT_FOUND)
        else:
            response = _refused(RefusalCode.UNSUPPORTED_VERSION)
        await response(scope, receive, send)


async def _deny_upgrade(websocket: WebSocket) -> None:
    """Refuse an upgrade at a path that is no WebSocket route of this service."""
    if websocket.path_params["version"] == API_VERSION:
        denial = _refused(RefusalCode.NOT_FOUND)
    else:
        denial = _refused(RefusalCode.UNSUPPORTED_VERSION)
    await websocket.send_denial_response(denial)


def mount_wire_routes(
    app: FastAPI, runtime: Runtime, settings: Settings, callers: SpeechCallers
) -> None:
    """Mount the speech, transcription and Realtime routes, then the catch-alls."""
    speech = _SpeechRoute(runtime, settings, callers)
    app.add_route(SPEECH_PATH, speech.speak, methods=["POST"], include_in_schema=False)
    hearing = _TranscriptionRoute(runtime, settings, callers)
    app.add_route(
        TRANSCRIPTIONS_PATH,
        hearing.transcribe,
        methods=["POST"],
        include_in_schema=False,
    )
    realtime = _RealtimeRoute(runtime, settings, callers)
    app.router.add_websocket_route(REALTIME_PATH, realtime.serve)
    app.router.add_websocket_route(f"{_VERSION_PREFIX}/{{rest:path}}", _deny_upgrade)
    for path in (_VERSION_PREFIX, f"{_VERSION_PREFIX}/{{rest:path}}"):
        app.add_route(path, _NotARoute(), include_in_schema=False)
