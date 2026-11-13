"""Raw-ASGI drivers and WAV readers for the speak wire."""

import asyncio
import itertools
import json
import time
from http import HTTPStatus
from typing import TYPE_CHECKING

from starlette.requests import Request

from speech_client.wire import (
    SPEECH_PATH,
    VOICES,
    Language,
    RefusalCode,
    ResponseFormat,
    VoiceId,
    speech_request_body,
)
from speech_service.config import Settings
from speech_service.control import create_control_app
from speech_service.service import Runtime, RuntimeDependencies
from speech_service.voices import loads_at_start
from tests.support.fakes_support import FakeSpeaking, sine_pcm
from tests.support.identity_support import valid_callers, valid_headers

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from httpx2 import Response as ClientResponse
    from starlette.responses import Response, StreamingResponse

    from speech_service.admission_gate import AdmissionGate, Lease
    from speech_service.callers import SpeechCallers
    from speech_service.service import SpeakingEngine


def info_comment_before_data(wav: bytes) -> str | None:
    """The RIFF INFO ICMT text, found by walking the chunks up to `data`."""
    position = 12
    while position + 8 <= len(wav):
        chunk_id = wav[position : position + 4]
        size = int.from_bytes(wav[position + 4 : position + 8], "little")
        body = wav[position + 8 : position + 8 + size]
        if chunk_id == b"data":
            return None
        if chunk_id == b"LIST" and body[:4] == b"INFO":
            assert size % 2 == 0
            assert body[4:8] == b"ICMT"
            comment_size = int.from_bytes(body[8:12], "little")
            assert body[12 : 12 + comment_size].endswith(b"\0")
            return body[12 : 12 + comment_size - 1].decode("ascii")
        position += 8 + size + size % 2
    return None


ASGI_TIMEOUT_SECONDS = 5.0


def speak_app(
    speaking: SpeakingEngine | None = None,
    settings: Settings | None = None,
    callers: SpeechCallers | None = None,
) -> FastAPI:
    """The private app over a fake runtime; its callers accept `valid_headers()`."""
    runtime = Runtime(
        {VoiceId.THORSTEN: speaking or FakeSpeaking()}, memory_probe=lambda: 0
    )
    return create_control_app(
        settings or Settings(), runtime, callers or valid_callers()
    )


def speak_response(runtime: Runtime, voice: VoiceId = VoiceId.THORSTEN) -> Response:
    """Drive the wire route's endpoint with the default body for `voice`."""
    app = create_control_app(Settings(), runtime, valid_callers())
    route = next(
        route
        for route in app.routes
        if getattr(route, "path", None) == SPEECH_PATH
        and "POST" in getattr(route, "methods", ())
    )
    body = speak_body(voice)

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    return asyncio.run(route.endpoint(Request(speak_scope(), receive)))


def response_body(response: StreamingResponse) -> bytes:
    async def deliver() -> bytes:
        messages: list[dict[str, object]] = []
        never = asyncio.Event()

        async def receive() -> dict[str, object]:
            await never.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        await response(
            {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}},
            receive,
            send,
        )
        return wav_body(messages)

    return asyncio.run(deliver())


def speak_scope(
    headers: Mapping[str, str] | None = None,
    *,
    path: str = SPEECH_PATH,
    content_type: str = "application/json",
) -> dict[str, object]:
    """The scope of a POST to `path`; `headers` default to a valid pair of tokens."""
    sent = valid_headers() if headers is None else headers
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (b"content-type", content_type.encode()),
            *((name.lower().encode(), value.encode()) for name, value in sent.items()),
        ],
        "client": ("test", 123),
        "server": ("test", 80),
    }


def speak_body(
    voice: VoiceId = VoiceId.THORSTEN,
    text: str = "Hallo",
    response_format: ResponseFormat = ResponseFormat.WAV,
    language: Language | None = None,
    instructions: str | None = None,
) -> bytes:
    """The body of a speak request; a voice of several languages gets its first.

    A voice of one language gets a `language` only when the caller names it.
    """
    languages = VOICES[voice].languages
    if language is None and len(languages) > 1:
        language = languages[0]
    return speech_request_body(text, voice, response_format, language, instructions)


def wav_body(messages: list[dict[str, object]]) -> bytes:
    return b"".join(
        message["body"]
        for message in messages
        if message["type"] == "http.response.body"
    )


async def speak_once(
    app: FastAPI, voice: VoiceId = VoiceId.THORSTEN
) -> list[dict[str, object]]:
    """One full, undisturbed request through the raw ASGI boundary."""
    messages: list[dict[str, object]] = []
    sent = False
    no_disconnect = asyncio.Event()  # never set: this request is never disturbed

    async def receive() -> dict[str, object]:
        nonlocal sent
        if not sent:
            sent = True
            return {
                "type": "http.request",
                "body": speak_body(voice),
                "more_body": False,
            }
        await no_disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    await asyncio.wait_for(
        app(speak_scope(), receive, send), timeout=ASGI_TIMEOUT_SECONDS
    )
    return messages


class SpeakCall:
    """One request running as a task, so a test sees its status before its end."""

    def __init__(
        self,
        app: FastAPI,
        body: bytes | None = None,
        scope: dict[str, object] | None = None,
    ) -> None:
        """Start the request now; `scope` defaults to a speak request's."""
        self.messages: list[dict[str, object]] = []
        self.began = time.monotonic()
        self.started_after = 0.0
        self.status_code = 0
        self.headers: dict[str, str] = {}
        self._started = asyncio.Event()
        self._disconnect = asyncio.Event()
        self.task = asyncio.create_task(
            self._run(app, body or speak_body(), scope or speak_scope())
        )

    async def _run(self, app: FastAPI, body: bytes, scope: dict[str, object]) -> None:
        sent = False

        async def receive() -> dict[str, object]:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await self._disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            self.messages.append(message)
            if message["type"] == "http.response.start":
                self.started_after = time.monotonic() - self.began
                self.status_code = int(message["status"])
                self.headers = {
                    key.decode(): value.decode() for key, value in message["headers"]
                }
                self._started.set()

        await app(scope, receive, send)

    async def status(self) -> int:
        """Wait for the status line and return it."""
        await asyncio.wait_for(self._started.wait(), timeout=ASGI_TIMEOUT_SECONDS)
        return self.status_code

    async def finish(self) -> bytes:
        """Wait for the end of the response and return its whole body."""
        await asyncio.wait_for(self.task, timeout=ASGI_TIMEOUT_SECONDS)
        return wav_body(self.messages)


def fake_runtime(
    speaking: SpeakingEngine | None = None, voice: VoiceId = VoiceId.THORSTEN
) -> Runtime:
    """A runtime whose `voice` speaks with `speaking`, a fake by default.

    A start voice holds `speaking` at once; any other voice is offered and loads
    it on its first request, beside a fake start voice.
    """
    engine = speaking or FakeSpeaking()
    if loads_at_start(voice):
        return Runtime({voice: engine}, memory_probe=lambda: 0)
    return Runtime(
        {VoiceId.THORSTEN: FakeSpeaking(), voice: None},
        memory_probe=lambda: 0,
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: engine,
            missing_artifacts=lambda _voice: (),
        ),
    )


def app_over(runtime: Runtime, **settings: float) -> FastAPI:
    """The private app over `runtime` with the wire settings a test sets."""
    return create_control_app(Settings(**settings), runtime, valid_callers())


def synthesis_gate(
    runtime: Runtime, voice: VoiceId = VoiceId.THORSTEN
) -> AdmissionGate:
    """The admission gate of `voice`'s slot."""
    return runtime.slots[voice].gate


def gate_acquisition_began(
    runtime: Runtime, voice: VoiceId = VoiceId.THORSTEN
) -> asyncio.Event:
    """An event set when a request calls the gate's `acquire`.

    `acquire` queues its waiter before its first suspension, so a test that
    resumes on this event finds the waiter queued.
    """
    began = asyncio.Event()
    gate = synthesis_gate(runtime, voice)
    acquire = gate.acquire

    async def noting(deadline: float, max_waiters: int) -> Lease | None:
        began.set()
        return await acquire(deadline, max_waiters)

    gate.acquire = noting
    return began


def gate_is_free(runtime: Runtime, voice: VoiceId = VoiceId.THORSTEN) -> bool:
    """Whether `voice`'s synthesis gate is free: no request left its lease behind."""
    lease = synthesis_gate(runtime, voice).try_acquire()
    if lease is None:
        return False
    lease.close()
    return True


def post_speech(
    client: TestClient, body: bytes, headers: Mapping[str, str] | None = None
) -> ClientResponse:
    """POST `body` to the wire's speech path; `headers` default to a valid pair."""
    sent = valid_headers() if headers is None else headers
    return client.post(
        SPEECH_PATH,
        content=body,
        headers={"content-type": "application/json", **sent},
    )


def assert_refusal(
    response: ClientResponse, status: int, code: RefusalCode, param: str | None
) -> None:
    """The response is OpenAI's error form with exactly the closed keys."""
    assert response.headers["content-type"] == "application/json"
    assert_refusal_body(response.status_code, response.json(), status, code, param)


def assert_call_refusal(
    call: SpeakCall, status: int, code: RefusalCode, param: str | None
) -> None:
    """The finished call was refused in OpenAI's error form."""
    assert call.headers["content-type"] == "application/json"
    assert_refusal_body(
        call.status_code, json.loads(wav_body(call.messages)), status, code, param
    )


def assert_refusal_body(
    actual_status: int,
    body: dict[str, dict[str, object]],
    status: int,
    code: RefusalCode,
    param: str | None,
) -> None:
    """`body` is the closed error form of the refusal `status`, `code`, `param`."""
    assert actual_status == status
    assert set(body) == {"error"}
    assert set(body["error"]) == {"message", "type", "param", "code"}
    assert body["error"]["code"] == code
    assert body["error"]["param"] == param
    assert body["error"]["type"] == (
        "server_error"
        if status >= HTTPStatus.INTERNAL_SERVER_ERROR
        else "invalid_request_error"
    )


def assert_unavailable(response: ClientResponse) -> None:
    """The voice is refused as unavailable, naming the voice."""
    assert_refusal(
        response, HTTPStatus.SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE, "voice"
    )


def status_of(messages: list[dict[str, object]]) -> int:
    """The status of an ASGI response sent as `messages`."""
    return next(
        int(message["status"])
        for message in messages
        if message["type"] == "http.response.start"
    )


class FailingSpeaking(FakeSpeaking):
    """A voice that fails once, after the given number of chunks, then speaks."""

    def __init__(
        self, *, chunks_before_failing: int, chunk_seconds: float | None = None
    ) -> None:
        """Fail the first synthesis after `chunks_before_failing` chunks.

        With `chunk_seconds`, each chunk before the failure is a tone that long.
        """
        super().__init__()
        self._chunks_before_failing = chunks_before_failing
        self._chunk_seconds = chunk_seconds
        self.failed = False

    def pcm_chunks(
        self, text: str, language: str, instructions: str | None = None
    ) -> Iterator[bytes]:
        """Yield the tone, or fail once."""
        if self.failed:
            yield from super().pcm_chunks(text, language, instructions)
            return
        self.failed = True
        if self._chunk_seconds is None:
            yield from itertools.islice(
                super().pcm_chunks(text, language, instructions),
                self._chunks_before_failing,
            )
        else:
            for _ in range(self._chunks_before_failing):
                yield sine_pcm(self._chunk_seconds, rate=self.sample_rate)
        message = "engine failed"
        raise RuntimeError(message)
