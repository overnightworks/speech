"""The client's speaking call against the real service and against a raw listener."""

from __future__ import annotations

import asyncio
import math
import socket
import struct
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING

import httpx2
import pytest

from speech_runtime_client.speak import (
    Busy,
    Cut,
    Malformed,
    OverCap,
    PastDeadline,
    Refused,
    Spoken,
    Unreachable,
    speak,
)
from speech_runtime_client.wire import (
    RETRY_AFTER_HEADER,
    SPEECH_PATH,
    RefusalCode,
    refusal,
    speech_request_body,
    speech_request_headers,
)
from speech_runtime_service.config import Settings
from speech_runtime_service.pcm import WAV_HEADER_BYTES
from speech_runtime_service.service import Runtime
from speech_runtime_service.voices import VoiceId
from tests.support.fakes_support import (
    GATE_TIMEOUT_SECONDS,
    FakeSpeaking,
    GatedSpeaking,
)
from tests.support.identity_support import valid_headers, valid_pair
from tests.support.speak_support import FailingSpeaking
from tests.support.tcp_server_support import (
    TcpServer,
    free_loopback_port,
    serving_over_tcp,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    from proof_identity import TokenPair
    from speech_runtime_client.speak import SpeakOutcome

PIPER = VoiceId.PIPER.value
DEADLINE_SECONDS = 0.5
MARGIN_SECONDS = 3.0
CAP_BYTES = 1024 * 1024
DRIP_SECONDS = 0.02


def _speak(
    base_url: str,
    *,
    voice: str = PIPER,
    deadline_seconds: float = GATE_TIMEOUT_SECONDS,
    cap_bytes: int = CAP_BYTES,
    pair: TokenPair | None = None,
) -> SpeakOutcome:
    pair = pair or valid_pair()
    return speak(
        base_url,
        "Hallo",
        voice,
        workload_token=pair.workload,
        person_token=pair.person,
        deadline_seconds=deadline_seconds,
        cap_bytes=cap_bytes,
    )


def _runtime(voice: FakeSpeaking | GatedSpeaking) -> Runtime:
    return Runtime(voice, memory_probe=lambda: 0)


@asynccontextmanager
async def _serving(
    voice: FakeSpeaking | GatedSpeaking,
) -> AsyncIterator[TcpServer]:
    """Serve `voice` over TCP; open every gate on exit so the server can stop."""
    async with serving_over_tcp(_runtime(voice)) as server:
        try:
            yield server
        finally:
            if isinstance(voice, GatedSpeaking):
                voice.release_all()


def _over_the_service(
    voice: FakeSpeaking | GatedSpeaking, **options: float
) -> SpeakOutcome:
    async def scenario() -> SpeakOutcome:
        async with _serving(voice) as server:
            return await asyncio.to_thread(_speak, server.base_url, **options)

    return asyncio.run(scenario())


def test_the_call_returns_the_wav_the_service_sent() -> None:
    """WIRE-88, WIRE-90: the whole chunked body comes back as the WAV, byte for byte."""

    async def scenario() -> tuple[SpeakOutcome, httpx2.Response]:
        async with _serving(FakeSpeaking()) as server:
            outcome = await asyncio.to_thread(_speak, server.base_url)
            async with httpx2.AsyncClient(
                base_url=server.base_url, headers=valid_headers()
            ) as client:
                return outcome, await client.post(
                    SPEECH_PATH, content=speech_request_body("Hallo", PIPER)
                )

    outcome, sent = asyncio.run(scenario())

    assert outcome == Spoken(sent.content)


def test_a_wav_of_exactly_the_cap_is_spoken() -> None:
    whole = _over_the_service(FakeSpeaking())
    assert isinstance(whole, Spoken)

    outcome = _over_the_service(FakeSpeaking(), cap_bytes=len(whole.wav))

    assert outcome == whole


def test_a_cap_below_the_wav_is_over_the_cap() -> None:
    outcome = _over_the_service(FakeSpeaking(), cap_bytes=WAV_HEADER_BYTES)

    assert outcome == OverCap()


def test_an_unknown_voice_is_refused_with_its_status_code_and_param() -> None:
    async def scenario() -> SpeakOutcome:
        async with _serving(FakeSpeaking()) as server:
            return await asyncio.to_thread(_speak, server.base_url, voice="alloy")

    outcome = asyncio.run(scenario())

    assert isinstance(outcome, Refused)
    assert outcome.refusal == refusal(RefusalCode.INVALID_REQUEST, "voice")
    assert outcome.status == 422  # noqa: PLR2004 - the wire's status for invalid_request


@pytest.mark.parametrize("chunks_before_failing", [0, 1])
def test_a_body_the_service_aborted_is_cut(chunks_before_failing: int) -> None:
    """WIRE-90: a body without its end is never a WAV."""
    voice = FailingSpeaking(chunks_before_failing=chunks_before_failing)

    assert _over_the_service(voice) == Cut()


def test_a_voice_that_stalls_leaves_the_call_past_the_deadline() -> None:
    voice = GatedSpeaking()

    began = time.monotonic()
    outcome = _over_the_service(voice, deadline_seconds=DEADLINE_SECONDS)

    assert outcome == PastDeadline()
    assert time.monotonic() - began < DEADLINE_SECONDS + MARGIN_SECONDS


def test_a_request_beyond_the_waiting_places_is_told_busy_with_the_wait() -> None:
    """WIRE-23: one held call, the default waiting places full, then one more."""
    places = Settings().max_waiting_requests

    async def scenario() -> tuple[SpeakOutcome, list[SpeakOutcome]]:
        voice = GatedSpeaking()
        async with _serving(voice) as server:
            held = asyncio.create_task(asyncio.to_thread(_speak, server.base_url))
            assert await asyncio.to_thread(
                voice.entered_first_step.wait, GATE_TIMEOUT_SECONDS
            )
            more = [
                asyncio.create_task(asyncio.to_thread(_speak, server.base_url))
                for _ in range(places + 1)
            ]
            done, waiting = await asyncio.wait(
                more, timeout=GATE_TIMEOUT_SECONDS, return_when=asyncio.FIRST_COMPLETED
            )
            voice.release_all()
            return done.pop().result(), await asyncio.gather(held, *waiting)

    first, later = asyncio.run(scenario())

    assert isinstance(first, Busy)
    assert first.wait.retry_after_seconds == math.ceil(Settings().busy_wait_seconds)
    assert all(isinstance(outcome, Spoken) for outcome in later)


def test_a_port_nothing_listens_on_is_unreachable() -> None:
    assert _speak(f"http://127.0.0.1:{free_loopback_port()}") == Unreachable()


@pytest.mark.parametrize(
    "options",
    [
        {"deadline_seconds": 0},
        {"deadline_seconds": -1},
        {"cap_bytes": 0},
        {"cap_bytes": -1},
        {"voice": ""},
    ],
)
def test_a_defect_of_the_caller_raises(options: dict[str, float | str]) -> None:
    with pytest.raises(ValueError, match=r"."):
        _speak("http://127.0.0.1:1", **options)


@pytest.mark.parametrize(
    "base_url",
    ["https://127.0.0.1:1", "http://127.0.0.1:1/v1", "127.0.0.1:1", "http://:1"],
)
def test_a_base_url_that_is_not_plain_http_without_a_path_raises(
    base_url: str,
) -> None:
    with pytest.raises(ValueError, match=r"base_url"):
        _speak(base_url)


# A raw listener answers what the real service never sends.

_CRLF = b"\r\n"


def _chunk(data: bytes) -> bytes:
    return b"%x\r\n%b\r\n" % (len(data), data)


def _head(status: str, *headers: str) -> bytes:
    return "\r\n".join([f"HTTP/1.1 {status}", *headers, "", ""]).encode()


_WAV_HEAD = _head("200 OK", "Content-Type: audio/wav", "Transfer-Encoding: chunked")


def _refusal_response(
    status: str, code: RefusalCode, *headers: str, padding: int = 0
) -> bytes:
    body = refusal(code).model_dump_json().encode() + b" " * padding
    return _head(status, f"Content-Length: {len(body)}", *headers) + body


def _read_request(connection: socket.socket) -> bytes:
    received = b""
    while b"\r\n\r\n" not in received:
        received += connection.recv(4096)
    head, _, body = received.partition(b"\r\n\r\n")
    length = next(
        int(line.split(b":")[1])
        for line in head.split(_CRLF)
        if line.lower().startswith(b"content-length")
    )
    while len(body) < length:
        body += connection.recv(4096)
    return head + b"\r\n\r\n" + body


@contextmanager
def _raw_listener(
    answer: Callable[[socket.socket, threading.Event], None],
) -> Iterator[tuple[str, list[bytes]]]:
    """Answer one connection with `answer`; yield the base URL and the requests."""
    requests: list[bytes] = []
    stopped = threading.Event()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            with connection:
                requests.append(_read_request(connection))
                answer(connection, stopped)
        except OSError:
            return

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}", requests
    finally:
        stopped.set()
        listener.shutdown(socket.SHUT_RDWR)
        listener.close()
        thread.join()


def _sends(raw: bytes) -> Callable[[socket.socket, threading.Event], None]:
    def answer(connection: socket.socket, stopped: threading.Event) -> None:
        del stopped
        connection.sendall(raw)

    return answer


def _sends_then_drips(
    raw: bytes, drip: bytes
) -> Callable[[socket.socket, threading.Event], None]:
    def answer(connection: socket.socket, stopped: threading.Event) -> None:
        connection.sendall(raw)
        while not stopped.wait(DRIP_SECONDS):
            connection.sendall(drip)

    return answer


def _resets(connection: socket.socket, stopped: threading.Event) -> None:
    del stopped
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))


def _closes(connection: socket.socket, stopped: threading.Event) -> None:
    del connection, stopped


@pytest.mark.parametrize(
    ("raw", "drip"),
    [
        pytest.param(b"HTTP/1.1 200 OK\r\n", b"X", id="inside-a-header-line"),
        pytest.param(_WAV_HEAD, b"1", id="inside-a-chunk-size-line"),
        pytest.param(_WAV_HEAD + b"ffff\r\n", b"\0", id="inside-a-chunk"),
    ],
)
def test_a_server_that_drips_cannot_hold_the_call_past_the_deadline(
    raw: bytes, drip: bytes
) -> None:
    """The deadline bounds the whole call, not each read."""
    with _raw_listener(_sends_then_drips(raw, drip)) as (base_url, _):
        began = time.monotonic()
        outcome = _speak(base_url, deadline_seconds=DEADLINE_SECONDS)

    assert outcome == PastDeadline()
    assert time.monotonic() - began < DEADLINE_SECONDS + MARGIN_SECONDS


def test_a_deadline_that_has_passed_before_the_connect_is_past_the_deadline() -> None:
    with _raw_listener(_closes) as (base_url, _):
        outcome = _speak(base_url, deadline_seconds=1e-9)

    assert outcome == PastDeadline()


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(_resets, id="reset-before-the-status-line"),
        pytest.param(_closes, id="close-before-the-status-line"),
        pytest.param(_sends(_WAV_HEAD + _chunk(b"RIFF")), id="close-without-the-end"),
    ],
)
def test_an_answer_that_ends_before_it_is_whole_is_cut(
    answer: Callable[[socket.socket, threading.Event], None],
) -> None:
    """WIRE-90: a missing end is a failed request."""
    with _raw_listener(answer) as (base_url, _):
        outcome = _speak(base_url)

    assert outcome == Cut()


_BUSY_WITHOUT_WAIT = _refusal_response("429 Too Many Requests", RefusalCode.BUSY)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"not http at all\r\n\r\n", id="bad-status-line"),
        pytest.param(b"HTTP/1.1 200 " + b"a" * 70_000 + b"\r\n", id="line-too-long"),
        pytest.param(
            _head("200 OK", *("X-Header: 1",) * 101), id="more-than-100-headers"
        ),
        pytest.param(
            _head(
                "200 OK", "Content-Type: application/json", "Transfer-Encoding: chunked"
            )
            + _chunk(b"{}")
            + _chunk(b""),
            id="200-that-is-not-audio-wav",
        ),
        pytest.param(
            _head("200 OK", "Content-Type: audio/wav", "Content-Length: 4") + b"RIFF",
            id="200-that-is-not-chunked",
        ),
        pytest.param(
            _head("422 Unprocessable Entity", "Content-Length: 15")
            + b'{"detail":"no"}',
            id="refusal-outside-the-error-form",
        ),
        pytest.param(
            _refusal_response("401 Unauthorized", RefusalCode.NOT_FOUND),
            id="refusal-whose-code-has-another-status",
        ),
        pytest.param(_BUSY_WITHOUT_WAIT, id="429-without-retry-after"),
        pytest.param(
            _refusal_response(
                "429 Too Many Requests", RefusalCode.BUSY, f"{RETRY_AFTER_HEADER}: soon"
            ),
            id="429-with-a-retry-after-that-is-no-number",
        ),
        pytest.param(
            _refusal_response("500 Internal Server Error", RefusalCode.UNAVAILABLE),
            id="a-status-the-wire-does-not-name",
        ),
        pytest.param(
            _refusal_response(
                "422 Unprocessable Entity",
                RefusalCode.INVALID_REQUEST,
                padding=64 * 1024,
            ),
            id="refusal-body-over-64-kib",
        ),
    ],
)
def test_an_answer_that_breaks_the_wire_is_malformed(raw: bytes) -> None:
    """WIRE-88: only a chunked audio/wav 200 and the wire's refusals are answers."""
    with _raw_listener(_sends(raw)) as (base_url, _):
        outcome = _speak(base_url)

    assert outcome == Malformed()


def test_the_request_is_the_wires_body_and_headers() -> None:
    pair = valid_pair()

    with _raw_listener(_sends(_WAV_HEAD + _chunk(b"RIFF") + _chunk(b""))) as (
        base_url,
        requests,
    ):
        outcome = _speak(base_url, pair=pair)

    assert outcome == Spoken(b"RIFF")
    (request,) = requests
    head, _, body = request.partition(b"\r\n\r\n")
    lines = head.decode().split("\r\n")
    assert lines[0] == f"POST {SPEECH_PATH} HTTP/1.1"
    assert body == speech_request_body("Hallo", PIPER)
    sent = {line.partition(": ")[0]: line.partition(": ")[2] for line in lines[1:]}
    expected = speech_request_headers(
        workload_token=pair.workload, person_token=pair.person
    )
    assert all(sent[name] == value for name, value in expected.items())


def test_a_429_with_a_whole_second_wait_is_busy() -> None:
    raw = _refusal_response(
        "429 Too Many Requests", RefusalCode.BUSY, f"{RETRY_AFTER_HEADER}: 7"
    )

    with _raw_listener(_sends(raw)) as (base_url, _):
        outcome = _speak(base_url)

    assert isinstance(outcome, Busy)
    assert outcome.wait.retry_after_seconds == 7  # noqa: PLR2004 - the header's value
