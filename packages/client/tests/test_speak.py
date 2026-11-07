"""The speaking call against a raw listener: what the real service never sends."""

from __future__ import annotations

import socket
import struct
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from speech_runtime_client.speak import (
    Busy,
    Cut,
    Malformed,
    PastDeadline,
    Spoken,
    Unreachable,
    speak,
)
from speech_runtime_client.wire import (
    RETRY_AFTER_HEADER,
    SPEECH_PATH,
    RefusalCode,
    ResponseFormat,
    refusal,
    speech_request_body,
    speech_request_headers,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from speech_runtime_client.speak import SpeakOutcome

VOICE = "piper"
WORKLOAD_TOKEN = "workload-token"  # noqa: S105 - a stand-in the raw listener never checks
PERSON_TOKEN = "person-token"  # noqa: S105 - a stand-in the raw listener never checks
DEADLINE_SECONDS = 0.5
MARGIN_SECONDS = 3.0
GENEROUS_DEADLINE_SECONDS = 30.0
CAP_BYTES = 1024 * 1024
DRIP_SECONDS = 0.02


def _speak(
    base_url: str,
    *,
    voice: str = VOICE,
    response_format: ResponseFormat = ResponseFormat.WAV,
    deadline_seconds: float = GENEROUS_DEADLINE_SECONDS,
    cap_bytes: int = CAP_BYTES,
) -> SpeakOutcome:
    return speak(
        base_url,
        "Hallo",
        voice,
        response_format=response_format,
        workload_token=WORKLOAD_TOKEN,
        person_token=PERSON_TOKEN,
        deadline_seconds=deadline_seconds,
        cap_bytes=cap_bytes,
    )


def test_a_port_nothing_listens_on_is_unreachable() -> None:
    # Bound and not listening: the connect is refused, no other test can take the port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as bound:
        bound.bind(("127.0.0.1", 0))
        outcome = _speak(f"http://127.0.0.1:{bound.getsockname()[1]}")

    assert outcome == Unreachable()


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
_OGG_HEAD = _head("200 OK", "Content-Type: audio/ogg", "Transfer-Encoding: chunked")
_WAV_BODY = b"RIFF\0\0\0\0WAVE"
_OGG_BODY = b"OggS\0\0\0\0"


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
        pytest.param(_sends(_WAV_HEAD + _chunk(_WAV_BODY)), id="close-without-the-end"),
        pytest.param(
            _sends(_WAV_HEAD + _chunk(_WAV_BODY) + b"0"),
            id="close-inside-the-last-chunk-line",
        ),
        pytest.param(
            _sends(_WAV_HEAD + _chunk(_WAV_BODY) + b"0\r\n"),
            id="close-before-the-final-blank-line",
        ),
        pytest.param(
            _sends(_head("422 Unprocessable Entity", "Content-Length: 500") + b"{"),
            id="refusal-shorter-than-its-length",
        ),
    ],
)
def test_an_answer_that_ends_before_it_is_whole_is_cut(
    answer: Callable[[socket.socket, threading.Event], None],
) -> None:
    """WIRE-90: a missing end is a failed request."""
    with _raw_listener(answer) as (base_url, _):
        outcome = _speak(base_url)

    assert outcome == Cut()


_CANONICAL = refusal(RefusalCode.INVALID_REQUEST)
_ECHOED = (
    _CANONICAL.model_copy(
        update={
            "error": _CANONICAL.error.model_copy(update={"message": PERSON_TOKEN}),
        },
    )
    .model_dump_json()
    .encode()
)
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
        pytest.param(
            _head("422 Unprocessable Entity", f"Content-Length: {len(_ECHOED)}")
            + _ECHOED,
            id="refusal-with-a-message-that-is-not-the-canonical-one",
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
        pytest.param(
            _WAV_HEAD
            + _chunk(_WAV_BODY)
            + b"0\r\n"
            + b"X-Trailer: "
            + b"a" * 70_000
            + b"\r\n\r\n",
            id="trailer-line-too-long",
        ),
    ],
)
def test_an_answer_that_breaks_the_wire_is_malformed(raw: bytes) -> None:
    """WIRE-88: only a chunked audio/wav 200 and the wire's refusals are answers."""
    with _raw_listener(_sends(raw)) as (base_url, _):
        outcome = _speak(base_url)

    assert outcome == Malformed()


def test_a_chunked_wav_with_a_trailer_is_spoken() -> None:
    raw = _WAV_HEAD + _chunk(_WAV_BODY) + b"0\r\nX-Trailer: 1\r\n\r\n"
    with _raw_listener(_sends(raw)) as (base_url, _):
        outcome = _speak(base_url)

    assert outcome == Spoken(_WAV_BODY)


@pytest.mark.parametrize(
    ("ordered", "raw"),
    [
        pytest.param(
            ResponseFormat.OPUS,
            _WAV_HEAD + _chunk(_WAV_BODY) + _chunk(b""),
            id="wav-media-type-for-an-opus-order",
        ),
        pytest.param(
            ResponseFormat.WAV,
            _OGG_HEAD + _chunk(_OGG_BODY) + _chunk(b""),
            id="ogg-media-type-for-a-wav-order",
        ),
        pytest.param(
            ResponseFormat.WAV,
            _WAV_HEAD + _chunk(b"not audio at all") + _chunk(b""),
            id="wav-media-type-over-a-body-of-text",
        ),
        pytest.param(
            ResponseFormat.WAV,
            _WAV_HEAD + _chunk(b"RIFF\0\0\0\0AVI ") + _chunk(b""),
            id="riff-that-is-not-wave",
        ),
        pytest.param(
            ResponseFormat.OPUS,
            _OGG_HEAD + _chunk(_WAV_BODY) + _chunk(b""),
            id="ogg-media-type-over-a-wav-body",
        ),
        pytest.param(
            ResponseFormat.OPUS,
            _OGG_HEAD + _chunk(b""),
            id="ogg-media-type-over-an-empty-body",
        ),
    ],
)
def test_a_200_that_is_not_the_ordered_format_is_malformed(
    ordered: ResponseFormat, raw: bytes
) -> None:
    """WIRE-88, WIRE-133: the media type and the container signature must match."""
    with _raw_listener(_sends(raw)) as (base_url, _):
        outcome = _speak(base_url, response_format=ordered)

    assert outcome == Malformed()


def test_a_chunked_ogg_with_the_ogg_signature_is_spoken_when_opus_is_ordered() -> None:
    """WIRE-133: an audio/ogg body that begins with OggS is the answer."""
    with _raw_listener(_sends(_OGG_HEAD + _chunk(_OGG_BODY) + _chunk(b""))) as (
        base_url,
        _,
    ):
        outcome = _speak(base_url, response_format=ResponseFormat.OPUS)

    assert outcome == Spoken(_OGG_BODY)


@pytest.mark.parametrize("ordered", list(ResponseFormat))
def test_the_request_orders_the_format_it_was_given(ordered: ResponseFormat) -> None:
    with _raw_listener(_sends(_WAV_HEAD + _chunk(_WAV_BODY) + _chunk(b""))) as (
        base_url,
        requests,
    ):
        _speak(base_url, response_format=ordered)

    (request,) = requests
    assert request.partition(b"\r\n\r\n")[2] == speech_request_body(
        "Hallo", VOICE, ordered
    )


def test_the_request_is_the_wires_body_and_headers() -> None:
    with _raw_listener(_sends(_WAV_HEAD + _chunk(_WAV_BODY) + _chunk(b""))) as (
        base_url,
        requests,
    ):
        outcome = _speak(base_url)

    assert outcome == Spoken(_WAV_BODY)
    (request,) = requests
    head, _, body = request.partition(b"\r\n\r\n")
    lines = head.decode().split("\r\n")
    assert lines[0] == f"POST {SPEECH_PATH} HTTP/1.1"
    assert body == speech_request_body("Hallo", VOICE, ResponseFormat.WAV)
    sent = {line.partition(": ")[0]: line.partition(": ")[2] for line in lines[1:]}
    expected = speech_request_headers(
        workload_token=WORKLOAD_TOKEN, person_token=PERSON_TOKEN
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
