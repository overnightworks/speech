"""The speaking call: one blocking `POST /v1/audio/speech`, bounded in time and size.

`speak` takes every input from its call site and returns one `SpeakOutcome`,
a closed union a caller matches with `assert_never`:

- `Spoken`: the WAV bytes exactly as received, read to the end of the chunked
  body (WIRE-90).
- `Refused`: a refusal of the wire, with its status, code and param.
- `Busy`: the 429 and the wait the service asks for; the app decides to retry.
- `Cut`: the connection was up and the answer ended, closed or was reset
  before it was whole.
- `PastDeadline`: the total deadline passed, at connect, send or any read.
- `OverCap`: the WAV would pass the byte cap; reading stops there.
- `Malformed`: the answer breaks the wire, for example a 200 that is not a
  chunked `audio/wav` (WIRE-88), a refusal outside the error form, a 429
  without a whole-second `Retry-After`, or a refusal body over 64 KiB.
- `Unreachable`: the name did not resolve or no address accepted the connection.

A timeout is `PastDeadline` before anything else; then a closed or reset
connection is `Cut`; only then a parse failure is `Malformed`.

A defect of the caller raises and is no outcome: an input the request refuses
(`pydantic.ValidationError`), a deadline or cap of 0 or below, and a base URL
that is not `http`, has no host or carries a path (`ValueError`). The service
is plain HTTP on internal networks.

The call blocks and holds no process-global state, so it runs in a worker
thread (`asyncio.to_thread`); the thread ends by the deadline. It logs nothing,
and no outcome carries the text or a token. The name lookup is bounded by the
system resolver, not by the deadline.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.client import (
    HTTPConnection,
    HTTPException,
    HTTPResponse,
    IncompleteRead,
    LineTooLong,
)
from typing import Final
from urllib.parse import urlsplit

from pydantic import ValidationError

from speech_runtime_client.wire import (
    SPEECH_PATH,
    BusyWait,
    Refusal,
    RefusalCode,
    RequestField,
    RetryAfterError,
    busy_wait,
    refusal,
    speech_request_body,
    speech_request_headers,
)

_READ_BYTES: Final = 64 * 1024
_TRAILER_LINE_BYTES: Final = 64 * 1024
_REFUSAL_BODY_LIMIT: Final = 64 * 1024
_WAV_MEDIA_TYPE: Final = "audio/wav"


@dataclass(frozen=True, slots=True)
class Spoken:
    """The WAV, exactly as the service sent it."""

    wav: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class Refused:
    """A refusal of the wire."""

    refusal: Refusal

    @property
    def status(self) -> HTTPStatus:
        """The HTTP status the service answered."""
        return self.refusal.status

    @property
    def code(self) -> RefusalCode:
        """Why the service refused."""
        return self.refusal.error.code

    @property
    def param(self) -> RequestField | None:
        """The request field at fault, if any."""
        return self.refusal.error.param


@dataclass(frozen=True, slots=True)
class Busy:
    """The service is busy and asks for a wait."""

    wait: BusyWait


@dataclass(frozen=True, slots=True)
class Cut:
    """The answer ended, closed or was reset before it was whole."""


@dataclass(frozen=True, slots=True)
class PastDeadline:
    """The total deadline passed."""


@dataclass(frozen=True, slots=True)
class OverCap:
    """The WAV would pass the byte cap."""


@dataclass(frozen=True, slots=True)
class Malformed:
    """The answer breaks the wire."""


@dataclass(frozen=True, slots=True)
class Unreachable:
    """The connection never came up."""


type SpeakOutcome = (
    Spoken | Refused | Busy | Cut | PastDeadline | OverCap | Malformed | Unreachable
)


class _DeadlineSocket(socket.socket):
    """A socket whose connect, sends and reads are each bounded by one deadline."""

    def __init__(self, family: int, kind: int, protocol: int, deadline: float) -> None:
        super().__init__(family, kind, protocol)
        self._deadline = deadline

    def arm(self) -> None:
        left = self._deadline - time.monotonic()
        if left <= 0:
            # settimeout(0) would make the socket non-blocking, a negative value raises.
            raise TimeoutError
        self.settimeout(left)

    def sendall(self, data: bytes, flags: int = 0) -> None:
        self.arm()
        super().sendall(data, flags)

    def recv_into(self, buffer: bytearray, nbytes: int = 0, flags: int = 0) -> int:
        self.arm()
        return super().recv_into(buffer, nbytes, flags)


def _connect(host: str, port: int, deadline: float) -> socket.socket:
    failure = OSError("the name resolved to no address")
    for family, kind, protocol, _, address in socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    ):
        candidate = _DeadlineSocket(family, kind, protocol, deadline)
        try:
            candidate.arm()
            candidate.connect(address)
        except TimeoutError:
            candidate.close()
            raise
        except OSError as error:
            candidate.close()
            failure = error
        else:
            return candidate
    raise failure


class _StrictResponse(HTTPResponse):
    """A response that treats a chunked body without its terminator as cut."""

    def _read_and_discard_trailer(self) -> None:
        # The base class accepts an end of stream where the final CRLF belongs.
        while True:
            line = self.fp.readline(_TRAILER_LINE_BYTES + 1)
            if len(line) > _TRAILER_LINE_BYTES:
                message = "trailer line"
                raise LineTooLong(message)
            if not line:
                # The argument is the bytes read, not a message.
                raise IncompleteRead(b"")  # noqa: EM101
            if line in {b"\r\n", b"\n"}:
                return


def _read_within(response: HTTPResponse, limit: int) -> bytes | None:
    """The whole body, or `None` once it would pass `limit` bytes."""
    body = bytearray()
    while piece := response.read1(_READ_BYTES):
        body += piece
        if len(body) > limit:
            return None
    if response.length:
        raise IncompleteRead(bytes(body), response.length)
    return bytes(body)


def _spoken(response: HTTPResponse, cap_bytes: int) -> SpeakOutcome:
    if response.headers.get_content_type() != _WAV_MEDIA_TYPE or not response.chunked:
        return Malformed()
    wav = _read_within(response, cap_bytes)
    return OverCap() if wav is None else Spoken(wav)


def _refusal(response: HTTPResponse) -> SpeakOutcome:
    body = _read_within(response, _REFUSAL_BODY_LIMIT)
    if body is None:
        return Malformed()
    try:
        parsed = Refusal.model_validate_json(body)
        wait = busy_wait(response.status, dict(response.getheaders()))
    except ValidationError, RetryAfterError:
        return Malformed()
    if parsed.status != response.status:
        return Malformed()
    if parsed != refusal(parsed.error.code, parsed.error.param):
        return Malformed()
    return Refused(parsed) if wait is None else Busy(wait)


def _answer(
    connection: HTTPConnection, body: bytes, headers: dict[str, str], cap_bytes: int
) -> SpeakOutcome:
    connection.request("POST", SPEECH_PATH, body, headers)
    response = connection.getresponse()
    if response.status == HTTPStatus.OK:
        return _spoken(response, cap_bytes)
    return _refusal(response)


# Every input comes from the call site, none has a default.
def speak(  # noqa: PLR0913
    base_url: str,
    text: str,
    voice: str,
    *,
    workload_token: str,
    person_token: str,
    deadline_seconds: float,
    cap_bytes: int,
) -> SpeakOutcome:
    """Speak `text` with `voice` at the service on `base_url`, within the bounds.

    `deadline_seconds` bounds the whole call, `cap_bytes` the WAV. See the module
    for the outcomes and for what raises.
    """
    address = urlsplit(base_url)
    if address.scheme != "http" or not address.hostname or address.path:
        message = "base_url must be http://host[:port] with no path"
        raise ValueError(message)
    if deadline_seconds <= 0 or cap_bytes <= 0:
        message = "deadline_seconds and cap_bytes must be above 0"
        raise ValueError(message)
    body = speech_request_body(text, voice)
    headers = speech_request_headers(
        workload_token=workload_token, person_token=person_token
    )
    port = address.port or 80
    deadline = time.monotonic() + deadline_seconds
    try:
        connected = _connect(address.hostname, port, deadline)
    except TimeoutError:
        return PastDeadline()
    except OSError:
        return Unreachable()
    connection = HTTPConnection(address.hostname, port)
    connection.response_class = _StrictResponse
    connection.sock = connected
    try:
        return _answer(connection, body, headers, cap_bytes)
    except TimeoutError:
        return PastDeadline()
    except OSError, IncompleteRead:
        return Cut()
    except HTTPException:
        return Malformed()
    finally:
        connection.close()
