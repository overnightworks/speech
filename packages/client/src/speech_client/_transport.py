"""The bounded transport of the client's calls: one origin, one deadline, one mapping.

`speak` and `fetch_health` open one connection through `exchange`, which owns
the mapping from transport exceptions to the shared outcomes: at connect a
timeout is `PastDeadline` and any other `OSError` is `Unreachable`; once
connected a timeout is `PastDeadline`, then `OSError` and `IncompleteRead` are
`Cut`, then `HTTPException` is `Malformed`.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from http.client import (
    HTTPConnection,
    HTTPException,
    HTTPResponse,
    IncompleteRead,
    LineTooLong,
)
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from speech_client.outcomes import Cut, Malformed, PastDeadline, Unreachable

if TYPE_CHECKING:
    from collections.abc import Callable

_READ_BYTES: Final = 64 * 1024
_TRAILER_LINE_BYTES: Final = 64 * 1024


@dataclass(frozen=True, slots=True)
class Origin:
    """Where the service listens."""

    host: str
    port: int


def checked_origin(base_url: str, deadline_seconds: float, cap_bytes: int) -> Origin:
    """The origin of `base_url`; a defect of the caller raises `ValueError`."""
    address = urlsplit(base_url)
    if address.scheme != "http" or not address.hostname or address.path:
        message = "base_url must be http://host[:port] with no path"
        raise ValueError(message)
    if deadline_seconds <= 0 or cap_bytes <= 0:
        message = "deadline_seconds and cap_bytes must be above 0"
        raise ValueError(message)
    return Origin(address.hostname, address.port or 80)


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


def read_within(response: HTTPResponse, limit: int) -> bytes | None:
    """The whole body, or `None` once it would pass `limit` bytes."""
    body = bytearray()
    while piece := response.read1(_READ_BYTES):
        body += piece
        if len(body) > limit:
            return None
    if response.length:
        raise IncompleteRead(bytes(body), response.length)
    return bytes(body)


def exchange[Answer](
    origin: Origin,
    deadline_seconds: float,
    ask: Callable[[HTTPConnection], Answer],
) -> Answer | Cut | PastDeadline | Malformed | Unreachable:
    """`ask` on one connection to `origin`, or the outcome that stopped it."""
    deadline = time.monotonic() + deadline_seconds
    try:
        connected = _connect(origin.host, origin.port, deadline)
    except TimeoutError:
        return PastDeadline()
    except OSError:
        return Unreachable()
    connection = HTTPConnection(origin.host, origin.port)
    connection.response_class = _StrictResponse
    connection.sock = connected
    try:
        return ask(connection)
    except TimeoutError:
        return PastDeadline()
    except OSError, IncompleteRead:
        return Cut()
    except HTTPException:
        return Malformed()
    finally:
        connection.close()
