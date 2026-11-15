"""A test balancer that routes each connection by `/health`, as readiness does."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http.client
import time
import zlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from speech_client.wire import HEALTH_PATH, Health

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

_LOOPBACK = "127.0.0.1"
_CHUNK_BYTES = 65_536
_HEALTH_TIMEOUT_SECONDS = 5


def health_of(port: int) -> Health | None:
    """The health the server on `port` answers, or `None` when it does not answer."""
    connection = http.client.HTTPConnection(
        _LOOPBACK, port, timeout=_HEALTH_TIMEOUT_SECONDS
    )
    try:
        connection.request("GET", HEALTH_PATH)
        return Health.model_validate_json(connection.getresponse().read())
    except OSError, http.client.HTTPException:
        return None
    finally:
        connection.close()


def is_ready(port: int) -> bool:
    """Whether the server on `port` can both speak and hear."""
    health = health_of(port)
    return health is not None and health.speaking.ready and health.hearing.ready


class _ServerMessages:
    """Reads the text messages of a backend off the bytes it sends to its client.

    The bytes are the upgrade answer, then unmasked frames whose messages are
    deflated with one context for the connection.
    """

    _END_OF_HEAD = b"\r\n\r\n"
    _SYNC_TAIL = b"\x00\x00\xff\xff"
    _TEXT = 0x1
    _COMPRESSED = 0x40
    _LENGTH_16_BITS = 126
    _LENGTH_64_BITS = 127

    def __init__(self, said: Callable[[str], None]) -> None:
        self._said = said
        self._buffer = b""
        self._upgraded = False
        self._inflate = zlib.decompressobj(-zlib.MAX_WBITS)

    def __call__(self, data: bytes) -> None:
        """Say each text message that `data` completes."""
        self._buffer += data
        if not self._upgraded:
            _, found, rest = self._buffer.partition(self._END_OF_HEAD)
            if not found:
                return
            self._upgraded, self._buffer = True, rest
        while (frame := self._take_frame()) is not None:
            first, payload = frame
            if first & 0x0F != self._TEXT:
                continue
            if first & self._COMPRESSED:
                payload = self._inflate.decompress(payload + self._SYNC_TAIL)
            self._said(payload.decode())

    def _take_frame(self) -> tuple[int, bytes] | None:
        buffer = self._buffer
        if len(buffer) < 2:  # noqa: PLR2004 - the two header bytes
            return None
        length, offset = buffer[1] & 0x7F, 2
        if length == self._LENGTH_16_BITS:
            offset = 4
        elif length == self._LENGTH_64_BITS:
            offset = 10
        if len(buffer) < offset:
            return None
        if offset > 2:  # noqa: PLR2004 - an extended length follows
            length = int.from_bytes(buffer[2:offset])
        if len(buffer) < offset + length:
            return None
        self._buffer = buffer[offset + length :]
        return buffer[0], buffer[offset : offset + length]


async def _pipe(
    source: asyncio.StreamReader,
    target: asyncio.StreamWriter,
    tap: Callable[[bytes], None] | None = None,
) -> None:
    while data := await source.read(_CHUNK_BYTES):
        if tap is not None:
            tap(data)
        target.write(data)
        await target.drain()


async def _relay(
    client: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    backend: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    backend_tap: Callable[[bytes], None] | None,
) -> None:
    """Copy both directions until either side ends, then close both."""
    pipes = [
        asyncio.create_task(_pipe(client[0], backend[1])),
        asyncio.create_task(_pipe(backend[0], client[1], backend_tap)),
    ]
    try:
        await asyncio.wait(pipes, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for pipe in pipes:
            pipe.cancel()
        await asyncio.gather(*pipes, return_exceptions=True)
        for writer in (client[1], backend[1]):
            writer.close()


@asynccontextmanager
async def balancing(
    backend_ports: Sequence[int],
    backend_ended: dict[int, float] | None = None,
    backend_said: Callable[[int, str], None] | None = None,
) -> AsyncIterator[int]:
    """Yield the port of a balancer sending each connection to the first ready backend.

    A connection finds no ready backend closed at once. When a connection ends,
    `backend_ended` gets the monotonic time of its end under its backend's port.
    `backend_said` gets each text message a backend sends, with the backend's port.
    """

    async def route(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        for port in backend_ports:
            if await asyncio.to_thread(is_ready, port):
                with contextlib.suppress(ConnectionError):
                    backend = await asyncio.open_connection(_LOOPBACK, port)
                    await _relay(
                        (reader, writer),
                        backend,
                        None
                        if backend_said is None
                        else _ServerMessages(functools.partial(backend_said, port)),
                    )
                if backend_ended is not None:
                    backend_ended[port] = time.monotonic()
                return
        writer.close()

    server = await asyncio.start_server(route, _LOOPBACK, 0)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        server.close()
        await server.wait_closed()
