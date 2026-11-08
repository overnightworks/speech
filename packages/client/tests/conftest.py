"""The raw listener the client tests share: it answers what the service never sends."""

from __future__ import annotations

import socket
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextlib import AbstractContextManager

_CRLF = b"\r\n"


def _read_request(connection: socket.socket) -> bytes:
    received = b""
    while b"\r\n\r\n" not in received:
        received += connection.recv(4096)
    head, _, body = received.partition(b"\r\n\r\n")
    length = next(
        (
            int(line.split(b":")[1])
            for line in head.split(_CRLF)
            if line.lower().startswith(b"content-length")
        ),
        0,
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


@pytest.fixture
def raw_listener() -> Callable[
    [Callable[[socket.socket, threading.Event], None]],
    AbstractContextManager[tuple[str, list[bytes]]],
]:
    return _raw_listener
