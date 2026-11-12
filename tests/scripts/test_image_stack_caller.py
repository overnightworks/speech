"""The image proof's caller: it reads an answer that arrives before its upload ends."""

from __future__ import annotations

import base64
import importlib.util
import socket
import threading
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest

from image_stack import FIXTURES
from prove_image import REFUSED_CONNECTION
from tests.support.tcp_server_support import free_loopback_port

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

_LOOPBACK = "127.0.0.1"
_UPLOAD_BYTES = 8 * 1024 * 1024
_REFUSAL = b"refused"
_ANSWER_TIMEOUT_SECONDS = 20


def load_caller() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "caller", FIXTURES / "caller.py"
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def refusing_port() -> Iterator[int]:
    """One connection: read the headers, answer, close with the upload unread."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind((_LOOPBACK, 0))
    listener.listen()

    def serve() -> None:
        connection, _ = listener.accept()
        with connection:
            received = b""
            while b"\r\n\r\n" not in received:
                chunk = connection.recv(4096)
                if not chunk:
                    return
                received += chunk
            connection.sendall(
                b"HTTP/1.1 401 Unauthorized\r\n"
                b"Content-Length: "
                + str(len(_REFUSAL)).encode()
                + b"\r\n\r\n"
                + _REFUSAL
            )

    server = threading.Thread(target=serve)
    server.start()
    try:
        yield listener.getsockname()[1]
    finally:
        server.join(timeout=_ANSWER_TIMEOUT_SECONDS)
        listener.close()


def specification(port: int, body: bytes) -> dict[str, str]:
    return {
        "url": f"http://{_LOOPBACK}:{port}/v1/audio/transcriptions",
        "method": "POST",
        "headers": {"Content-Type": "application/octet-stream"},
        "body": base64.b64encode(body).decode("ascii"),
    }


def test_the_refusal_is_read_when_it_arrives_before_the_upload_ends(
    refusing_port: int,
) -> None:
    caller = load_caller()

    answer = caller.reply(
        specification(refusing_port, bytes(_UPLOAD_BYTES)), _ANSWER_TIMEOUT_SECONDS
    )

    assert answer["status"] == HTTPStatus.UNAUTHORIZED
    assert base64.b64decode(str(answer["body"])) == _REFUSAL


def test_a_connection_nothing_listens_on_is_named_by_its_error() -> None:
    caller = load_caller()

    answer = caller.reply(specification(free_loopback_port(), b""))

    assert answer == {"error": REFUSED_CONNECTION}
