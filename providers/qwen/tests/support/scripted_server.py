"""A speech server on a Unix socket that streams the parts a test scripts."""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
from http import HTTPStatus
from typing import TYPE_CHECKING

from speech_qwen.server import Server

if TYPE_CHECKING:
    from pathlib import Path

HANG_UP = object()
"""A part that makes the server wait for the client to close the connection."""


class ScriptedSpeechServer:
    """Answers a speech request with the scripted parts, one HTTP chunk each."""

    def __init__(self, socket_path: Path) -> None:
        """Listen on `socket_path` and answer with `parts` once a test sets them."""
        self.requests: list[dict[str, object]] = []
        self.parts: list[bytes | threading.Event | object] = []
        self.status = HTTPStatus.OK
        self.sent = 0
        self.hung_up = threading.Event()
        self.server = Server(process=None, socket_path=socket_path)
        scripted = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_arguments: object) -> None:
                """Keep the test output free of access logs."""

            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                scripted.requests.append(json.loads(self.rfile.read(length)))
                if scripted.status != HTTPStatus.OK:
                    self.send_response(scripted.status)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for part in scripted.parts:
                    if part is HANG_UP:
                        self.connection.recv(1)
                        scripted.hung_up.set()
                        return
                    if isinstance(part, threading.Event):
                        part.wait(10)
                        continue
                    scripted.sent += 1
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")

        class Listener(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True

        self._listener = Listener(str(socket_path), Handler)
        threading.Thread(target=self._listener.serve_forever, daemon=True).start()

    def close(self) -> None:
        """Stop listening."""
        self._listener.shutdown()
        self._listener.server_close()
