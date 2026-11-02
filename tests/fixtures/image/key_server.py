"""Serve the proof's key set over TLS, as the identity issuer does for the service."""

import http.server
import os
import ssl
from pathlib import Path

KEY_SET_PATH = os.environ["PROOF_KEY_SET_PATH"]
DIRECTORY = Path("/key-server")
HTTPS_PORT = 443


class Handler(http.server.BaseHTTPRequestHandler):
    """Answer the key set's path, and nothing else."""

    def do_GET(self) -> None:
        """Serve the key set at its path; any other path is not found."""
        if self.path != KEY_SET_PATH:
            self.send_error(http.HTTPStatus.NOT_FOUND)
            return
        body = (DIRECTORY / "key-set.json").read_bytes()
        self.send_response(http.HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(DIRECTORY / "tls.crt", DIRECTORY / "tls.key")
server = http.server.ThreadingHTTPServer(("0.0.0.0", HTTPS_PORT), Handler)  # noqa: S104 - the proof's isolated network
server.socket = context.wrap_socket(server.socket, server_side=True)
server.serve_forever()
