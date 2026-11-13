"""A stand-in for `vllm-omni serve` that listens on --uds and answers /health."""

import argparse
import http.server
import os
import signal
import socketserver
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["healthy", "unhealthy", "exits"])
parser.add_argument("--uds", required=True)
parser.add_argument("--pid-file")
arguments = parser.parse_args()

print("child-stdout-noise", flush=True)  # noqa: T201 - the test checks it reaches no output
print("before-ready", file=sys.stderr, flush=True)  # noqa: T201 - the test checks it arrives
if arguments.pid_file:
    Path(arguments.pid_file).write_text(str(os.getpid()), encoding="ascii")
if arguments.mode == "exits":
    sys.exit(3)
if arguments.mode == "unhealthy":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


class Handler(http.server.BaseHTTPRequestHandler):
    """Answers /health, and /noise and /quit for the tests."""

    def log_message(self, *_arguments: object) -> None:
        """Keep the output free of access logs."""

    def do_GET(self) -> None:
        """Answer by the server's mode."""
        if self.path == "/noise":
            print("after-ready", file=sys.stderr, flush=True)  # noqa: T201 - the test checks it is discarded
        healthy = arguments.mode == "healthy"
        self.send_response(200 if healthy else 503)
        self.send_header("Content-Length", "0")
        self.end_headers()
        if self.path == "/quit":
            os._exit(0)


socketserver.ThreadingUnixStreamServer(arguments.uds, Handler).serve_forever()
