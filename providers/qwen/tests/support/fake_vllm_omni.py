"""A stand-in for `vllm-omni serve` that listens on --uds and answers /health."""

import argparse
import http.server
import multiprocessing
import os
import signal
import socketserver
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["healthy", "unhealthy", "exits", "stages"])
parser.add_argument("--uds", required=True)
parser.add_argument("--pid-file")
parser.add_argument("--stage-pid-file")
parser.add_argument("--stage-gate-file")
arguments = parser.parse_args()

print("child-stdout-noise", flush=True)  # noqa: T201 - the test checks it reaches no output
print("before-ready", file=sys.stderr, flush=True)  # noqa: T201 - the test checks it arrives
if arguments.pid_file:
    Path(arguments.pid_file).write_text(str(os.getpid()), encoding="ascii")
if arguments.mode == "exits":
    sys.exit(3)
if arguments.mode == "unhealthy":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

_STAGE = """
import ctypes, signal, sys, time
from pathlib import Path

while not Path(sys.argv[1]).exists():
    time.sleep(0.01)
ctypes.CDLL(None).prctl(1, signal.SIGTERM)
time.sleep(3600)
"""


def start_stage() -> None:
    """Start as vLLM does: a resource tracker, then a stage arming its signal late."""
    multiprocessing.get_context("spawn").Semaphore()
    stage = subprocess.Popen([sys.executable, "-c", _STAGE, arguments.stage_gate_file])  # noqa: S603 - the stage is the program of this stand-in
    Path(arguments.stage_pid_file).write_text(str(stage.pid), encoding="ascii")


if arguments.mode == "stages":
    start_stage()


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
