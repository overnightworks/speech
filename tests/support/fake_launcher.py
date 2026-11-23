"""A stand-in for `speech-qwen-server`: it listens on --socket like the real server.

The behaviour is the words in `behavior` under --cache. Everything it observes is
written under --cache for the tests to read.
"""

import argparse
import ctypes
import http.server
import json
import os
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

PR_SET_PDEATHSIG = 1

parser = argparse.ArgumentParser()
parser.add_argument("--expected-parent-pid", type=int, required=True)
parser.add_argument("--cache", type=Path, required=True)
parser.add_argument("--socket", type=Path, required=True)
arguments = parser.parse_args()
cache = arguments.cache
cache.mkdir(parents=True, exist_ok=True)
behaviors = set((cache / "behavior").read_text(encoding="utf-8").split())
if (
    ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0
    or os.getppid() != arguments.expected_parent_pid
):
    raise SystemExit(1)


def wait_for(path: Path) -> None:
    while not path.exists():
        time.sleep(0.01)


if (cache / "hold-exit").exists():

    def exit_when_released(_signal: int, _frame: object) -> None:
        (cache / f"exit-held-{os.getpid()}").touch()
        wait_for(cache / f"release-exit-{os.getpid()}")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, exit_when_released)
with (cache / "server-starts").open("a", encoding="utf-8") as starts:
    starts.write(f"{os.getpid()}\n")
(cache / "server-observed").write_text(
    repr(str(Path.cwd()))
    + "\n"
    + repr(os.environ.copy())
    + "\n"
    + repr(os.getpid())
    + "\n"
    + repr({**vars(arguments), "cache": str(cache), "socket": str(arguments.socket)}),
    encoding="utf-8",
)
(cache / "socket-directory-mode").write_text(
    str(arguments.socket.parent.stat().st_mode & 0o777), encoding="utf-8"
)
numba_cache = os.environ.get("NUMBA_CACHE_DIR")
if numba_cache is not None:
    (Path(numba_cache) / "index").touch()
    (cache / "numba-cache-mode").write_text(
        str(Path(numba_cache).stat().st_mode & 0o777), encoding="utf-8"
    )
debug_log = Path(tempfile.gettempdir()) / f"server-debug-{os.getpid()}.log"
debug_log.touch()
(cache / "server-debug-log").write_text(str(debug_log), encoding="utf-8")
print("server noise before ready", file=sys.stderr, flush=True)  # noqa: T201 - the tests read what reaches the service

if "term-resistant" in behaviors:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if "held-ready" in behaviors:
    release = cache / f"release-{os.getpid()}"
    wait_for(release)
    outcome = release.read_text(encoding="utf-8")
    if outcome == "exit":
        raise SystemExit(1)
    if outcome == "failed":
        while True:
            time.sleep(1)
if "startup-exit" in behaviors:
    raise SystemExit(1)
if "startup-oversized" in behaviors:
    sys.stderr.write("ROOT CAUSE: engine core failed\n")
    sys.stderr.write("Traceback line\n" * (1024 * 1024 // 15))
    sys.stderr.flush()
    raise SystemExit(1)
if "stale-socket" in behaviors:
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(arguments.socket))
    (cache / "scenario-begun").touch()
    while True:
        time.sleep(1)
if "with-stage" in behaviors:
    stage = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    (cache / "stage").write_text(str(stage.pid), encoding="utf-8")


def chunked(data: bytes) -> bytes:
    """Frame `data` as one chunk of a chunked response."""
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


class Handler(http.server.BaseHTTPRequestHandler):
    """Answers /health and the speech request by the scripted word in its text."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_arguments: object) -> None:
        """Keep the output free of access logs."""

    def do_GET(self) -> None:
        """Answer the health check."""
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        """Answer a speech request as a chunked stream the text scripts."""
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        with (cache / "requests").open("a", encoding="utf-8") as requests:
            requests.write(json.dumps(request) + "\n")
        text = request["input"]
        print(text, file=sys.stderr, flush=True)  # noqa: T201 - the tests read what reaches the service
        if text == "chatty":
            sys.stderr.write("x" * 1024 * 1024)
            sys.stderr.flush()
        if "hold-first-speech" in behaviors and not (cache / "speech-held").exists():
            (cache / "speech-held").touch()
            wait_for(cache / "release-speech")
        if text == "refused" or "refuse-speech" in behaviors:
            self.send_response(500)
            self.send_header("Content-Length", str(len(text)))
            self.end_headers()
            self.wfile.write(text.encode())
            return
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        if text != "no-rate":
            rate = 48000 if "other-rate" in behaviors else 24000
            self.send_header("X-Audio-Sample-Rate", str(rate))
        self.end_headers()
        answer = {"odd": (b"\x00\x01\x02",), "split-sample": (b"\x00", b"\x01")}
        for chunk in answer.get(text, (b"\x00\x01",)):
            self.wfile.write(chunked(chunk))
            self.wfile.flush()
        if text == "midstream-exit":
            os._exit(1)
        if text == "gate":
            wait_for(cache / "release-second")
            self.wfile.write(chunked(b"\x02\x03"))
        if text == "wait-disconnect":
            self.connection.recv(1)
            (cache / "disconnect-observed").touch()
            return
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def exit_when_told() -> None:
    wait_for(cache / "exit-now")
    os._exit(1)


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """One thread per connection."""

    daemon_threads = True


if "idle-exit" in behaviors:
    threading.Thread(target=exit_when_told, daemon=True).start()
Server(str(arguments.socket), Handler).serve_forever()
