"""Plays the Qwen worker around a fake server, run as its own process."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from speech_qwen.server import STDERR_THREAD_NAME, Server, start_server

_FAKE = Path(__file__).with_name("fake_vllm_omni.py")


def start_fake_server(mode: str, directory: Path, *extra: str) -> Server:
    """Start the fake server in `mode` on a socket in `directory`."""
    socket_path = directory / "server.sock"
    return start_server(
        [sys.executable, str(_FAKE), mode, "--uds", str(socket_path), *extra],
        socket_path,
        shared_memory_directory=directory / "shm",
    )


def _wait_to_be_killed(directory: Path) -> None:
    server = start_fake_server("healthy", directory)
    sys.stdout.write(f"{server.process.pid}\n")
    sys.stdout.flush()
    threading.Event().wait()


def _discard_stderr_once_started(directory: Path) -> None:
    server = start_fake_server("healthy", directory)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 2)
    for path in ("/noise", "/quit"):
        connection = server.connect(5)
        connection.request("GET", path)
        connection.getresponse().read()
    server.process.wait()
    for thread in threading.enumerate():
        if thread.name == STDERR_THREAD_NAME:
            thread.join()


if __name__ == "__main__":
    {"kill": _wait_to_be_killed, "discard": _discard_stderr_once_started}[sys.argv[1]](
        Path(sys.argv[2])
    )
