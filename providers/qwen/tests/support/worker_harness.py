"""Plays the Qwen worker around a fake server, run as its own process."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from speech_qwen.server import STDERR_THREAD_NAME, Server, exit_worker, start_server

if TYPE_CHECKING:
    from collections.abc import Callable

_USER_ID = os.getuid()
_FAKE = Path(__file__).with_name("fake_vllm_omni.py")


def _ignore() -> None:
    """Leave the process to the test when the server ends."""


def start_fake_server(
    mode: str,
    directory: Path,
    *extra: str,
    when_server_ends: Callable[[], None] = _ignore,
    user_id: int = _USER_ID,
) -> Server:
    """Start the fake server in `mode` on a socket in `directory`."""
    socket_path = directory / "server.sock"
    return start_server(
        [sys.executable, str(_FAKE), mode, "--uds", str(socket_path), *extra],
        socket_path,
        shared_memory_directory=directory / "shm",
        user_id=user_id,
        when_server_ends=when_server_ends,
    )


def _serve_idle(directory: Path) -> None:
    server = start_fake_server("healthy", directory, when_server_ends=exit_worker)
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
    {"idle": _serve_idle, "discard": _discard_stderr_once_started}[sys.argv[1]](
        Path(sys.argv[2])
    )
