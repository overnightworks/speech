"""The vLLM-Omni server child the worker speaks to over a Unix socket."""

from __future__ import annotations

import contextlib
import errno
import http.client
import os
import shutil
import socket
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import IO, TYPE_CHECKING

from speech_qwen.parent_death import install_parent_death_signal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

START_TIMEOUT_SECONDS = 600
"""How long a server may take to answer its health check, loading included."""
_HEALTH_POLL_SECONDS = 1.0
_HEALTH_REQUEST_SECONDS = 2.0
_STOP_GRACE_SECONDS = 5.0
_STDERR_READ_BYTES = 64 * 1024
_STDERR_DESCRIPTOR = 2
_SHARED_MEMORY_DIRECTORY = Path("/dev", "shm")
"""Where the server's shared-memory connector keeps its wake directories."""
_OWN_USER_ID = os.getuid()
STDERR_THREAD_NAME = "speech-qwen-server-stderr"
_WATCH_THREAD_NAME = "speech-qwen-server-watch"


class ServerStartError(RuntimeError):
    """The server ended or stayed unhealthy before it could speak."""


class _UnixConnection(http.client.HTTPConnection):
    """An HTTP connection over the Unix socket the server listens on."""

    def __init__(self, socket_path: Path, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:
        """Connect to the socket path instead of a host and port."""
        unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            unix_socket.settimeout(self.timeout)
            unix_socket.connect(str(self._socket_path))
        except BaseException:
            unix_socket.close()
            raise
        self.sock = unix_socket


@dataclass(frozen=True, slots=True)
class Server:
    """A healthy server: its process and the socket it listens on."""

    process: subprocess.Popen[bytes]
    socket_path: Path

    def connect(self, timeout: float) -> http.client.HTTPConnection:
        """Open a connection whose reads give up after `timeout` seconds."""
        return _UnixConnection(self.socket_path, timeout)


def fresh_socket_path() -> Path:
    """Return a socket path no earlier start used, in a directory only we enter."""
    return Path(tempfile.mkdtemp(prefix="speech-qwen-server-")) / "server.sock"


def exit_worker() -> None:
    """End the worker at once: without its server it cannot speak."""
    os._exit(1)


def start_server(
    command: Sequence[str],
    socket_path: Path,
    *,
    shared_memory_directory: Path = _SHARED_MEMORY_DIRECTORY,
    user_id: int = _OWN_USER_ID,
    when_server_ends: Callable[[], None] = exit_worker,
) -> Server:
    """Start `command`, which listens on `socket_path`, and return once it is healthy.

    Call on the worker's main thread before any other thread exists: the parent
    death signal follows the thread that created the child. What an earlier
    server of `user_id` left in `shared_memory_directory` is removed first. Once
    the server is healthy, `when_server_ends` runs when its process ends.
    """
    _remove_leftover_shared_memory(shared_memory_directory, user_id)
    process = subprocess.Popen(  # noqa: S603 - argv is the closed server command with no shell
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        close_fds=True,
        preexec_fn=partial(install_parent_death_signal, os.getpid()),  # noqa: PLW1509 - no thread exists yet so the signal is set before exec
    )
    threading.Thread(
        target=_drain_stderr,
        args=(process.stderr,),
        name=STDERR_THREAD_NAME,
        daemon=True,
    ).start()
    server = Server(process, socket_path)
    try:
        _wait_until_healthy(server)
    except BaseException:
        _stop(process)
        raise
    threading.Thread(
        target=_call_when_ended,
        args=(process, when_server_ends),
        name=_WATCH_THREAD_NAME,
        daemon=True,
    ).start()
    return server


def _call_when_ended(
    process: subprocess.Popen[bytes], when_server_ends: Callable[[], None]
) -> None:
    process.wait()
    when_server_ends()


def _remove_leftover_shared_memory(directory: Path, user_id: int) -> None:
    """Remove the wake directories of `user_id` that no process reads from any more.

    The directory is writable by every user, so an entry counts only when it is
    a real directory (a symbolic link is never followed) that `user_id` owns.
    """
    for entry in directory.glob("omni_shm_*"):
        status = entry.lstat()
        if (
            stat.S_ISDIR(status.st_mode)
            and status.st_uid == user_id
            and not any(_is_read(fifo) for fifo in entry.iterdir())
        ):
            shutil.rmtree(entry)


def _is_read(fifo: Path) -> bool:
    """Whether a live process has `fifo` open for reading.

    Opening a FIFO for writing without waiting fails with ENXIO exactly when no
    reader holds it, and a killed server's readers are gone with it. What cannot
    be judged counts as read, so it stays.
    """
    try:
        descriptor = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as error:
        return error.errno != errno.ENXIO
    os.close(descriptor)
    return True


def _drain_stderr(stream: IO[bytes]) -> None:
    """Hand the server's stderr to the worker's own, the service's startup file."""
    with stream:
        for chunk in iter(partial(stream.read1, _STDERR_READ_BYTES), b""):
            with contextlib.suppress(OSError):
                os.write(_STDERR_DESCRIPTOR, chunk)


def _wait_until_healthy(server: Server) -> None:
    """Return once the server answers; it counts only while its process is alive."""
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while True:
        if server.process.poll() is not None:
            raise ServerStartError
        if _is_healthy(server):
            return
        if time.monotonic() >= deadline:
            raise ServerStartError
        with contextlib.suppress(subprocess.TimeoutExpired):
            server.process.wait(timeout=_HEALTH_POLL_SECONDS)


def _is_healthy(server: Server) -> bool:
    connection = server.connect(_HEALTH_REQUEST_SECONDS)
    try:
        connection.request("GET", "/health")
        return connection.getresponse().status == HTTPStatus.OK
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _stop(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=_STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
