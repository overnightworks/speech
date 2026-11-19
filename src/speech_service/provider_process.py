"""The service's own child: Qwen's launcher, which becomes the vLLM-Omni server."""

from __future__ import annotations

import http.client
import logging
import os
import queue
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, StrEnum
from http import HTTPStatus
from pathlib import Path
from typing import IO, TYPE_CHECKING

from speech_service.failures import PayloadFreeError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

TERMINATE_GRACE_SECONDS = 5
HEALTH_DEADLINE_SECONDS = 600
"""How long a server may take to answer its health check, loading included."""
_HEALTH_POLL_SECONDS = 0.25
_HEALTH_REQUEST_SECONDS = 2.0
_COMMAND_WAIT_SECONDS = 0.25
_READ_SIZE = 64 * 1024
# A failed vLLM-Omni start prints about 11 KiB: the root cause, then a long traceback.
_STDERR_TAIL_BYTES = 64 * 1024
_STDERR_DRAIN_JOIN_SECONDS = 5
SYNTHESIS_FAILED = "speech provider synthesis failed"
"""The message of a failed synthesis."""
_SERVER_DIRECTORY_PREFIX = "speech-provider-"
"""The name prefix of each server's private directory in the temporary directory.

It holds the server's socket and is removed before the server's
ended-before-ready record is logged.
"""
_SOCKET_NAME = "server.sock"
_HEALTH_PATH = "/health"
_NVIDIA_LIBRARY_DIRECTORIES = (
    "cublas:cuda_cupti:cuda_nvrtc:cuda_runtime:cudnn:cufft:curand:"
    "cusolver:cusparse:cusparselt:nccl:nvjitlink:nvtx"
)


class ProviderFailedError(RuntimeError):
    """The provider failed and cannot serve requests.

    The provider's startup stderr travels as a note, which the layer that stops
    the exception logs once through `exc_info`.
    """

    def __init__(self, startup_stderr: str = "") -> None:
        """Carry the one message every failed provider reports."""
        super().__init__("speech provider failed")
        if startup_stderr:
            self.add_note(startup_stderr)


class ProviderSynthesisFailedError(PayloadFreeError):
    """The server failed one synthesis request."""

    def __init__(self) -> None:
        """Carry the one message every failed synthesis reports."""
        super().__init__(SYNTHESIS_FAILED)


class ProviderCleanupFailedError(RuntimeError):
    """The provider process could not be terminated and reaped."""

    def __init__(self) -> None:
        """Carry the one message every failed cleanup reports."""
        super().__init__("speech provider cleanup failed")


class ProviderUnavailableError(RuntimeError):
    """The configured provider executable cannot be launched."""

    def __init__(self) -> None:
        """Carry the one message every unavailable provider reports."""
        super().__init__("speech provider unavailable")


class ProviderNotReadyError(RuntimeError):
    """The provider is not in a state that accepts requests."""

    def __init__(self) -> None:
        """Carry the one message every not-ready provider reports."""
        super().__init__("speech provider is not ready")


class UnsupportedLanguageError(ValueError):
    """The requested language is not one the provider speaks."""

    def __init__(self) -> None:
        """Carry the one message every unsupported language reports."""
        super().__init__("unsupported speech provider language")


class ProviderId(StrEnum):
    """The provider directories and executable suffixes the service may use."""

    QWEN = "qwen"


class ProviderStartFailure(StrEnum):
    """Why a provider ended before it was ready."""

    EXITED = "exited"
    """The provider ended by itself."""
    START_FAILED = "start_failed"
    """A signal ended the provider: the service stopped it or the system killed it."""
    STDERR_OVERSIZED = "stderr_oversized"
    """The provider wrote more startup stderr than the service keeps."""


class _OwnerExit(Enum):
    ORDINARY_CLOSE = "ordinary_close"
    PROVIDER_FAILURE = "provider_failure"


class _StderrTail:
    """The last bytes a server wrote to stderr, kept only until it is ready."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tail = b""
        self._size = 0
        self._kept = True

    def feed(self, chunk: bytes) -> None:
        """Keep `chunk` and trim to the tail while the server is not ready."""
        with self._lock:
            if self._kept:
                self._size += len(chunk)
                self._tail = (self._tail + chunk)[-_STDERR_TAIL_BYTES:]

    def discard(self) -> None:
        """Drop what was kept and keep nothing from now on."""
        with self._lock:
            self._kept = False
            self._tail = b""

    @property
    def kept(self) -> bool:
        """Whether the server is not ready yet, so its stderr is still kept."""
        with self._lock:
            return self._kept

    @property
    def size(self) -> int:
        """How many bytes the server wrote while the tail was kept."""
        with self._lock:
            return self._size

    @property
    def text(self) -> str:
        """The kept tail."""
        with self._lock:
            return self._tail.decode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class ProviderLaunch:
    """Immutable launcher path and arguments of one isolated provider."""

    executable: Path | None
    arguments: tuple[str, ...]


class ProviderProcess:
    """One retained server process, owned by a dedicated creator thread."""

    def __init__(
        self,
        launch: ProviderLaunch,
        *,
        fatal_callback: Callable[[], None] | None = None,
    ) -> None:
        """Store immutable launch data; construction never creates a process."""
        self._executable = launch.executable
        self._arguments = launch.arguments
        self._fatal_callback = fatal_callback or (lambda: None)
        self.ready = False
        self._state_lock = threading.Lock()
        self._commands: queue.Queue[_OwnerExit] = queue.Queue(maxsize=1)
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._owner: threading.Thread | None = None
        self._socket_path: Path | None = None
        self._startup_stderr = ""
        self._failed = False
        self._closing = False
        self._cleanup_failed = False

    def load(self) -> None:
        """Start the stable creator and wait until its server answers health."""
        with self._state_lock:
            if self._failed or self._closing:
                raise ProviderFailedError
            if self.ready:
                return
            if self._owner is None:
                self._owner = threading.Thread(
                    target=self._run_owner,
                    name="speech-provider-owner",
                )
                self._owner.start()
        self._ready.wait()
        with self._state_lock:
            ready = self.ready
        if not ready:
            raise ProviderFailedError(self._startup_stderr)

    def connect(self, timeout: float) -> http.client.HTTPConnection:
        """Open a connection to the ready server; its reads give up after `timeout`.

        `OSError` when the server no longer listens.
        """
        with self._state_lock:
            socket_path = self._socket_path
            if socket_path is None or not self.ready or self._closing or self._failed:
                raise ProviderNotReadyError
        return _connect_unix(socket_path, timeout)

    def mark_failed(self) -> None:
        """Stop serving: the server is stopped and the next load starts a new one."""
        self._set_failed()
        self._request_owner_stop(_OwnerExit.PROVIDER_FAILURE)

    def close(self) -> None:
        """Ask the owner to terminate and reap; this operation is idempotent."""
        with self._state_lock:
            self._closing = True
            owner = self._owner
        if owner is None:
            return
        self._request_owner_stop(_OwnerExit.ORDINARY_CLOSE)
        if not self._stopped.wait((2 * TERMINATE_GRACE_SECONDS) + 2):
            self._fatal_callback()
            raise ProviderCleanupFailedError
        owner.join(timeout=1)
        if owner.is_alive() or self._cleanup_failed:
            self._fatal_callback()
            raise ProviderCleanupFailedError

    def set_fatal_callback(self, callback: Callable[[], None]) -> None:
        """Give the runtime its process-lifecycle failure notification seam."""
        with self._state_lock:
            self._fatal_callback = callback

    def _run_owner(self) -> None:
        """Own one server; its private directory goes once the server is reaped."""
        child: subprocess.Popen[bytes] | None = None
        scratch: tempfile.TemporaryDirectory[str] | None = None
        tail = _StderrTail()
        drain: threading.Thread | None = None
        provider_failed = False
        lifecycle_failed = False
        try:
            scratch = tempfile.TemporaryDirectory(prefix=_SERVER_DIRECTORY_PREFIX)
            socket_path = Path(scratch.name) / _SOCKET_NAME
            child = self._spawn(socket_path, Path(scratch.name))
            drain = threading.Thread(
                target=_drain_stderr,
                args=(_require_stderr(child), tail),
                name="speech-provider-stderr",
                daemon=True,
            )
            drain.start()
            with self._state_lock:
                self._socket_path = socket_path
                closing = self._closing
            outcome = (
                _OwnerExit.ORDINARY_CLOSE
                if closing
                else self._serve_owner(child, socket_path, tail)
            )
            if outcome is _OwnerExit.PROVIDER_FAILURE:
                self._set_failed()
                provider_failed = True
        except Exception:
            logger.exception("provider_owner.failed")
            lifecycle_failed = True
            self._set_failed()
        finally:
            lifecycle_failed |= self._release(child, scratch, drain)
            self._startup_stderr = _log_exit(
                child, tail, provider_failed=provider_failed
            )
            with self._state_lock:
                self._socket_path = None
                self.ready = False
            self._ready.set()
            self._stopped.set()
            if child is not None and lifecycle_failed:
                self._fatal_callback()

    def _release(
        self,
        child: subprocess.Popen[bytes] | None,
        scratch: tempfile.TemporaryDirectory[str] | None,
        drain: threading.Thread | None,
    ) -> bool:
        """Reap the server, remove its directory and stderr; whether reaping failed."""
        reaping_failed = False
        try:
            self._terminate_and_reap(child)
        except Exception:
            logger.exception("provider_cleanup.failed")
            reaping_failed = True
            self._cleanup_failed = True
        if scratch is not None:
            try:
                scratch.cleanup()
            except Exception:
                logger.exception("provider_scratch_cleanup.failed")
        if drain is not None and child is not None and child.stderr is not None:
            drain.join(timeout=_STDERR_DRAIN_JOIN_SECONDS)
            if not drain.is_alive():
                child.stderr.close()
        return reaping_failed

    def _serve_owner(
        self, child: subprocess.Popen[bytes], socket_path: Path, tail: _StderrTail
    ) -> _OwnerExit:
        outcome = self._await_health(child, socket_path)
        if outcome is not None:
            return outcome
        tail.discard()
        with self._state_lock:
            self.ready = True
        self._ready.set()
        while True:
            try:
                command = self._commands.get(timeout=_COMMAND_WAIT_SECONDS)
            except queue.Empty:
                if child.poll() is not None:
                    return _OwnerExit.PROVIDER_FAILURE
                continue
            return command

    def _await_health(
        self, child: subprocess.Popen[bytes], socket_path: Path
    ) -> _OwnerExit | None:
        """Wait for the child's server to answer; `None` once it does while alive."""
        deadline = time.monotonic() + HEALTH_DEADLINE_SECONDS
        while True:
            if child.poll() is not None:
                return _OwnerExit.PROVIDER_FAILURE
            if _is_healthy(socket_path) and child.poll() is None:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _OwnerExit.PROVIDER_FAILURE
            with suppress(queue.Empty):
                return self._commands.get(timeout=min(_HEALTH_POLL_SECONDS, remaining))

    def _spawn(self, socket_path: Path, scratch: Path) -> subprocess.Popen[bytes]:
        entrypoint = self._executable
        if not is_executable_file(entrypoint):
            raise ProviderUnavailableError
        directory = entrypoint.parents[2]
        return subprocess.Popen(  # noqa: S603 - argv is the fixed provider entrypoint and flags with no shell
            [
                str(entrypoint),
                "--expected-parent-pid",
                str(os.getpid()),
                *self._arguments,
                "--socket",
                str(socket_path),
            ],
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=directory,
            env=provider_environment(directory, scratch),
            bufsize=0,
            close_fds=True,
            start_new_session=True,
        )

    def _set_failed(self) -> None:
        with self._state_lock:
            self._failed = True
            self.ready = False

    def _request_owner_stop(self, command: _OwnerExit) -> None:
        with suppress(queue.Full):
            self._commands.put_nowait(command)

    def _terminate_and_reap(self, child: subprocess.Popen[bytes] | None) -> None:
        if child is None:
            return
        # The launcher leads its process group, and the server it becomes keeps
        # its stage processes in that group: they outlive a server that died
        # first, so the group is signalled whether or not the server is alive.
        _signal_group(child.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            child.wait(timeout=TERMINATE_GRACE_SECONDS)
        _signal_group(child.pid, signal.SIGKILL)
        child.wait(timeout=TERMINATE_GRACE_SECONDS)


def _signal_group(group: int, signal_number: signal.Signals) -> None:
    with suppress(ProcessLookupError):
        os.killpg(group, signal_number)


def _connect_unix(socket_path: Path, timeout: float) -> http.client.HTTPConnection:
    """Open an HTTP connection over the Unix socket at `socket_path`.

    `OSError` when nothing listens there; every read gives up after `timeout`.
    """
    unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        unix_socket.settimeout(timeout)
        unix_socket.connect(str(socket_path))
    except BaseException:
        unix_socket.close()
        raise
    connection = http.client.HTTPConnection("localhost", timeout=timeout)
    connection.sock = unix_socket
    return connection


def _is_healthy(socket_path: Path) -> bool:
    try:
        connection = _connect_unix(socket_path, _HEALTH_REQUEST_SECONDS)
    except OSError:
        return False
    try:
        connection.request("GET", _HEALTH_PATH)
        return connection.getresponse().status == HTTPStatus.OK
    except OSError, http.client.HTTPException:
        return False
    finally:
        connection.close()


def _drain_stderr(stream: IO[bytes], tail: _StderrTail) -> None:
    """Read stderr until it ends, so a full pipe never blocks the server."""
    with suppress(OSError, ValueError):
        while chunk := stream.read(_READ_SIZE):
            tail.feed(chunk)


def _log_exit(
    child: subprocess.Popen[bytes] | None,
    tail: _StderrTail,
    *,
    provider_failed: bool,
) -> str:
    """Log how a provider exited and return the startup stderr for the failure.

    A provider that never became ready is logged with its exit status and a
    closed reason, and its stderr tail is returned for the exception that
    reaches the stopping layer; one that was ready and then failed has only its
    exit status logged, and the text is empty.

    Call after the provider is reaped. The exit status is the provider's own
    when it ended by itself; a negative number is the signal that stopped it,
    which is -15 when the service stopped it.
    """
    if child is None:
        return ""
    if tail.kept:
        logger.warning(
            "provider.exited",
            extra={
                "phase": "before_ready",
                "exit_status": child.returncode,
                "reason": _start_failure(child.returncode, tail.size),
            },
        )
        return tail.text
    if provider_failed:
        logger.warning(
            "provider.exited",
            extra={"phase": "after_ready", "exit_status": child.returncode},
        )
    return ""


def _start_failure(exit_status: int, stderr_size: int) -> ProviderStartFailure:
    if stderr_size > _STDERR_TAIL_BYTES:
        return ProviderStartFailure.STDERR_OVERSIZED
    if exit_status < 0:
        return ProviderStartFailure.START_FAILED
    return ProviderStartFailure.EXITED


def _require_stderr(child: subprocess.Popen[bytes]) -> IO[bytes]:
    if child.stderr is None:
        raise ProviderFailedError
    return child.stderr


def provider_entrypoint(
    provider_root: Path | None, provider: ProviderId
) -> Path | None:
    """Return one closed launcher path without searching another location."""
    if provider_root is None:
        return None
    return (
        provider_root
        / provider.value
        / ".venv"
        / "bin"
        / (f"speech-{provider.value}-server")
    )


def is_executable_file(entrypoint: Path | None) -> bool:
    """Whether `entrypoint` names a regular file this process may execute."""
    if entrypoint is None:
        return False
    try:
        mode = entrypoint.stat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) and os.access(entrypoint, os.X_OK)


def provider_environment(provider_directory: Path, scratch: Path) -> dict[str, str]:
    """Return the minimal offline environment, with the venv's CUDA libraries.

    `scratch` is the server's private writable directory.
    """
    environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PATH": os.defpath,
        # A numba function compiled with a cache refuses to import without a
        # writable cache directory, and the image's root is read-only.
        "NUMBA_CACHE_DIR": str(scratch),
        # Libraries the server imports leave files in the default temporary
        # directory (onnxruntime a log per process, torch its compile cache),
        # which would outlive the server in the service's own small /tmp.
        "TMPDIR": str(scratch),
        # Torch's native operations would launch triton kernels, and triton
        # builds a C module at first use.
        "TORCH_DISABLE_NATIVE_JIT": "1",
        # The server's FlashInfer sampler JIT-compiles with nvcc, which the
        # image does not carry; its PyTorch sampler needs no CUDA toolkit.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # The Qwen server keeps its compile and kernel caches under the home
        # directory, which the image's read-only root does not let it create.
        "HOME": str(scratch),
        # The server reports usage to its vendor unless told not to.
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
    }
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices
    site_packages_directories = sorted(
        (provider_directory / ".venv" / "lib").glob("python3.*/site-packages")
    )
    libraries = (
        site_packages / "nvidia" / package / "lib"
        for site_packages in site_packages_directories
        for package in _NVIDIA_LIBRARY_DIRECTORIES.split(":")
    )
    existing = [str(directory) for directory in libraries if directory.is_dir()]
    if existing:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(existing)
    return environment
