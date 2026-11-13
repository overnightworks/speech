"""The server child answers on a fresh socket and dies with its worker."""

from __future__ import annotations

import ctypes
import hashlib
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from speech_qwen import server as server_module
from speech_qwen.parent_death import install_parent_death_signal
from speech_qwen.server import ServerStartError, fresh_socket_path
from support.worker_harness import start_fake_server

if TYPE_CHECKING:
    from collections.abc import Iterator

PR_SET_PDEATHSIG = 1
PR_GET_PDEATHSIG = 2
_HARNESS = Path(__file__).parent / "support" / "worker_harness.py"


def _start(mode: str, tmp_path: Path, *extra: str) -> server_module.Server:
    return start_fake_server(mode, tmp_path, *extra)


def _start_and_stop(tmp_path: Path, **options: int) -> None:
    process = start_fake_server("healthy", tmp_path, **options).process
    process.kill()
    process.wait()


def _has_ended(pid: int) -> bool:
    """Whether `pid` is gone or a zombie that its adopter has yet to reap."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            status = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        except FileNotFoundError:
            return True
        if status.rpartition(")")[2].split()[0] == "Z":
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def started(tmp_path: Path) -> Iterator[server_module.Server]:
    (tmp_path / "shm").mkdir()
    server = _start("healthy", tmp_path)
    yield server
    server.process.kill()
    server.process.wait()


def test_a_healthy_server_stays_in_the_workers_process_group(
    started: server_module.Server,
) -> None:
    assert os.getpgid(started.process.pid) == os.getpgrp()


def test_a_server_that_ends_before_it_answers_fails_the_start(
    tmp_path: Path,
) -> None:
    (tmp_path / "shm").mkdir()

    with pytest.raises(ServerStartError):
        _start("exits", tmp_path)


def test_a_server_that_never_answers_fails_the_start_and_is_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "shm").mkdir()
    monkeypatch.setattr(server_module, "START_TIMEOUT_SECONDS", 1.5)
    pid_file = tmp_path / "pid"

    with pytest.raises(ServerStartError):
        _start("unhealthy", tmp_path, "--pid-file", str(pid_file))

    assert _has_ended(int(pid_file.read_text(encoding="ascii")))


def test_a_bind_that_fails_is_a_failed_start(tmp_path: Path) -> None:
    (tmp_path / "shm").mkdir()
    taken = tmp_path / "server.sock"

    with socket.socket(socket.AF_UNIX) as occupant:
        occupant.bind(str(taken))

        with pytest.raises(ServerStartError):
            _start("healthy", tmp_path)


def test_every_start_gets_a_socket_nothing_listens_on() -> None:
    first, second = fresh_socket_path(), fresh_socket_path()

    assert first != second
    assert not first.exists()
    assert stat.S_IMODE(first.parent.stat().st_mode) == stat.S_IRWXU


def _wake_directory(shm: Path, name: str) -> Path:
    """Make a wake directory holding one FIFO, as the server's connector does."""
    directory = shm / name
    directory.mkdir(parents=True, mode=0o700)
    os.mkfifo(directory / "0_fifo", 0o600)
    return directory


def test_leftovers_of_servers_that_are_gone_are_removed_at_the_next_start(
    tmp_path: Path,
) -> None:
    shm = tmp_path / "shm"
    scope = hashlib.sha256(b"a launch scope").hexdigest()[:24]
    leftovers = [
        _wake_directory(shm, f"omni_shm_wake_{os.getuid()}_{scope}"),
        _wake_directory(shm, f"omni_shm_wake_{os.getuid()}_{os.getpid()}"),
    ]
    in_use = _wake_directory(shm, f"omni_shm_wake_{os.getuid()}_{'0' * 24}")
    other_names = [shm / "omni_shm_other", _wake_directory(shm, "unrelated")]
    other_names[0].touch()
    reader = os.open(in_use / "0_fifo", os.O_RDONLY | os.O_NONBLOCK)

    try:
        _start_and_stop(tmp_path)
    finally:
        os.close(reader)

    assert not any(leftover.exists() for leftover in leftovers)
    assert in_use.exists()
    assert all(path.exists() for path in other_names)


def test_leftovers_of_another_user_stay(tmp_path: Path) -> None:
    shm = tmp_path / "shm"
    foreign = _wake_directory(shm, f"omni_shm_wake_{os.getuid()}_{os.getpid()}")

    _start_and_stop(tmp_path, user_id=os.getuid() + 1)

    assert foreign.exists()


def test_a_symbolic_link_among_the_leftovers_is_neither_followed_nor_removed(
    tmp_path: Path,
) -> None:
    shm = tmp_path / "shm"
    target = _wake_directory(shm, "unrelated")
    link = shm / "omni_shm_link"
    link.symlink_to(target)

    _start_and_stop(tmp_path)

    assert link.is_symlink()
    assert target.exists()


def _run_harness(
    role: str, tmp_path: Path, **streams: int | None
) -> subprocess.Popen[str]:
    return subprocess.Popen(  # noqa: S603 - the harness is a file of these tests
        [sys.executable, str(_HARNESS), role, str(tmp_path)],
        text=True,
        **streams,
    )


def test_the_server_dies_with_its_worker(tmp_path: Path) -> None:
    (tmp_path / "shm").mkdir()
    with _run_harness("idle", tmp_path, stdout=subprocess.PIPE) as worker:
        server_pid = int(worker.stdout.readline())

        worker.kill()
        worker.wait()

        assert _has_ended(server_pid)


def test_the_worker_ends_when_its_server_dies_after_the_start(
    tmp_path: Path,
) -> None:
    (tmp_path / "shm").mkdir()
    with _run_harness("idle", tmp_path, stdout=subprocess.PIPE) as worker:
        try:
            os.kill(int(worker.stdout.readline()), signal.SIGKILL)

            assert worker.wait(timeout=10) != 0
        finally:
            worker.kill()


@pytest.fixture
def parent_death_signal_cleared() -> Iterator[None]:
    yield
    ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, 0)


@pytest.mark.usefixtures("parent_death_signal_cleared")
def test_the_signal_is_set_while_the_creator_lives() -> None:
    install_parent_death_signal(os.getppid())

    signal_number = ctypes.c_int()
    ctypes.CDLL(None).prctl(PR_GET_PDEATHSIG, ctypes.byref(signal_number))
    assert signal_number.value == signal.SIGKILL


@pytest.mark.usefixtures("parent_death_signal_cleared")
def test_a_creator_that_is_already_gone_fails_the_signal() -> None:
    with pytest.raises(RuntimeError):
        install_parent_death_signal(os.getppid() + 1)


def test_the_servers_output_reaches_the_service_only_before_the_worker_is_ready(
    tmp_path: Path,
) -> None:
    (tmp_path / "shm").mkdir()
    with (
        (tmp_path / "stdout").open("w") as stdout,
        (tmp_path / "stderr").open("w") as stderr,
    ):
        worker = _run_harness("discard", tmp_path, stdout=stdout, stderr=stderr)
        worker.wait()

    assert worker.returncode == 0
    assert "before-ready" in (tmp_path / "stderr").read_text()
    assert "after-ready" not in (tmp_path / "stderr").read_text()
    assert (tmp_path / "stdout").read_text() == ""
