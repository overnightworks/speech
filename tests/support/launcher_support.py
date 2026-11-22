"""A fake `speech-qwen-server` and the readers of what it observed."""

import ast
import json
import os
import subprocess
import sys
import time
from enum import StrEnum
from pathlib import Path

import pytest

from speech_service.provider_process import ProviderId, provider_entrypoint

SERVER_WAIT_SECONDS = 10.0
_FAKE_LAUNCHER = Path(__file__).with_name("fake_launcher.py")


def provider_launcher(root: Path) -> Path:
    """Return the launcher path the service resolves under a provider root."""
    entrypoint = provider_entrypoint(root, ProviderId.QWEN)
    assert entrypoint is not None
    return entrypoint


def closed_provider(tmp_path: Path, behavior: str = "normal") -> tuple[Path, Path]:
    """Write the fake launcher under a provider root; return that root and its cache."""
    root = tmp_path / "providers"
    entrypoint = provider_launcher(root)
    entrypoint.parent.mkdir(parents=True, exist_ok=True)
    entrypoint.write_text(
        f"#!{sys.executable}\n{_FAKE_LAUNCHER.read_text(encoding='utf-8')}",
        encoding="utf-8",
    )
    entrypoint.chmod(0o700)
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "behavior").write_text(behavior, encoding="utf-8")
    return root, cache


class LoadOutcome(StrEnum):
    """What a `held-ready` server does once its load is released."""

    READY = "ready"
    FAILED = "failed"
    """It never answers and lives on, until the health deadline ends the load."""
    EXIT = "exit"


def server_starts(cache: Path) -> list[int]:
    """The pids of every server started over `cache`, in start order."""
    log = cache / "server-starts"
    if not log.exists():
        return []
    return [int(line) for line in log.read_text(encoding="utf-8").splitlines()]


def received_requests(cache: Path) -> list[dict[str, object]]:
    """The body of every speech request the servers over `cache` received."""
    log = cache / "requests"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def wait_for_starts(cache: Path, count: int) -> list[int]:
    """The start log once it holds `count` starts; fails after a bounded wait."""
    deadline = time.monotonic() + SERVER_WAIT_SECONDS
    while len(started := server_starts(cache)) < count:
        assert time.monotonic() < deadline, f"{count} server starts never came"
        time.sleep(0.01)
    return started


def release_load(cache: Path, pid: int, outcome: LoadOutcome) -> None:
    """Let the `held-ready` server `pid` end its load with `outcome`."""
    _write_atomically(cache / f"release-{pid}", outcome.value)


def wait_for_held_speech(cache: Path) -> None:
    """Return once a `hold-first-speech` server holds its first request."""
    deadline = time.monotonic() + SERVER_WAIT_SECONDS
    while not (cache / "speech-held").exists():
        assert time.monotonic() < deadline, "no speech request was held"
        time.sleep(0.01)


def release_speech(cache: Path) -> None:
    """Let a `hold-first-speech` server answer the request it holds."""
    _write_atomically(cache / "release-speech", "")


def hold_exits(cache: Path) -> None:
    """Make every server started from now on wait on SIGTERM for its exit release."""
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "hold-exit").touch()


def wait_for_held_exit(cache: Path, pid: int) -> None:
    """Return once the held server `pid` got its SIGTERM; fails after a bounded wait."""
    deadline = time.monotonic() + SERVER_WAIT_SECONDS
    while not (cache / f"exit-held-{pid}").exists():
        assert time.monotonic() < deadline, f"server {pid} was never stopped"
        time.sleep(0.01)


def release_exit(cache: Path, pid: int) -> None:
    """Let the held server `pid` exit on the SIGTERM it waits on."""
    _write_atomically(cache / f"release-exit-{pid}", "")


def _write_atomically(path: Path, text: str) -> None:
    staged = path.with_name(f"{path.name}.staged")
    staged.write_text(text, encoding="utf-8")
    staged.replace(path)


def observed(cache: Path) -> tuple[str, dict[str, str], int, dict[str, str]]:
    lines = (cache / "server-observed").read_text(encoding="utf-8").splitlines()
    return (
        ast.literal_eval(lines[0]),
        ast.literal_eval(lines[1]),
        ast.literal_eval(lines[2]),
        ast.literal_eval(lines[3]),
    )


def numba_cache(cache: Path) -> tuple[Path, int]:
    """The server's numba cache directory and the permission bits it found there."""
    _, environment, _, _ = observed(cache)
    mode = (cache / "numba-cache-mode").read_text(encoding="utf-8")
    return Path(environment["NUMBA_CACHE_DIR"]), int(mode)


def debug_log(cache: Path) -> Path:
    """The log the server left in its default temporary directory, as libraries do."""
    return Path((cache / "server-debug-log").read_text(encoding="utf-8"))


def assert_reaped(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def is_running(pid: int) -> bool:
    """Whether the process `pid` still exists, not yet reaped."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def fail_every_reap(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Make each server started from now on fail its cleanup once it is reaped.

    The returned list fills with the pids of those servers, in start order.
    """
    real_popen = subprocess.Popen
    started: list[int] = []

    class ReapedThenFailing:
        def __init__(self, process: subprocess.Popen[bytes]) -> None:
            self._process = process

        def __getattr__(self, name: str) -> object:
            return getattr(self._process, name)

        def wait(self, timeout: float | None = None) -> int:
            self._process.wait(timeout=timeout)
            message = "cleanup failed after reap"
            raise RuntimeError(message)

    def popen(*arguments: object, **keywords: object) -> ReapedThenFailing:
        process = real_popen(*arguments, **keywords)
        started.append(process.pid)
        return ReapedThenFailing(process)

    monkeypatch.setattr("speech_service.provider_process.subprocess.Popen", popen)
    return started
