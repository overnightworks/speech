"""A fake provider worker and the readers of what it observed."""

import ast
import os
import subprocess
import sys
import time
from enum import StrEnum
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

WORKER_WAIT_SECONDS = 10.0

_WORKER = r"""#!{python}
import argparse
import os
import signal
import struct
import time
from pathlib import Path

protocol_fd = os.dup(1)
null = os.open(os.devnull, os.O_WRONLY)
os.dup2(null, 1)
os.close(null)
print("provider print noise", flush=True)
os.write(1, b"native provider noise\n")

from speech_runtime_provider_contract import (
    MAX_FRAME_LENGTH, Frame, FrameKind, ProviderFailure, read_frame, write_frame
)

parser = argparse.ArgumentParser()
parser.add_argument("--expected-parent-pid", required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--cache", required=True)
if {provider!r} == "qwen":
    parser.add_argument("--speaker", required=True)
arguments = parser.parse_args()
cache = Path(arguments.cache)
cache.mkdir(parents=True, exist_ok=True)
def wait_for(path):
    while not path.exists():
        time.sleep(0.01)


if (cache / "hold-exit").exists():
    def exit_when_released(_signal, _frame):
        (cache / f"exit-held-{{os.getpid()}}").touch()
        wait_for(cache / f"release-exit-{{os.getpid()}}")
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, exit_when_released)
with (cache / "worker-starts").open("a", encoding="utf-8") as starts:
    starts.write(f"{{os.getpid()}}\n")
(cache / "worker-observed").write_text(
    repr(os.getcwd()) + "\n" + repr(os.environ.copy()) + "\n" +
    repr(os.getpid()) + "\n" + repr(arguments.__dict__), encoding="utf-8"
)

stream = os.fdopen(protocol_fd, "wb", buffering=0)
behavior = {behavior!r}
if behavior == "held-ready":
    release = cache / f"release-{{os.getpid()}}"
    wait_for(release)
    outcome = release.read_text(encoding="utf-8")
    if outcome == "exit":
        raise SystemExit(1)
    if outcome == "failed":
        write_frame(stream, Frame(FrameKind.FAILED, 0, b"\x01"))
        while True:
            time.sleep(1)
if behavior == "startup-exit":
    raise SystemExit(1)
if behavior == "startup-oversized":
    stream.write(struct.pack("!I", MAX_FRAME_LENGTH + 1))
    raise SystemExit(1)
if behavior == "startup-failed":
    write_frame(stream, Frame(FrameKind.FAILED, 0, b"\x01"))
    raise SystemExit(1)
rate = 48000 if behavior == "other-rate" else 24000
write_frame(stream, Frame(FrameKind.READY, 0, bytes((0, 1)) + rate.to_bytes(4, "big")))
if behavior == "idle-exit":
    wait_for(cache / "exit-now")
    raise SystemExit(1)
if behavior == "term-resistant":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
last_terminal = 0
while True:
    frame = read_frame(__import__("sys").stdin.buffer)
    if frame.kind is FrameKind.CANCEL and frame.request_id == last_terminal:
        continue
    if frame.kind is not FrameKind.SYNTHESIZE:
        write_frame(stream, Frame(FrameKind.FAILED, 0, b"\x01"))
        raise SystemExit(1)
    request_id = frame.request_id
    text = frame.payload[1:].decode()
    print("synthesis print noise", flush=True)
    os.write(1, b"synthesis native noise\n")
    with (cache / "request-ids").open("a", encoding="utf-8") as ids:
        ids.write(f"{{request_id}}\n")
    if text == "stale":
        write_frame(stream, Frame(FrameKind.PCM, request_id + 1, b"\x00\x01"))
        continue
    if text == "wrong-direction":
        write_frame(stream, Frame(FrameKind.READY, 0, b"\x00\x01\x00\x00]\xc0"))
        continue
    if text == "unexpected-cancelled":
        write_frame(stream, Frame(FrameKind.CANCELLED, request_id, b""))
        continue
    if text == "failed":
        write_frame(stream, Frame(FrameKind.FAILED, request_id, b"\x01"))
        raise SystemExit(1)
    write_frame(stream, Frame(FrameKind.PCM, request_id, b"\x00\x01"))
    if text == "midstream-exit":
        raise SystemExit(1)
    if text == "wait-cancel":
        cancel = read_frame(__import__("sys").stdin.buffer)
        with (cache / "cancel-observed").open("a", encoding="utf-8") as observed:
            observed.write(f"{{cancel.request_id}}\n")
        write_frame(stream, Frame(FrameKind.CANCELLED, cancel.request_id, b""))
        last_terminal = request_id
        continue
    if behavior == "partial-cancel":
        read_frame(__import__("sys").stdin.buffer)
        (cache / "cancel-observed").touch()
        stream.write(struct.pack("!IBQ", 11, FrameKind.COMPLETE, request_id) + b"x")
        time.sleep(30)
    write_frame(stream, Frame(FrameKind.COMPLETE, request_id, b""))
    if behavior == "duplicate-terminal":
        write_frame(stream, Frame(FrameKind.COMPLETE, request_id, b""))
    last_terminal = request_id
"""


def closed_provider(
    tmp_path: Path, behavior: str = "normal", provider_name: str = "qwen"
) -> tuple[Path, Path]:
    root = tmp_path / "providers"
    provider = root / provider_name
    entrypoint = provider / ".venv" / "bin" / f"speech-runtime-{provider_name}-worker"
    entrypoint.parent.mkdir(parents=True, exist_ok=True)
    entrypoint.write_text(
        _WORKER.format(
            python=sys.executable,
            behavior=behavior,
            provider=provider_name,
        ),
        encoding="utf-8",
    )
    entrypoint.chmod(0o700)
    return root, tmp_path / "cache"


class LoadOutcome(StrEnum):
    """What a `held-ready` worker does once its load is released."""

    READY = "ready"
    FAILED = "failed"
    EXIT = "exit"


def worker_starts(cache: Path) -> list[int]:
    """The pids of every worker started over `cache`, in start order."""
    log = cache / "worker-starts"
    if not log.exists():
        return []
    return [int(line) for line in log.read_text(encoding="utf-8").splitlines()]


def wait_for_starts(cache: Path, count: int) -> list[int]:
    """The start log once it holds `count` starts; fails after a bounded wait."""
    deadline = time.monotonic() + WORKER_WAIT_SECONDS
    while len(started := worker_starts(cache)) < count:
        assert time.monotonic() < deadline, f"{count} worker starts never came"
        time.sleep(0.01)
    return started


def release_load(cache: Path, pid: int, outcome: LoadOutcome) -> None:
    """Let the `held-ready` worker `pid` end its load with `outcome`."""
    _write_atomically(cache / f"release-{pid}", outcome.value)


def hold_exits(cache: Path) -> None:
    """Make every worker started from now on wait on SIGTERM for its exit release."""
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "hold-exit").touch()


def wait_for_held_exit(cache: Path, pid: int) -> None:
    """Return once the held worker `pid` got its SIGTERM; fails after a bounded wait."""
    deadline = time.monotonic() + WORKER_WAIT_SECONDS
    while not (cache / f"exit-held-{pid}").exists():
        assert time.monotonic() < deadline, f"worker {pid} was never stopped"
        time.sleep(0.01)


def release_exit(cache: Path, pid: int) -> None:
    """Let the held worker `pid` exit on the SIGTERM it waits on."""
    _write_atomically(cache / f"release-exit-{pid}", "")


def _write_atomically(path: Path, text: str) -> None:
    staged = path.with_name(f"{path.name}.staged")
    staged.write_text(text, encoding="utf-8")
    staged.replace(path)


def observed(cache: Path) -> tuple[str, dict[str, str], int, dict[str, str]]:
    lines = (cache / "worker-observed").read_text(encoding="utf-8").splitlines()
    return (
        ast.literal_eval(lines[0]),
        ast.literal_eval(lines[1]),
        ast.literal_eval(lines[2]),
        ast.literal_eval(lines[3]),
    )


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
    """Make each worker started from now on fail its cleanup once it is reaped.

    The returned list fills with the pids of those workers, in start order.
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

    monkeypatch.setattr(
        "speech_runtime_service.provider_process.subprocess.Popen", popen
    )
    return started
