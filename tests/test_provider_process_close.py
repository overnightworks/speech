"""A provider process that is slow to stop, to clean up, to drain or to answer."""

from __future__ import annotations

import logging
import os
import signal
import socket
import tempfile
import threading
from typing import TYPE_CHECKING

import pytest

from speech_service.provider_process import (
    CLOSE_SECONDS,
    ProviderCleanupFailedError,
    _is_healthy,
)
from tests.support.launcher_support import (
    LoadOutcome,
    assert_reaped,
    hold_exits,
    provider_server,
    release_exit,
    release_load,
    server_starts,
    stage_stderr_outcome,
    wait_for_held_exit,
    wait_for_starts,
)
from tests.support.process_state import has_ended

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from speech_service.provider_process import ProviderProcess

_OWNER = "speech_service.provider_process"
_SMALL_SECONDS = 0.05


def test_an_owner_that_has_not_stopped_within_the_close_bound_is_fatal_and_close_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    server = provider_server(tmp_path)
    hold_exits(cache)
    fatal = threading.Event()
    server.set_fatal_callback(fatal.set)
    server.load()
    (pid,) = server_starts(cache)

    monkeypatch.setattr(f"{_OWNER}.CLOSE_SECONDS", _SMALL_SECONDS)
    try:
        with pytest.raises(ProviderCleanupFailedError):
            server.close()
        assert fatal.is_set()
    finally:
        wait_for_held_exit(cache, pid)
        release_exit(cache, pid)
        monkeypatch.setattr(f"{_OWNER}.CLOSE_SECONDS", CLOSE_SECONDS)
        server.close()
    assert_reaped(pid)


def test_a_second_load_while_the_first_waits_starts_no_second_server(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    server = provider_server(tmp_path, "held-ready")
    loads = [threading.Thread(target=server.load) for _ in range(2)]
    loads[0].start()
    (pid,) = wait_for_starts(cache, 1)
    loads[1].start()

    release_load(cache, pid, LoadOutcome.READY)
    for load in loads:
        load.join()

    assert server.ready
    assert server_starts(cache) == [pid]
    server.close()
    assert_reaped(pid)


def test_a_failed_scratch_cleanup_is_logged_and_the_release_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class FailingCleanup(tempfile.TemporaryDirectory[str]):
        def cleanup(self) -> None:
            super().cleanup()
            message = "scratch cleanup failed"
            raise OSError(message)

    monkeypatch.setattr(
        f"{_OWNER}.tempfile",
        type("Tempfile", (), {"TemporaryDirectory": FailingCleanup}),
    )
    server = provider_server(tmp_path)
    server.load()
    (pid,) = server_starts(tmp_path / "cache")

    with caplog.at_level(logging.ERROR):
        server.close()

    assert "provider_scratch_cleanup.failed" in {
        record.message for record in caplog.records
    }
    assert_reaped(pid)


@pytest.fixture
def stage_holding_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ProviderProcess, int]]:
    monkeypatch.setattr(f"{_OWNER}._STDERR_DRAIN_JOIN_SECONDS", _SMALL_SECONDS)
    cache = tmp_path / "cache"
    server = provider_server(tmp_path, "detached-stage")
    server.load()
    (pid,) = server_starts(cache)
    stage = int((cache / "stage").read_text(encoding="utf-8"))
    try:
        yield server, pid
    finally:
        os.kill(stage, signal.SIGKILL)
    assert has_ended(stage)


def test_a_grandchild_that_holds_stderr_does_not_hold_the_release(
    stage_holding_stderr: tuple[ProviderProcess, int],
) -> None:
    server, pid = stage_holding_stderr

    server.close()

    assert_reaped(pid)


def test_the_providers_stderr_is_closed_for_a_grandchild_that_holds_it(
    tmp_path: Path, stage_holding_stderr: tuple[ProviderProcess, int]
) -> None:
    server, _pid = stage_holding_stderr

    server.close()

    assert stage_stderr_outcome(tmp_path / "cache") == "stderr-closed"


@pytest.mark.parametrize("spoiled", ["health-dropped", "health-garbled"])
def test_a_dropped_or_garbled_health_answer_is_not_ready_and_the_poll_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spoiled: str
) -> None:
    monkeypatch.setattr(f"{_OWNER}._HEALTH_POLL_SECONDS", _SMALL_SECONDS)
    server = provider_server(tmp_path, spoiled)

    server.load()

    assert server.ready
    assert (tmp_path / "cache" / "health-spoiled").exists()
    server.close()


def test_a_health_request_that_gets_no_answer_in_time_is_not_healthy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(f"{_OWNER}._HEALTH_REQUEST_SECONDS", _SMALL_SECONDS)
    socket_path = tmp_path / "silent.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as silent:
        silent.bind(str(socket_path))
        silent.listen()

        assert not _is_healthy(socket_path)
