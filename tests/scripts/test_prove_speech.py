"""The proof's log check rejects a service log that holds spoken text."""

from __future__ import annotations

import functools
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from speech_runtime_client.outcomes import Unreachable
from speech_runtime_client.wire import Health
from tests.support.clock_support import FakeClock
from tests.support.script_support import script_module

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from speech_runtime_client.health import HealthOutcome

PROVE = script_module("prove_speech")
PIPE_BYTES = 65536


@pytest.mark.parametrize(
    ("log", "texts", "leaked"),
    [
        ("INFO started\nINFO text: Hallo Welt", ["Hallo Welt"], ["Hallo Welt"]),
        ("INFO started\nINFO speaking done", ["Hallo Welt", "Die Kamera"], []),
        ("INFO started", ["", "  "], []),
    ],
)
def test_the_log_check_reports_only_the_texts_the_log_holds(
    log: str, texts: list[str], leaked: list[str]
) -> None:
    assert PROVE.spoken_in_log(log, texts) == leaked


FLEET_BUDGET_SECONDS = 1830


@pytest.fixture
def restored_terminate_handler() -> Iterator[None]:
    """Put the TERM handler back after a test that lets the proof install its own."""
    previous = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, previous)


@pytest.fixture
def terminate_handler() -> Iterator[None]:
    """Install the proof's TERM handler for one test and put the old one back."""
    previous = signal.getsignal(signal.SIGTERM)
    PROVE.exit_on_terminate()
    yield
    signal.signal(signal.SIGTERM, previous)


def run_returning(
    status: int, *, takes_lock: bool = False
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    """A fake for the re-exec: the flock command ends with this status."""

    def run(
        command: list[str], env: dict[str, str], *, check: bool
    ) -> subprocess.CompletedProcess[bytes]:
        assert not check
        if takes_lock:
            PROVE.announce_lock_held(env)
        return subprocess.CompletedProcess(command, status)

    return run


def test_the_proof_runs_itself_inside_the_fleet_lock_form() -> None:
    command = PROVE.fleet_command(["--help"])

    assert command[:5] == [
        "flock",
        "-o",
        "-w",
        str(PROVE.LOCK_WAIT_SECONDS),
        str(PROVE.LOCK_PATH),
    ]
    assert command[5:8] == ["setpriv", "--pdeathsig", "TERM"]
    assert command[8:12] == [
        "timeout",
        "-k",
        str(PROVE.KILL_GRACE_SECONDS),
        str(PROVE.CAP_SECONDS),
    ]
    assert command[12] == sys.executable
    assert Path(command[13]) == Path(PROVE.__file__)
    assert command[14:] == ["--help"]


def test_another_script_runs_under_the_same_fleet_form() -> None:
    other = Path("/scripts/other_proof.py")

    command = PROVE.fleet_command(["--device", "cpu"], other)

    assert Path(command[13]) == other
    assert command[14:] == ["--device", "cpu"]


def test_the_cap_and_kill_grace_fit_the_fleet_budget() -> None:
    assert PROVE.CAP_SECONDS + PROVE.KILL_GRACE_SECONDS <= FLEET_BUDGET_SECONDS


def test_a_run_outside_the_lock_hands_over_to_the_fleet_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PROVE.LOCK_HELD_ENV, raising=False)
    started: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def run(
        command: list[str], env: dict[str, str], *, check: bool
    ) -> subprocess.CompletedProcess[bytes]:
        assert not check
        started.append(command)
        environments.append(env)
        return subprocess.CompletedProcess(command, 0)

    other = Path("/scripts/other_proof.py")

    assert PROVE.run_under_lock(["--help"], run, other) == 0
    assert started == [PROVE.fleet_command(["--help"], other)]
    assert PROVE.LOCK_HELD_ENV in environments[0]


def test_a_busy_lock_fails_the_run_and_says_why() -> None:
    busy_lock = run_returning(PROVE.FLOCK_TIMED_OUT_STATUS)

    with pytest.raises(SystemExit) as failure:
        PROVE.run_under_lock([], busy_lock)

    assert "busy" in str(failure.value)
    assert str(PROVE.LOCK_WAIT_SECONDS) in str(failure.value)


def test_a_missing_setpriv_fails_the_run_naming_it() -> None:
    cannot_execute = run_returning(PROVE.FLOCK_COMMAND_UNAVAILABLE_STATUS)

    with pytest.raises(SystemExit) as failure:
        PROVE.run_under_lock([], cannot_execute)

    assert "setpriv" in str(failure.value)


def test_a_failed_proof_under_the_lock_keeps_its_own_exit_status() -> None:
    status = PROVE.FLOCK_TIMED_OUT_STATUS

    assert PROVE.run_under_lock([], run_returning(status, takes_lock=True)) == status


def speaking_health(*, ready: bool) -> Health:
    return Health.model_validate(
        {
            "speaking": {"ready": ready, "voices": {}},
            "hearing": {"model": "none", "ready": False},
            "card_memory_mb": 7,
        }
    )


def test_the_default_wait_ends_at_once_when_the_server_exited(tmp_path: Path) -> None:
    def never_waits(_: float) -> None:
        pytest.fail("the wait went on after the server exited")

    server = Mock(stdout=None, returncode=3)
    server.poll.return_value = 3
    running = PROVE.speech_server(
        "http://127.0.0.1:1",
        {},
        tmp_path,
        start=lambda *_, **__: server,
        wait_ready=functools.partial(
            PROVE.wait_for_speaking_ready,
            fetch=lambda *_: speaking_health(ready=False),
            sleep=never_waits,
        ),
    )

    with pytest.raises(SystemExit, match="exited with status 3"), running:
        pass


@pytest.mark.usefixtures("terminate_handler")
def test_term_stops_the_speech_server_through_its_teardown(tmp_path: Path) -> None:
    server = Mock(stdout=None, returncode=0)
    running = PROVE.speech_server(
        "http://127.0.0.1:1",
        {},
        tmp_path,
        start=lambda *_, **__: server,
        wait_ready=lambda *_: speaking_health(ready=True),
    )

    def term_while_running() -> None:
        with running:
            os.kill(os.getpid(), signal.SIGTERM)

    with pytest.raises(SystemExit):
        term_while_running()

    server.terminate.assert_called_once_with()


@pytest.mark.usefixtures("terminate_handler")
def test_a_load_that_never_falls_ends_at_term() -> None:
    with pytest.raises(SystemExit):
        PROVE.wait_for_load(
            read_load=lambda: PROVE.LOAD_CEILING + 1,
            sleep=lambda _: os.kill(os.getpid(), signal.SIGTERM),
        )


@pytest.mark.usefixtures("restored_terminate_handler")
def test_term_during_the_proof_under_the_lock_runs_its_teardown(
    tmp_path: Path,
) -> None:
    lock_directory = tmp_path / f"{PROVE.LOCK_DIRECTORY_PREFIX}test"
    lock_directory.mkdir()
    token = lock_directory / "acquired"
    torn_down: list[bool] = []

    def proof_body() -> None:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            torn_down.append(True)

    with pytest.raises(SystemExit):
        PROVE.prove_under_lock({PROVE.LOCK_HELD_ENV: str(token)}, proof_body)

    assert torn_down == [True]
    assert token.exists()


def test_a_lock_marker_outside_a_lock_directory_is_refused(tmp_path: Path) -> None:
    marker = tmp_path / "1"

    with pytest.raises(SystemExit):
        PROVE.announce_lock_held({PROVE.LOCK_HELD_ENV: str(marker)})

    assert not marker.exists()


def test_a_service_ends_when_the_thread_that_started_it_ends() -> None:
    sleeps_until_told = "import time; print('up', flush=True); time.sleep(600)"
    started: list[subprocess.Popen[str]] = []

    def start_and_wait_until_up() -> None:
        child = PROVE.start_with_parent_death(
            [sys.executable, "-c", sleeps_until_told],
            stdout=subprocess.PIPE,
            text=True,
        )
        started.append(child)
        child.stdout.readline()

    thread = threading.Thread(target=start_and_wait_until_up)
    thread.start()
    thread.join()
    child = started[0]
    try:
        assert child.wait(timeout=30) == -signal.SIGTERM
    finally:
        child.kill()
        child.wait()
        child.stdout.close()


FAKE_SERVICE = """
import os, sys
for number in range(LINES):
    print("log line", number, "x" * 100, flush=True)
print("last line", flush=True)
with open(os.environ["SERVICE_LOGGED_FIFO"], "w") as logged:
    logged.write("done")
import threading
threading.Event().wait()
"""


def test_the_default_start_never_blocks_a_service_that_logs_past_the_pipe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = tmp_path / "packages" / "speech_runtime_service"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(
        FAKE_SERVICE.replace("LINES", str(PIPE_BYTES // 100 * 2))
    )
    logged = tmp_path / "logged"
    os.mkfifo(logged)
    env = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": str(tmp_path / "packages"),
        "SERVICE_LOGGED_FIFO": str(logged),
    }

    def wait_until_logged(*_: object) -> dict[str, object]:
        assert logged.read_text() == "done"
        return {}

    with PROVE.speech_server(
        "http://127.0.0.1:1", env, tmp_path, wait_ready=wait_until_logged
    ):
        pass

    output = capsys.readouterr().out
    assert "last line" in output
    assert "server_exit" in output


@pytest.mark.parametrize(
    ("health", "detail_in_message"),
    [
        (speaking_health(ready=False), True),
        (None, False),
    ],
    ids=["the last health is named", "no health ever answered"],
)
def test_a_wait_that_runs_out_names_the_last_health_it_saw(
    health: Health | None,
    *,
    detail_in_message: bool,
) -> None:
    clock = FakeClock()

    def answer(*_: object) -> HealthOutcome:
        return Unreachable() if health is None else health

    with pytest.raises(SystemExit, match="not ready within") as failure:
        PROVE.wait_for_speaking_ready(
            "http://127.0.0.1:1",
            lambda: None,
            fetch=answer,
            clock=clock.now,
            sleep=clock.jump,
        )

    assert ("card_memory_mb" in str(failure.value)) is detail_in_message


def test_a_child_whose_parent_is_not_its_starter_never_runs(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "ran"

    child = PROVE.start_with_parent_death(
        [sys.executable, "-c", f"open({str(marker)!r}, 'w')"],
        parent_pid=1,
    )

    assert child.wait(timeout=30) == 1
    assert not marker.exists()
