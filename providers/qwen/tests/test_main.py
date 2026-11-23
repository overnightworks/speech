"""The launcher runs the server in its own group and ends the group with the service."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from speech_qwen import __main__ as launcher
from support.process_state import await_pid_file, group_has_ended, process_state

if TYPE_CHECKING:
    from collections.abc import Iterator

_FAKE_SERVER = Path(__file__).parent / "support" / "fake_vllm_omni.py"

# Between SIGRTMIN and SIGRTMAX, where `signal.Signals` has no member.
_UNNAMED_REAL_TIME_SIGNAL = 40

_SERVER_STATUS_AFTER_SIGTERM = 7

_SERVER_STATUS_OF_THE_PINNED_COMMAND = 5

_LAUNCHER = """
import sys
from speech_qwen import __main__ as launcher

launcher.main(
    ["--expected-parent-pid", sys.argv[1], "--cache", "unused", "--socket", "unused"],
    remove_leftovers=lambda: None,
    server_command=lambda cache, socket: sys.argv[2:],
)
"""

_SERVICE = f"""
import os
import subprocess
import sys
import threading

launcher = subprocess.Popen(
    [sys.executable, "-c", {_LAUNCHER!r}, str(os.getpid()), *sys.argv[1:]],
    start_new_session=True,
    stdout=subprocess.DEVNULL,
)
print(launcher.pid, flush=True)
threading.Event().wait()
"""


def _arguments(tmp_path: Path, parent: int) -> list[str]:
    return [
        "--expected-parent-pid",
        str(parent),
        "--cache",
        str(tmp_path / "cache"),
        "--socket",
        str(tmp_path / "server.sock"),
    ]


def _killed_by(killing_signal: int) -> str:
    return f"import os; os.kill(os.getpid(), {killing_signal})"


@pytest.fixture
def restored_sigusr2() -> Iterator[None]:
    saved = signal.getsignal(signal.SIGUSR2)
    yield
    signal.signal(signal.SIGUSR2, saved)


def _run_launcher(
    *server: str, leads_its_group: bool = True, stderr: int | None = None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(  # noqa: S603 - the launcher is the program of this test
        [sys.executable, "-c", _LAUNCHER, str(os.getpid()), *server],
        start_new_session=leads_its_group,
        stderr=stderr,
        timeout=60,
        check=False,
    )


def test_the_launcher_ties_itself_to_the_service_then_cleans_up_and_starts_the_server(
    tmp_path: Path,
) -> None:
    steps: list[object] = []
    command = [sys.executable, "-c", "pass"]

    def served(cache: Path, socket_path: Path) -> list[str]:
        steps.append(("command", cache, socket_path))
        return command

    arguments = _arguments(tmp_path, parent=4321)

    with pytest.raises(SystemExit) as ended:
        launcher.main(
            arguments,
            tie_to_parent=lambda pid: steps.append(("tie", pid)),
            remove_leftovers=lambda: steps.append("clean"),
            server_command=served,
        )

    assert ended.value.code == 0
    assert steps == [
        ("tie", 4321),
        "clean",
        ("command", tmp_path / "cache", tmp_path / "server.sock"),
    ]


def test_the_launcher_asks_the_pinned_server_for_its_command_unless_given_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "speech_qwen.model.pinned_server_command",
        lambda _cache, _socket: [
            sys.executable,
            "-c",
            f"raise SystemExit({_SERVER_STATUS_OF_THE_PINNED_COMMAND})",
        ],
    )

    arguments = _arguments(tmp_path, parent=1)

    with pytest.raises(SystemExit) as ended:
        launcher.main(
            arguments,
            tie_to_parent=lambda _pid: None,
            remove_leftovers=lambda: None,
        )

    assert ended.value.code == _SERVER_STATUS_OF_THE_PINNED_COMMAND


def test_a_launcher_that_does_not_lead_its_group_ends_without_starting_a_server(
    tmp_path: Path,
) -> None:
    started = tmp_path / "started"
    server = f"open({str(started)!r}, 'w')"

    ended = _run_launcher(
        sys.executable,
        "-c",
        server,
        leads_its_group=False,
        stderr=subprocess.PIPE,
    )

    assert ended.returncode == 1
    assert b"Traceback" not in ended.stderr
    assert not started.exists()


def test_the_parent_death_signal_kills_the_launchers_whole_group(
    tmp_path: Path,
) -> None:
    # A real group kill ends the launcher before coverage writes, so the line
    # that issues it is proven by this outcome and not seen by coverage.
    ready_file = tmp_path / "server.ready"
    server = (
        "import os, time\n"
        "from pathlib import Path\n"
        f"Path({str(ready_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(3600)\n"
    )
    with subprocess.Popen(  # noqa: S603 - the launcher is the program of this test
        [
            sys.executable,
            "-c",
            _LAUNCHER,
            str(os.getpid()),
            sys.executable,
            "-c",
            server,
        ],
        start_new_session=True,
    ) as launcher_process:
        try:
            await_pid_file(ready_file)
            os.kill(launcher_process.pid, signal.SIGUSR1)

            assert launcher_process.wait(timeout=60) == -signal.SIGKILL
            assert group_has_ended(launcher_process.pid)
        finally:
            with suppress(ProcessLookupError):
                os.killpg(launcher_process.pid, signal.SIGKILL)


@pytest.mark.usefixtures("restored_sigusr2")
@pytest.mark.parametrize(
    ("killing_signal", "disposition_reset"),
    [(signal.SIGUSR2, True), (signal.SIGKILL, False)],
)
def test_the_launcher_raises_its_servers_killing_signal_at_itself(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    killing_signal: signal.Signals,
    disposition_reset: bool,  # noqa: FBT001 - a parametrized case, not an API
) -> None:
    # In-process because a real self-kill ends the launcher before coverage writes.
    raised: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda *kill: raised.append(kill))
    signal.signal(signal.SIGUSR2, lambda *_: None)
    arguments = _arguments(tmp_path, parent=1)

    with pytest.raises(SystemExit):
        launcher.main(
            arguments,
            tie_to_parent=lambda _pid: None,
            remove_leftovers=lambda: None,
            server_command=lambda *_: [
                sys.executable,
                "-c",
                _killed_by(killing_signal),
            ],
        )

    assert raised == [(os.getpid(), killing_signal)]
    assert (signal.getsignal(signal.SIGUSR2) is signal.SIG_DFL) is disposition_reset


@pytest.mark.parametrize(
    ("server_program", "launcher_status"),
    [
        (_killed_by(signal.SIGTERM), -signal.SIGTERM),
        (_killed_by(signal.SIGKILL), -signal.SIGKILL),
        (_killed_by(signal.SIGSEGV), -signal.SIGSEGV),
        (_killed_by(_UNNAMED_REAL_TIME_SIGNAL), -_UNNAMED_REAL_TIME_SIGNAL),
        ("raise SystemExit(0)", 0),
        ("raise SystemExit(3)", 3),
    ],
)
def test_the_launcher_ends_as_its_server_ended(
    server_program: str, launcher_status: int
) -> None:
    ended = _run_launcher(sys.executable, "-c", server_program)

    assert ended.returncode == launcher_status


def test_the_launcher_survives_the_group_sigterm_and_leaves_its_server_the_default(
    tmp_path: Path,
) -> None:
    ready_file = tmp_path / "server.ready"
    inherited_file = tmp_path / "server.inherited"
    server = (
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
        f"Path({str(inherited_file)!r}).write_text(str(signal.getsignal(signal.SIGTERM)))\n"
        "signal.signal(signal.SIGTERM, lambda *_: "
        f"sys.exit({_SERVER_STATUS_AFTER_SIGTERM}))\n"
        f"Path({str(ready_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(3600)\n"
    )
    with subprocess.Popen(  # noqa: S603 - the launcher is the program of this test
        [
            sys.executable,
            "-c",
            _LAUNCHER,
            str(os.getpid()),
            sys.executable,
            "-c",
            server,
        ],
        start_new_session=True,
    ) as launcher_process:
        try:
            await_pid_file(ready_file)
            os.killpg(launcher_process.pid, signal.SIGTERM)

            assert launcher_process.wait(timeout=60) == _SERVER_STATUS_AFTER_SIGTERM
            assert inherited_file.read_text() == str(signal.SIG_DFL)
        finally:
            with suppress(ProcessLookupError):
                os.killpg(launcher_process.pid, signal.SIGKILL)


def test_the_whole_group_ends_with_the_service_even_before_a_stage_arms_its_own_signal(
    tmp_path: Path,
) -> None:
    stage_pid_file = tmp_path / "stage.pid"
    stage_gate_file = tmp_path / "stage.gate"
    server = [
        sys.executable,
        str(_FAKE_SERVER),
        "stages",
        "--uds",
        str(tmp_path / "server.sock"),
        "--stage-pid-file",
        str(stage_pid_file),
        "--stage-gate-file",
        str(stage_gate_file),
    ]
    group = None
    with subprocess.Popen(  # noqa: S603 - the service is the program of this test
        [sys.executable, "-c", _SERVICE, *server],
        stdout=subprocess.PIPE,
        text=True,
    ) as service:
        try:
            group = int(service.stdout.readline())
            stage = await_pid_file(stage_pid_file)
            state = process_state(stage)
            assert state is not None
            assert "zombie" not in state
            assert not stage_gate_file.exists()

            service.kill()
            service.wait(timeout=60)

            assert group_has_ended(group)
        finally:
            service.kill()
            if group is not None:
                with suppress(ProcessLookupError):
                    os.killpg(group, signal.SIGKILL)
