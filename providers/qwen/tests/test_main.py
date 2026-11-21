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

_HANDLED_SIGNALS = (signal.SIGUSR1, signal.SIGUSR2, signal.SIGTERM)

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
def restored_signal_handlers() -> Iterator[None]:
    saved = {number: signal.getsignal(number) for number in _HANDLED_SIGNALS}
    yield
    for number, handler in saved.items():
        signal.signal(number, handler)


def _run_launcher(tmp_path: Path, *server: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(  # noqa: S603 - the launcher is the program of this test
        [sys.executable, "-c", _LAUNCHER, str(os.getpid()), *server],
        cwd=tmp_path,
        start_new_session=True,
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

    with pytest.raises(SystemExit) as ended:
        launcher.main(
            _arguments(tmp_path, parent=1),
            tie_to_parent=lambda _pid: None,
            remove_leftovers=lambda: None,
        )

    assert ended.value.code == _SERVER_STATUS_OF_THE_PINNED_COMMAND


def test_a_launcher_whose_service_is_already_gone_ends_without_starting_a_server(
    tmp_path: Path,
) -> None:
    started: list[str] = []

    def gone(_pid: int) -> None:
        raise RuntimeError

    arguments = _arguments(tmp_path, parent=1)

    with pytest.raises(SystemExit) as ended:
        launcher.main(
            arguments,
            tie_to_parent=gone,
            remove_leftovers=lambda: started.append("clean"),
            server_command=lambda *_: started.append("command") or [],
        )

    assert ended.value.code == 1
    assert started == []


def test_a_launcher_that_does_not_lead_its_group_ends_without_starting_a_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[str] = []
    monkeypatch.setattr(os, "getpgrp", lambda: os.getpid() + 1)

    with pytest.raises(SystemExit) as ended:
        launcher.main(
            _arguments(tmp_path, parent=os.getppid()),
            remove_leftovers=lambda: started.append("clean"),
            server_command=lambda *_: started.append("command") or [],
        )

    assert ended.value.code == 1
    assert started == []


@pytest.mark.usefixtures("restored_signal_handlers")
def test_a_tied_launcher_ends_its_group_on_the_parent_death_signal_not_on_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    armed: list[tuple[int, signal.Signals, object]] = []
    group_kills: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "getpgrp", os.getpid)
    monkeypatch.setattr(os, "killpg", lambda *kill: group_kills.append(kill))
    monkeypatch.setattr(
        launcher,
        "install_parent_death_signal",
        lambda pid, number: armed.append((pid, number, signal.getsignal(number))),
    )

    with pytest.raises(SystemExit):
        launcher.main(
            _arguments(tmp_path, parent=4321),
            remove_leftovers=lambda: None,
            server_command=lambda *_: [sys.executable, "-c", "pass"],
        )

    (pid, number, handler_at_arming) = armed[0]
    assert (pid, number) == (4321, signal.SIGUSR1)
    assert callable(handler_at_arming)
    handler_at_arming(number, None)
    assert group_kills == [(os.getpid(), signal.SIGKILL)]
    sigterm_handler = signal.getsignal(signal.SIGTERM)
    assert callable(sigterm_handler)
    sigterm_handler(signal.SIGTERM, None)
    assert group_kills == [(os.getpid(), signal.SIGKILL)]


@pytest.mark.usefixtures("restored_signal_handlers")
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
    raised: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda *kill: raised.append(kill))
    signal.signal(signal.SIGUSR2, lambda *_: None)

    with pytest.raises(SystemExit):
        launcher.main(
            _arguments(tmp_path, parent=1),
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
    tmp_path: Path, server_program: str, launcher_status: int
) -> None:
    ended = _run_launcher(tmp_path, sys.executable, "-c", server_program)

    assert ended.returncode == launcher_status


def test_the_launcher_survives_the_services_group_sigterm_and_waits_for_the_server(
    tmp_path: Path,
) -> None:
    ready_file = tmp_path / "server.ready"
    server = (
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
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
        cwd=tmp_path,
        start_new_session=True,
    ) as launcher_process:
        try:
            await_pid_file(ready_file)
            os.killpg(launcher_process.pid, signal.SIGTERM)

            assert launcher_process.wait(timeout=60) == _SERVER_STATUS_AFTER_SIGTERM
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
