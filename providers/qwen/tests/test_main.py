"""The launcher ties the server to the service and becomes it."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

from speech_qwen import __main__ as launcher
from support.process_state import has_ended

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parent: int) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "speech-qwen-server",
            "--expected-parent-pid",
            str(parent),
            "--cache",
            str(tmp_path / "cache"),
            "--socket",
            str(tmp_path / "server.sock"),
        ],
    )


def test_the_launcher_ties_itself_to_the_service_then_cleans_up_and_becomes_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps: list[object] = []
    command = ["/closed/vllm-omni", "serve", "checkpoint"]
    _arguments(monkeypatch, tmp_path, parent=4321)
    monkeypatch.setattr(
        launcher, "install_parent_death_signal", lambda pid: steps.append(("tie", pid))
    )
    monkeypatch.setattr(
        launcher, "remove_leftover_shared_memory", lambda: steps.append("clean")
    )

    def served(cache: Path, socket_path: Path) -> list[str]:
        steps.append(("command", cache, socket_path))
        return command

    monkeypatch.setattr("speech_qwen.model.pinned_server_command", served)
    monkeypatch.setattr(
        os, "execv", lambda path, argv: steps.append(("exec", path, argv))
    )

    launcher.main()

    assert steps == [
        ("tie", 4321),
        "clean",
        ("command", tmp_path / "cache", tmp_path / "server.sock"),
        ("exec", command[0], command),
    ]


def test_a_launcher_whose_service_is_already_gone_ends_without_starting_a_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[str] = []

    def gone(_pid: int) -> None:
        raise RuntimeError

    _arguments(monkeypatch, tmp_path, parent=1)
    monkeypatch.setattr(launcher, "install_parent_death_signal", gone)
    monkeypatch.setattr(
        launcher, "remove_leftover_shared_memory", lambda: started.append("clean")
    )
    monkeypatch.setattr(os, "execv", lambda *_: started.append("exec"))

    assert launcher.main() == 1
    assert started == []


_SERVICE = r"""
import os
import subprocess
import sys
import threading

LAUNCHER = '''
import os
import sys
from speech_qwen import __main__ as launcher
import speech_qwen.model

launcher.remove_leftover_shared_memory = lambda: None
speech_qwen.model.pinned_server_command = lambda cache, socket: [
    sys.executable, "-c", "import os, time\\nwhile True: time.sleep(1)"
]
sys.argv = [
    "speech-qwen-server", "--expected-parent-pid", sys.argv[1],
    "--cache", "unused", "--socket", "unused",
]
launcher.main()
'''

launcher = subprocess.Popen([sys.executable, "-c", LAUNCHER, str(os.getpid())])
print(launcher.pid, flush=True)
threading.Event().wait()
"""


def test_the_server_dies_with_the_service_that_started_its_launcher() -> None:
    with subprocess.Popen(  # noqa: S603 - the service is the program of this test
        [sys.executable, "-c", _SERVICE], stdout=subprocess.PIPE, text=True
    ) as service:
        server_pid = int(service.stdout.readline())
        assert not has_ended(server_pid, within=0.5)

        service.kill()
        service.wait()

        assert has_ended(server_pid)
