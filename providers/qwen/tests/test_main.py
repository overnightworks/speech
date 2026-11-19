"""The launcher ties the server to the service and becomes it."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from speech_qwen import __main__ as launcher
from support.process_state import has_ended

if TYPE_CHECKING:
    from pathlib import Path


def _arguments(tmp_path: Path, parent: int) -> list[str]:
    return [
        "--expected-parent-pid",
        str(parent),
        "--cache",
        str(tmp_path / "cache"),
        "--socket",
        str(tmp_path / "server.sock"),
    ]


def test_the_launcher_ties_itself_to_the_service_then_cleans_up_and_becomes_the_server(
    tmp_path: Path,
) -> None:
    steps: list[object] = []
    command = ["/closed/vllm-omni", "serve", "checkpoint"]

    def served(cache: Path, socket_path: Path) -> list[str]:
        steps.append(("command", cache, socket_path))
        return command

    launcher.main(
        _arguments(tmp_path, parent=4321),
        tie_to_parent=lambda pid: steps.append(("tie", pid)),
        remove_leftovers=lambda: steps.append("clean"),
        server_command=served,
        execute=lambda path, argv: steps.append(("exec", path, argv)),
    )

    assert steps == [
        ("tie", 4321),
        "clean",
        ("command", tmp_path / "cache", tmp_path / "server.sock"),
        ("exec", command[0], command),
    ]


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
            execute=lambda *_: started.append("exec"),
        )

    assert ended.value.code == 1
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

launcher.main(
    ["--expected-parent-pid", sys.argv[1], "--cache", "unused", "--socket", "unused"],
    remove_leftovers=lambda: None,
    server_command=lambda cache, socket: [
        sys.executable, "-c", "import os, time\\nwhile True: time.sleep(1)"
    ],
)
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
