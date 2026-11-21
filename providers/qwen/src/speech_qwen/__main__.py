"""The launcher the service starts: it runs `vllm-omni serve` and reaps its group.

The service starts the launcher as the leader of a process group of its own, so
the server, its stages and the `multiprocessing` resource tracker all live in
that group. The server's stages arm their own parent-death signal late, so a
server that dies while they spawn leaves them behind. The launcher therefore
stays alive: its parent-death signal kills the whole group, and the service's
orderly group stop still reaches the server while the launcher waits for it.
The server carries no death signal of its own, because the launcher is its only
parent and the launcher's death ends the group either way: through its handler
when the service dies, through the service's group kill when the launcher dies
alone.
"""

import argparse
import os
import signal
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from types import FrameType

from speech_qwen.parent_death import install_parent_death_signal
from speech_qwen.server import remove_leftover_shared_memory


def main(
    argv: Sequence[str] | None = None,
    *,
    tie_to_parent: Callable[[int], None] | None = None,
    remove_leftovers: Callable[[], None] = remove_leftover_shared_memory,
    server_command: Callable[[Path, Path], list[str]] | None = None,
) -> None:
    """Tie the group to the service, run the server, and end as the server ends.

    The keywords are the boundaries: the parent-death handling, the shared-memory
    clean-up and the closed command of the pinned server (imported only when used).
    """
    arguments = _arguments(argv)
    try:
        (tie_to_parent or _tie_group_to_parent)(arguments.expected_parent_pid)
    except RuntimeError:
        raise SystemExit(1) from None
    remove_leftovers()
    if server_command is None:
        from speech_qwen.model import pinned_server_command as server_command

    command = server_command(arguments.cache, arguments.socket)
    # The scanner does not accept a guard on the two trusted-service paths.
    with subprocess.Popen(command) as server:  # NOSONAR(S8705)  # noqa: S603 - the command is the closed pinned server
        _end_as(server.wait())


def _tie_group_to_parent(expected_parent_pid: int) -> None:
    """Kill the whole process group when the service dies.

    The handler is installed before the signal is armed, so a service that dies
    in between is still answered. The service's SIGTERM reaches the server
    through the group and is met here by a handler, not by SIG_IGN, because a
    handler resets across the server's exec and SIG_IGN would be inherited.
    """
    if os.getpgrp() != os.getpid():
        raise RuntimeError
    signal.signal(signal.SIGUSR1, _kill_group)
    signal.signal(signal.SIGTERM, _wait_for_the_server)
    install_parent_death_signal(expected_parent_pid, signal.SIGUSR1)


def _kill_group(_signal_number: int, _frame: FrameType | None) -> None:
    os.killpg(os.getpgrp(), signal.SIGKILL)


def _wait_for_the_server(_signal_number: int, _frame: FrameType | None) -> None:
    """Do nothing: the server's own end ends the launcher."""


def _end_as(server_status: int) -> None:
    """End as the server ended: a signal death as that signal, an exit as itself."""
    if server_status < 0:
        killing_signal = -server_status  # an int: real-time signals have no enum member
        if killing_signal != signal.SIGKILL:  # its disposition cannot be set
            signal.signal(killing_signal, signal.SIG_DFL)
        os.kill(os.getpid(), killing_signal)
    raise SystemExit(server_status)


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-parent-pid", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
