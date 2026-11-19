"""The launcher the service starts: it becomes `vllm-omni serve` on the given socket."""

import argparse
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from speech_qwen.parent_death import install_parent_death_signal
from speech_qwen.server import remove_leftover_shared_memory


def main(
    argv: Sequence[str] | None = None,
    *,
    tie_to_parent: Callable[[int], None] = install_parent_death_signal,
    remove_leftovers: Callable[[], None] = remove_leftover_shared_memory,
    server_command: Callable[[Path, Path], list[str]] | None = None,
    execute: Callable[[str, list[str]], object] = os.execv,
) -> None:
    """Tie this process to the service, then replace it with the server.

    The parent-death signal survives the exec, so the server dies with the
    thread of the service that started this launcher. The keywords are the
    boundaries: the signal, the shared-memory clean-up, the closed command of
    the pinned server (imported only when used) and the exec.
    """
    arguments = _arguments(argv)
    try:
        tie_to_parent(arguments.expected_parent_pid)
    except RuntimeError:
        raise SystemExit(1) from None
    remove_leftovers()
    if server_command is None:
        from speech_qwen.model import pinned_server_command as server_command

    command = server_command(arguments.cache, arguments.socket)
    execute(command[0], command)


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-parent-pid", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
