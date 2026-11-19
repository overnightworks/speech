"""The launcher the service starts: it becomes `vllm-omni serve` on the given socket."""

import argparse
import os
from pathlib import Path

from speech_qwen.parent_death import install_parent_death_signal
from speech_qwen.server import remove_leftover_shared_memory


def main() -> int:
    """Tie this process to the service, then replace it with the server.

    The parent-death signal survives the exec, so the server dies with the
    thread of the service that started this launcher.
    """
    arguments = _arguments()
    try:
        install_parent_death_signal(arguments.expected_parent_pid)
    except RuntimeError:
        return 1
    remove_leftover_shared_memory()
    from speech_qwen.model import pinned_server_command

    command = pinned_server_command(arguments.cache, arguments.socket)
    os.execv(command[0], command)  # noqa: S606 - the argv is the closed server command
    return 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-parent-pid", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
