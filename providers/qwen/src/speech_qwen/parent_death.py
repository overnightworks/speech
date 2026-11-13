"""The parent-death signal a worker and the server it starts both carry."""

import ctypes
import os
import signal

_PR_SET_PDEATHSIG = 1


def install_parent_death_signal(expected_parent_pid: int) -> None:
    """Kill this process when its exact creator thread ends.

    Fails when the creator is already gone, since the signal would never come.
    """
    if (
        ctypes.CDLL(None).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL) != 0
        or os.getppid() != expected_parent_pid
    ):
        raise RuntimeError
