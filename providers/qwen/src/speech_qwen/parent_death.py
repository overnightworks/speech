"""The parent-death signal the launcher and the voice-design server carry."""

import ctypes
import os
import signal

_PR_SET_PDEATHSIG = 1


def install_parent_death_signal(
    expected_parent_pid: int, signal_number: signal.Signals
) -> None:
    """Send this process `signal_number` when its exact creator thread ends.

    Fails when the creator is already gone, since the signal would never come.
    """
    if (
        ctypes.CDLL(None).prctl(_PR_SET_PDEATHSIG, signal_number) != 0
        or os.getppid() != expected_parent_pid
    ):
        raise RuntimeError
