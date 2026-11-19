"""The kernel's view of a process the tests started."""

import time
from pathlib import Path


def process_state(pid: int) -> str | None:
    """Read the kernel's state line for `pid`; None once the pid is gone."""
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines()
    except FileNotFoundError, ProcessLookupError:
        return None
    return next(line for line in lines if line.startswith("State:"))


def has_ended(pid: int, *, within: float = 5) -> bool:
    """Whether `pid` is gone or a zombie, within `within` seconds."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        state = process_state(pid)
        if state is None or "zombie" in state:
            return True
        time.sleep(0.01)
    return False
