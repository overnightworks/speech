"""Whether a process the tests started has ended."""

import time
from pathlib import Path


def has_ended(pid: int, *, within: float = 5) -> bool:
    """Whether `pid` is gone or a zombie its adopter has yet to reap, in `within` s."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            status = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        except FileNotFoundError, ProcessLookupError:
            return True
        if status.rpartition(")")[2].split()[0] == "Z":
            return True
        time.sleep(0.01)
    return False
