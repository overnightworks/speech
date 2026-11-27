"""Whether a process the tests started has ended."""

from pathlib import Path

from tests.support.poll_support import poll


def has_ended(pid: int, *, within: float = 5) -> bool:
    """Whether `pid` is gone or a zombie its adopter has yet to reap, in `within` s."""
    return poll(lambda: _is_gone_or_zombie(pid), within=within)


def _is_gone_or_zombie(pid: int) -> bool:
    try:
        status = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except FileNotFoundError, ProcessLookupError:
        return True
    return status.rpartition(")")[2].split()[0] == "Z"
