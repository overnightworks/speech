"""The kernel's view of a process the tests started."""

import time
from pathlib import Path


def process_state(pid: int) -> str | None:
    """Read the kernel's state line for `pid`; None once the pid is gone."""
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines()
    except (FileNotFoundError, ProcessLookupError):
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


def _live_group_members(group: int) -> list[int]:
    members = []
    for entry in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = entry.read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError):
            continue
        state, _parent, process_group = stat.rpartition(")")[2].split()[:3]
        if int(process_group) == group and state != "Z":
            members.append(int(entry.parent.name))
    return members


def group_has_ended(group: int, *, within: float = 10) -> bool:
    """Whether no live process of `group` is left, within `within` seconds."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not _live_group_members(group):
            return True
        time.sleep(0.01)
    return False


def await_pid_file(path: Path, *, within: float = 10) -> int:
    """Read the pid a process writes to `path`; fail when it never comes."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if path.exists() and (text := path.read_text(encoding="ascii")):
            return int(text)
        time.sleep(0.01)
    raise AssertionError(path)
