"""The hearing stage proof's use of the card: memory samples and a holder of it."""

from __future__ import annotations

import concurrent.futures
import contextlib
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Self

import hearing_figures as figures
from prove_speech import (
    add_cuda_library_path,
    query_nvidia,
    start_with_parent_death,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from types import TracebackType


CARD_HOLDER = Path(__file__).resolve().with_name("hold_card.py")
MEMORY_SAMPLE_SECONDS = 0.25
FREED_TOLERANCE_MIB = 64
FREED_WAIT_SECONDS = 15.0
FREED_POLL_SECONDS = 0.5
HOLDER_STOP_SECONDS = 10
HELD_WAIT_SECONDS = 120.0
HELD_LINE = re.compile(r"held (?P<mib>\d+)")


def used_mib() -> int:
    """The card's used memory in mebibytes; fails loud when it cannot be read."""
    return figures.parse_memory(
        query_nvidia(
            "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"
        )
    ).used_mib


class MemorySampler:
    """Samples the card's used memory in a thread while it is entered."""

    def __init__(self) -> None:
        """Prepare to sample; nothing runs before entry."""
        self.samples: list[int] = []
        self._failure: Exception | None = None
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        try:
            while True:
                self.samples.append(used_mib())
                if self._stopped.wait(MEMORY_SAMPLE_SECONDS):
                    return
        except Exception as failure:  # noqa: BLE001 - kept and raised from __exit__
            self._failure = failure

    def __enter__(self) -> Self:
        """Start sampling."""
        self._thread.start()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        """Stop sampling; a failed sample would understate the peak, so raise it."""
        self._stopped.set()
        self._thread.join()
        if self._failure is not None and error is None:
            raise self._failure


def _read_held_mib(lines: Iterable[str], said: list[str]) -> int:
    """The mebibytes of the holder's `held <n>` line; every line read goes to `said`.

    A native library may write its own log lines on the holder's stdout first, so
    only a line that is exactly the holder's sentence counts. Zero when the output
    ends without it.
    """
    for line in lines:
        said.append(line.rstrip("\n"))
        match = HELD_LINE.fullmatch(said[-1])
        if match:
            return int(match["mib"])
    return 0


def _await_held_mib(holder: subprocess.Popen[str], wait_seconds: float) -> int:
    """The mebibytes the holder says it holds; ends the stage when it never says."""
    said: list[str] = []
    reader = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending = reader.submit(_read_held_mib, holder.stdout or (), said)
    try:
        held_mib = pending.result(timeout=wait_seconds)
    except concurrent.futures.TimeoutError:
        message = f"the card holder said nothing within {wait_seconds}s: {said}"
        raise SystemExit(message) from None
    finally:
        reader.shutdown(wait=False)
    if held_mib <= 0:
        message = f"the card holder did not hold the card: {said}"
        raise SystemExit(message)
    return held_mib


@contextlib.contextmanager
def held_card(wait_seconds: float = HELD_WAIT_SECONDS) -> Iterator[int]:
    """Hold the card's memory in a second process; always end that process.

    The holder asks the kernel for TERM when the thread that entered this ends, so
    a proof killed outright cannot leave the card full. Enter it on the proof's
    main thread. A holder that ended while the stage ran fails the stage, since
    its memory was no longer held.
    """
    environment = os.environ.copy()
    add_cuda_library_path(environment)
    holder = start_with_parent_death(
        [sys.executable, str(CARD_HOLDER)],
        env=environment,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        yield _await_held_mib(holder, wait_seconds)
        if holder.poll() is not None:
            message = (
                f"the card holder ended with status {holder.returncode}"
                " while the stage ran"
            )
            raise SystemExit(message)
    finally:
        holder.terminate()
        try:
            holder.wait(timeout=HOLDER_STOP_SECONDS)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait()


def card_is_freed(
    baseline_mib: int,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """True once the card is back within a margin of `baseline_mib`."""
    deadline = clock() + FREED_WAIT_SECONDS
    while clock() < deadline:
        if used_mib() <= baseline_mib + FREED_TOLERANCE_MIB:
            return True
        sleep(FREED_POLL_SECONDS)
    return False
