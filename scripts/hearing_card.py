"""The hearing stage proof's use of the card: memory samples and a holder of it."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Self

import hearing_figures as figures
from prove_speech import add_cuda_library_path, query_nvidia

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType


SETPRIV = "setpriv"
CARD_HOLDER = Path(__file__).resolve().with_name("hold_card.py")
MEMORY_SAMPLE_SECONDS = 0.25
FREED_TOLERANCE_MIB = 64
FREED_WAIT_SECONDS = 15.0
FREED_POLL_SECONDS = 0.5
HOLDER_STOP_SECONDS = 10


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


@contextlib.contextmanager
def held_card() -> Iterator[int]:
    """Hold the card's memory in a second process; always end that process.

    The holder asks the kernel for TERM when this process dies, so a proof killed
    outright cannot leave the card full.
    """
    environment = os.environ.copy()
    add_cuda_library_path(environment)
    holder = subprocess.Popen(  # noqa: S603 - argv is setpriv with this interpreter and the shipped holder script
        [SETPRIV, "--pdeathsig", "TERM", sys.executable, str(CARD_HOLDER)],
        env=environment,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        said = holder.stdout.readline().split() if holder.stdout else []
        if said[:1] != ["held"] or int(said[1]) <= 0:
            message = f"the card holder did not hold the card: {said}"
            raise SystemExit(message)
        yield int(said[1])
    finally:
        holder.terminate()
        try:
            holder.wait(timeout=HOLDER_STOP_SECONDS)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait()


def card_is_freed(baseline_mib: int) -> bool:
    """True once the card is back within a margin of `baseline_mib`."""
    deadline = time.monotonic() + FREED_WAIT_SECONDS
    while time.monotonic() < deadline:
        if used_mib() <= baseline_mib + FREED_TOLERANCE_MIB:
            return True
        time.sleep(FREED_POLL_SECONDS)
    return False
