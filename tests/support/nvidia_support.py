"""A fake `nvidia-smi` on PATH whose reading a test changes between calls."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

TOTAL_MIB = 24_000


class FakeNvidiaSmi:
    """The card as `nvidia-smi` reports it; each call prints the current reading."""

    def __init__(self, directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Put the fake first on PATH, with a card that reports nothing yet."""
        self._reading = directory / "reading"
        script = directory / "nvidia-smi"
        script.write_text(f'#!/bin/sh\nexec cat "{self._reading}"\n')
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")

    def use(self, used_mib: int) -> None:
        """Report `used_mib` from the next call on; a call never sees half a reading."""
        partial = self._reading.with_name("reading.partial")
        partial.write_text(f"{used_mib}, {TOTAL_MIB}\n")
        partial.replace(self._reading)

    def fail(self) -> None:
        """Make every call fail, as a card that cannot be read does."""
        self._reading.unlink(missing_ok=True)
