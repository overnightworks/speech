"""The turn end measurement refuses a count that leaves nothing to measure."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "measure_turn_end.py"
BAD_ARGUMENT_EXIT = 2


@pytest.mark.parametrize(
    "option", ["--recordings-per-language", "--transcribed-per-language"]
)
@pytest.mark.parametrize("count", ["0", "-3"])
def test_a_count_below_one_fails_loud(option: str, count: str) -> None:
    run = subprocess.run(
        ["uv", "run", "--script", str(SCRIPT), f"{option}={count}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )

    assert run.returncode == BAD_ARGUMENT_EXIT
    assert "not a positive count" in run.stderr
