"""The hearing proof's use of the card ends its holder and never understates a peak."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.support.script_support import script_module

CARD = script_module("hearing_card")
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
DEADLINE_SECONDS = 10.0
HELD_MIB = 123

FAKE_HOLDER = f"""
import os, pathlib, threading
pathlib.Path(os.environ["HOLDER_PIDFILE"]).write_text(str(os.getpid()))
print("held", {HELD_MIB}, flush=True)
threading.Event().wait()
"""

KILLED_PARENT = """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import hearing_card
hearing_card.CARD_HOLDER = Path(sys.argv[2])
with hearing_card.held_card():
    print("holding", flush=True)
    time.sleep(60)
"""


def fake_nvidia(directory: Path, readings: list[str]) -> None:
    """Put an `nvidia-smi` on PATH that prints the readings, then fails."""
    counter = directory / "calls"
    counter.write_text("0")
    script = directory / "nvidia-smi"
    lines = "\n".join(
        f'  {n}) echo "{reading}";;' for n, reading in enumerate(readings)
    )
    script.write_text(
        f"""#!/bin/sh
n=$(cat {counter})
echo $((n + 1)) > {counter}
case $n in
{lines}
  *) exit 1;;
esac
"""
    )
    script.chmod(0o755)


def wait_for_calls(counter: Path, calls: int) -> None:
    deadline = time.monotonic() + DEADLINE_SECONDS
    while int(counter.read_text()) < calls:
        assert time.monotonic() < deadline
        time.sleep(0.05)


def is_running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


def wait_until_gone(pid: int) -> bool:
    deadline = time.monotonic() + DEADLINE_SECONDS
    while time.monotonic() < deadline:
        if not is_running(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def holder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "fake_holder.py"
    script.write_text(FAKE_HOLDER)
    pidfile = tmp_path / "holder.pid"
    monkeypatch.setattr(CARD, "CARD_HOLDER", script)
    monkeypatch.setenv("HOLDER_PIDFILE", str(pidfile))
    return pidfile


def hold_and_fail(pidfile: Path) -> None:
    with CARD.held_card() as held:
        assert held == HELD_MIB
        assert pidfile.exists()
        message = "body failed"
        raise RuntimeError(message)


def test_the_holder_ends_when_the_body_of_the_hold_raises(holder: Path) -> None:
    with pytest.raises(RuntimeError, match="body failed"):
        hold_and_fail(holder)

    assert wait_until_gone(int(holder.read_text()))


def test_the_holder_ends_when_the_proof_is_killed_outright(
    holder: Path, tmp_path: Path
) -> None:
    fake = tmp_path / "fake_holder.py"
    proof = subprocess.Popen(
        [sys.executable, "-c", KILLED_PARENT, str(SCRIPTS), str(fake)],
        stdout=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    try:
        assert proof.stdout is not None
        assert proof.stdout.readline().strip() == "holding"
        pid = int(holder.read_text())
        proof.send_signal(signal.SIGKILL)
        proof.wait()

        assert wait_until_gone(pid)
    finally:
        proof.kill()
        proof.wait()
        if holder.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(holder.read_text()), signal.SIGKILL)


def test_a_failed_sample_fails_the_measurement_instead_of_shortening_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readings = ["100, 1000", "300, 1000"]
    fake_nvidia(tmp_path, readings)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(ValueError), CARD.MemorySampler():  # noqa: PT011 - the parse error of the failed reading
        wait_for_calls(tmp_path / "calls", len(readings) + 1)


def test_the_samples_of_a_healthy_card_give_its_highest_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_nvidia(tmp_path, ["100, 1000"] * 50)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    with CARD.MemorySampler() as sampler:
        pass

    assert sampler.samples == [100]


@pytest.mark.parametrize(
    ("readings", "freed"),
    [(["100, 1000"], True), (["900, 1000"] * 50, False)],
    ids=["back at the baseline", "still held"],
)
def test_the_card_is_freed_once_it_is_back_at_the_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    readings: list[str],
    *,
    freed: bool,
) -> None:
    fake_nvidia(tmp_path, readings)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(CARD, "FREED_WAIT_SECONDS", 0.2)
    monkeypatch.setattr(CARD, "FREED_POLL_SECONDS", 0.01)

    assert CARD.card_is_freed(100) is freed
