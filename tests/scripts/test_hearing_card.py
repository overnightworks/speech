"""The hearing proof's use of the card ends its holder and never understates a peak."""

from __future__ import annotations

import contextlib
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.support.clock_support import FakeClock
from tests.support.nvidia_support import FakeNvidiaSmi
from tests.support.script_support import script_module

CARD = script_module("hearing_card")
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
DEADLINE_SECONDS = 10.0
HELD_MIB = 123
READING_MIB = 100

FAKE_HOLDER = f"""
import os, pathlib, threading
pathlib.Path(os.environ["HOLDER_PIDFILE"]).write_text(str(os.getpid()))
if "HOLDER_FIFO" in os.environ:
    lifeline = open(os.environ["HOLDER_FIFO"], "w")
for line in os.environ.get("HOLDER_NOISE", "").splitlines():
    print(line, flush=True)
if "HOLDER_SILENT" in os.environ:
    threading.Event().wait()
print("held", {HELD_MIB}, flush=True)
if "HOLDER_EXITS" not in os.environ:
    threading.Event().wait()
"""

KILLED_PARENT = """
import sys, threading
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import hearing_card
hearing_card.CARD_HOLDER = Path(sys.argv[2])
with hearing_card.held_card():
    print("holding", flush=True)
    threading.Event().wait()
"""


def has_exited(pid: int) -> bool:
    """Block until the process is a zombie or gone, within the test's deadline."""
    deadline = time.monotonic() + DEADLINE_SECONDS
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return True
        if state in {"Z", "X"}:
            return True
    return False


def holder_ends(lifeline: int) -> bool:
    """Block until the holder, the lifeline's only writer, has ended."""
    hung_up, _, _ = select.select([lifeline], [], [], DEADLINE_SECONDS)
    return bool(hung_up)


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

    assert not Path(f"/proc/{holder.read_text()}").exists()


@pytest.mark.usefixtures("holder")
def test_the_holders_line_is_found_behind_a_line_a_native_library_printed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "HOLDER_NOISE", "[2026-11-02 15:03:01.907] [info] Using CUDA allocator: x"
    )

    with CARD.held_card() as held:
        assert held == HELD_MIB


def test_a_holder_that_never_says_it_holds_the_card_fails_with_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "silent_holder.py"
    script.write_text("print('[info] Using CUDA allocator: x', flush=True)")
    monkeypatch.setattr(CARD, "CARD_HOLDER", script)

    with pytest.raises(SystemExit, match="Using CUDA allocator"), CARD.held_card():
        pass


def test_a_holder_that_stays_silent_ends_the_stage_at_its_own_deadline(
    holder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOLDER_SILENT", "1")
    monkeypatch.setenv("HOLDER_NOISE", "[info] loading the model")

    with (
        pytest.raises(SystemExit, match=r"said nothing.*loading the model"),
        CARD.held_card(wait_seconds=0.5),
    ):
        pytest.fail("the stage ran without a held card")

    assert not Path(f"/proc/{holder.read_text()}").exists()


def test_a_holder_that_ends_after_its_line_fails_the_stage(
    holder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOLDER_EXITS", "1")

    with pytest.raises(SystemExit, match="ended with status 0"), CARD.held_card():
        assert has_exited(int(holder.read_text()))


def test_the_holder_ends_when_the_proof_is_killed_outright(
    holder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fifo = tmp_path / "lifeline"
    os.mkfifo(fifo)
    monkeypatch.setenv("HOLDER_FIFO", str(fifo))
    lifeline = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    proof = subprocess.Popen(
        [
            sys.executable,
            "-c",
            KILLED_PARENT,
            str(SCRIPTS),
            str(tmp_path / "fake_holder.py"),
        ],
        stdout=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    try:
        assert proof.stdout is not None
        assert proof.stdout.readline().strip() == "holding"
        proof.send_signal(signal.SIGKILL)
        proof.wait()

        assert holder_ends(lifeline)
    finally:
        os.close(lifeline)
        proof.kill()
        proof.wait()
        if holder.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(holder.read_text()), signal.SIGKILL)


def test_a_failed_sample_fails_the_measurement_instead_of_shortening_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeNvidiaSmi(tmp_path, monkeypatch).fail()

    with pytest.raises(ValueError, match="unavailable"), CARD.MemorySampler():
        pass


def test_a_healthy_card_keeps_its_reading_among_the_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeNvidiaSmi(tmp_path, monkeypatch).use(READING_MIB)

    with CARD.MemorySampler() as sampler:
        pass

    assert READING_MIB in sampler.samples


@pytest.mark.parametrize(
    ("used_mib", "freed"),
    [(100, True), (900, False)],
    ids=["back at the baseline", "still held"],
)
def test_the_card_is_freed_once_it_is_back_at_the_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    used_mib: int,
    *,
    freed: bool,
) -> None:
    FakeNvidiaSmi(tmp_path, monkeypatch).use(used_mib)
    clock = FakeClock()

    assert CARD.card_is_freed(100, clock.now, clock.jump) is freed
