"""The stage proof reads the card through `nvidia-smi` and says when it cannot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.support.nvidia_support import TOTAL_MIB, FakeNvidiaSmi
from tests.support.script_support import script_module

if TYPE_CHECKING:
    from pathlib import Path

PROVE = script_module("prove_speech")
USED_MIB = 1234


def test_the_cards_memory_is_its_used_and_total_on_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeNvidiaSmi(tmp_path, monkeypatch).use(USED_MIB)

    assert PROVE.nvidia_memory() == f"{USED_MIB}, {TOTAL_MIB}"


@pytest.mark.parametrize(
    ("reading", "expected"),
    [("11\n222\n", "11 222"), ("", "none")],
    ids=["compute processes on one line", "no compute process"],
)
def test_the_processes_computing_on_the_card_are_listed_on_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reading: str, expected: str
) -> None:
    FakeNvidiaSmi(tmp_path, monkeypatch).print_raw(reading)

    assert PROVE.nvidia_compute_pids() == expected


def test_a_card_that_cannot_be_read_is_a_note_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeNvidiaSmi(tmp_path, monkeypatch).fail()

    assert PROVE.nvidia_compute_pids().startswith("unavailable:")


def test_a_machine_without_nvidia_smi_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))

    assert PROVE.nvidia_compute_pids() == "unavailable: nvidia-smi not found"
