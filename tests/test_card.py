"""The card's used memory comes from nvidia-smi and is 0 whenever it cannot be read."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from speech_service.card import card_memory_mb
from tests.support.nvidia_support import FakeNvidiaSmi

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def nvidia_smi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeNvidiaSmi:
    return FakeNvidiaSmi(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("1234\n", 1234),
        ("2048, extra\n", 2048),
        ("512\n1024\n", 512),
        ("", 0),
        ("N/A\n", 0),
    ],
    ids=[
        "plain_number",
        "first_csv_field",
        "first_card_only",
        "empty_output",
        "not_a_number",
    ],
)
def test_card_memory_reads_the_first_card_or_reports_zero(
    nvidia_smi: FakeNvidiaSmi, output: str, expected: int
) -> None:
    nvidia_smi.print_raw(output)
    assert card_memory_mb() == expected


def test_card_memory_is_zero_when_the_call_fails(nvidia_smi: FakeNvidiaSmi) -> None:
    nvidia_smi.fail()
    assert card_memory_mb() == 0


def test_card_memory_is_zero_without_an_nvidia_smi_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    assert card_memory_mb() == 0
