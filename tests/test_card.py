"""The card's used memory comes from nvidia-smi and is 0 whenever it cannot be read."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from speech_service.card import card_memory_mb

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fake_nvidia_smi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty PATH directory; a test writes its nvidia-smi into it."""
    directory = tmp_path / "bin"
    directory.mkdir()
    monkeypatch.setenv("PATH", str(directory))
    return directory / "nvidia-smi"


def _install(executable: Path, body: str) -> None:
    executable.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    executable.chmod(0o755)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("echo 1234", 1234),
        ("echo '2048, extra'", 2048),
        ("printf '512\\n1024\\n'", 512),
        ("exit 1", 0),
        (":", 0),
        ("echo 'N/A'", 0),
    ],
    ids=[
        "plain_number",
        "first_csv_field",
        "first_card_only",
        "failing_call",
        "empty_output",
        "not_a_number",
    ],
)
def test_card_memory_reads_the_first_card_or_reports_zero(
    fake_nvidia_smi: Path, body: str, expected: int
) -> None:
    _install(fake_nvidia_smi, body)
    assert card_memory_mb() == expected


@pytest.mark.usefixtures("fake_nvidia_smi")
def test_card_memory_is_zero_without_an_nvidia_smi_binary() -> None:
    assert card_memory_mb() == 0
