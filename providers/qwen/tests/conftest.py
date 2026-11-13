"""Fixtures the Qwen provider's tests share."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import pytest

from support.scripted_server import ScriptedSpeechServer

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def private_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


@pytest.fixture
def speech_server(tmp_path: Path) -> Iterator[ScriptedSpeechServer]:
    scripted = ScriptedSpeechServer(tmp_path / "speech.sock")
    yield scripted
    scripted.close()
