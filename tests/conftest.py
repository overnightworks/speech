"""The environment every test starts in: the image's catalogue setting."""

import pytest

from shipped_catalogue import CATALOGUE_FILE


@pytest.fixture(autouse=True)
def _shipped_catalogue_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEECH_CATALOGUE_FILE", str(CATALOGUE_FILE))
