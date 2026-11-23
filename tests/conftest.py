"""The environment every test starts in: the catalogue setting, telemetry off."""

import pytest

from shipped_catalogue import CATALOGUE_FILE
from speech_service.onnxruntime_telemetry import disable_onnxruntime_telemetry

disable_onnxruntime_telemetry()


@pytest.fixture(autouse=True)
def _shipped_catalogue_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEECH_CATALOGUE_FILE", str(CATALOGUE_FILE))
