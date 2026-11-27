"""The gpu image offers the catalogue voices."""

import re

import pytest

from shipped_catalogue import CATALOGUE
from speech_service.catalogue import offered_voices
from speech_service.config import Settings
from tests.support.browser_support import REPOSITORY

_DOCKERFILE = (REPOSITORY / "Dockerfile").read_text()
_STAGES = re.split(r"^FROM ", _DOCKERFILE, flags=re.MULTILINE)[1:]


def _settings_of(stage_name: str, monkeypatch: pytest.MonkeyPatch) -> Settings:
    stage = next(
        stage for stage in _STAGES if re.match(rf"\S+ AS {stage_name}\n", stage)
    )
    for name, value in re.findall(r"\b(SPEECH_[A-Z_]+)=(\S+)", stage):
        monkeypatch.setenv(name, value)
    return Settings()


@pytest.mark.parametrize("stage_name", ["gpu-weights", "gpu"])
def test_a_gpu_stage_offers_every_catalogue_voice_by_its_own_settings(
    stage_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_of(stage_name, monkeypatch)

    assert settings.voices is None
    assert offered_voices(
        CATALOGUE, settings.voices, settings.carried_engines
    ) == tuple(CATALOGUE)
