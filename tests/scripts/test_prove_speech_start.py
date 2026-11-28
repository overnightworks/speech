"""The proof's service starts from the checkout on its own identity and ends with it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from shipped_catalogue import PIPER_VOICES
from speech_service.hearing_models import HearingModel
from speech_service.voices import piper_voice_paths
from tests.support.gpu_voice_support import _RECORDING_PIPER
from tests.support.script_support import script_module

HEARING = script_module("prove_hearing")
PROVE = script_module("prove_speech")
SCRIPTS = Path(PROVE.__file__).parent


@pytest.fixture
def piper_beside_the_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The service finds fake Piper voice files, and a fake `piper` module."""
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "piper.py").write_text(_RECORDING_PIPER, encoding="utf-8")
    voice_cache = tmp_path / "voices"
    voice_cache.mkdir()
    for voice in PIPER_VOICES:
        for path in piper_voice_paths(voice_cache, voice.value):
            path.touch()
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(modules), str(SCRIPTS)]))
    monkeypatch.setenv("SPEECH_VOICE_CACHE", str(voice_cache))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))


@pytest.mark.usefixtures("piper_beside_the_checkout")
def test_the_proofs_service_starts_on_its_own_identity_and_leaves_no_process() -> None:
    start = PROVE.Start(set(PIPER_VOICES), device="cpu")

    with PROVE.started_speech(start) as started:
        service = started.service
        pid = service.running.pid
        directory = started.work_directory
        offered = set(service.running.health.speaking.voices)
        # The fake voice cannot synthesise, so a request its caller identity passes
        # ends in the aborted body; a refused identity ends in its status.
        with pytest.raises(SystemExit, match="speak body ended before it was complete"):
            PROVE.speak_measured(
                service.base_url, "Hallo", PIPER_VOICES[0], service.minting
            )

        assert offered == set(PIPER_VOICES)
        assert Path(f"/proc/{pid}").is_dir()

    assert not directory.exists()
    assert not Path(f"/proc/{pid}").exists()


@pytest.mark.usefixtures("piper_beside_the_checkout")
def test_a_hearing_model_missing_from_the_cache_ends_the_start_when_the_service_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEECH_HUGGINGFACE_CACHE", str(tmp_path / "empty-cache"))

    with (
        pytest.raises(SystemExit, match="service exited with status"),
        HEARING.hearing_service(
            "cpu", HearingModel.SMALL.value, PROVE.PIPER_PROOF_VOICE
        ),
    ):
        pass
