"""The speaking engines the runtime builds: the local Piper voice and the providers."""

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from speech_provider_contract import QWEN_MODEL_ID, QWEN_SAMPLE_RATE

from speech_client.wire import VOICES, SpeakingModel, VoiceId
from speech_service.config import Settings
from speech_service.speaking import PiperSpeaking, speaking_for_voice
from speech_service.voices import PIPER_VOICES

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("voice", PIPER_VOICES)
def test_every_piper_voice_constructs_its_own_local_engine(voice: VoiceId) -> None:
    engine = speaking_for_voice(Settings(), voice)

    assert isinstance(engine, PiperSpeaking)
    assert engine.model_name == voice.value


@pytest.mark.parametrize(
    "voice", [v for v in VoiceId if VOICES[v].model is SpeakingModel.QWEN]
)
def test_every_qwen_voice_constructs_the_complete_waveform_voice(
    voice: VoiceId,
) -> None:
    engine = speaking_for_voice(Settings(device="cuda"), voice)

    assert engine.model_name == QWEN_MODEL_ID
    assert engine.sample_rate == QWEN_SAMPLE_RATE


def test_piper_load_requires_local_onnx_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[object] = []

    class LocalVoice:
        config = SimpleNamespace(sample_rate=22_050)

    def load(path: object) -> LocalVoice:
        loaded.append(path)
        return LocalVoice()

    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=load)),
    )
    engine = PiperSpeaking("local-voice", tmp_path)

    with pytest.raises(FileNotFoundError):
        engine.load()
    (tmp_path / "local-voice.onnx").touch()
    (tmp_path / "local-voice.onnx.json").touch()

    engine.load()

    assert loaded == [tmp_path / "local-voice.onnx"]
    assert engine.ready is True
