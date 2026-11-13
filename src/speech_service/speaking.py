"""The closed factory for resident and isolated local speaking engines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never

from speech_provider_contract import (
    QWEN_DEVICE,
    QWEN_MODEL_ID,
    QWEN_SAMPLE_RATE,
    QwenSpeaker,
)

from speech_client.wire import VOICES, SpeakingModel, VoiceId
from speech_service.provider_process import (
    ProviderId,
    ProviderLaunch,
    ProviderProcess,
    provider_entrypoint,
)
from speech_service.voices import piper_voice_paths

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from speech_service.config import Settings


class PiperSpeaking:
    """One Piper voice, loaded once and held for the process."""

    def __init__(self, model_name: str, cache: Path) -> None:
        """Remember the catalogue name and where ONNX files are kept."""
        self.model_name = model_name
        self.sample_rate = 0
        self.ready = False
        self._cache = cache
        self._voice = None

    def load(self) -> None:
        """Load the exact local Piper artifacts and keep them resident."""
        from piper import PiperVoice

        onnx, metadata = piper_voice_paths(self._cache, self.model_name)
        if not onnx.is_file() or not metadata.is_file():
            raise FileNotFoundError
        voice = PiperVoice.load(onnx)
        self._voice = voice
        self.sample_rate = voice.config.sample_rate
        self.ready = True

    def pcm_chunks(
        self, text: str, language: str, instructions: str | None = None
    ) -> Generator[bytes]:
        """Yield 16-bit mono PCM as soon as Piper produces a chunk."""
        del language, instructions
        if self._voice is None:
            message = "speaking model is not loaded"
            raise RuntimeError(message)
        for chunk in self._voice.synthesize(text):
            yield chunk.audio_int16_bytes

    def close(self) -> None:
        """Piper stays retained for the process and needs no explicit cleanup."""


LocalSpeaking = PiperSpeaking | ProviderProcess


_QWEN_SPEAKERS: Final = {VoiceId.RYAN: QwenSpeaker.RYAN}


def speaking_for_voice(settings: Settings, voice: VoiceId) -> LocalSpeaking:
    """Construct one supported local engine from the closed catalogue id."""
    match VOICES[voice].model:
        case SpeakingModel.PIPER:
            return PiperSpeaking(voice.value, settings.voice_cache)
        case SpeakingModel.QWEN:
            return ProviderProcess(
                ProviderLaunch(
                    model_name=QWEN_MODEL_ID,
                    sample_rate=QWEN_SAMPLE_RATE,
                    executable=provider_entrypoint(
                        settings.provider_root, ProviderId.QWEN
                    ),
                    arguments=(
                        "--device",
                        QWEN_DEVICE,
                        "--cache",
                        str(settings.huggingface_cache),
                        "--speaker",
                        _QWEN_SPEAKERS[voice].value,
                    ),
                ),
            )
        case unreachable:
            assert_never(unreachable)
