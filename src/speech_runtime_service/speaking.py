"""The closed factory for resident and isolated local speaking engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from speech_runtime_provider_contract import (
    QWEN_MODEL_ID,
    QWEN_SAMPLE_RATE,
)

from speech_runtime_service.provider_process import (
    ProviderId,
    ProviderLaunch,
    ProviderProcess,
    provider_entrypoint,
)
from speech_runtime_service.voices import (
    DEFAULT_SPEAKING_MODEL,
    VoiceId,
    piper_voice_paths,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from speech_runtime_service.config import Settings


class PiperSpeaking:
    """One Piper voice, loaded once and held for the process."""

    streams = False

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

    def pcm_chunks(self, text: str, language: str) -> Generator[bytes]:
        """Yield 16-bit mono PCM as soon as Piper produces a chunk."""
        del language
        if self._voice is None:
            message = "speaking model is not loaded"
            raise RuntimeError(message)
        for chunk in self._voice.synthesize(text):
            yield chunk.audio_int16_bytes

    def close(self) -> None:
        """Piper stays retained for the process and needs no explicit cleanup."""


LocalSpeaking = PiperSpeaking | ProviderProcess


def speaking_for_voice(settings: Settings, voice: VoiceId) -> LocalSpeaking:
    """Construct one supported local engine from the closed catalogue id."""
    if voice is VoiceId.PIPER:
        return PiperSpeaking(DEFAULT_SPEAKING_MODEL, settings.voice_cache)
    if voice is VoiceId.QWEN:
        return ProviderProcess(
            ProviderLaunch(
                model_name=QWEN_MODEL_ID,
                sample_rate=QWEN_SAMPLE_RATE,
                streams=False,
                executable=provider_entrypoint(settings.provider_root, ProviderId.QWEN),
                arguments=(
                    "--device",
                    settings.device,
                    "--cache",
                    str(settings.huggingface_cache),
                    "--speaker",
                    settings.qwen_speaker.value,
                ),
            ),
        )
    raise ValueError
