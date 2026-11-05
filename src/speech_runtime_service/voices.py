"""The closed voice catalogue and its side-effect-free artifact check."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Final

from huggingface_hub import try_to_load_from_cache
from speech_runtime_provider_contract import (
    QWEN_ARTIFACTS,
    QWEN_MODEL_ID,
    QWEN_REVISION,
    QWEN_SAMPLE_RATE,
)

from speech_runtime_service.provider_process import (
    ProviderId,
    is_executable_file,
    provider_entrypoint,
)

if TYPE_CHECKING:
    from pathlib import Path

    from speech_runtime_service.config import Settings

DEFAULT_SPEAKING_MODEL = "de_DE-thorsten-medium"


class VoiceId(StrEnum):
    """The closed catalogue of voices the service can speak with."""

    PIPER = "piper"
    QWEN = "qwen3-tts-0.6b"


RESIDENT_VOICE: Final = VoiceId.PIPER
"""The voice loaded at start and always offered; every other voice is a GPU voice."""

_MODELS: Final[dict[VoiceId, str]] = {
    VoiceId.PIPER: DEFAULT_SPEAKING_MODEL,
    VoiceId.QWEN: QWEN_MODEL_ID,
}
_SAMPLE_RATES_BEFORE_LOAD: Final[dict[VoiceId, int]] = {VoiceId.QWEN: QWEN_SAMPLE_RATE}


def model_for_voice(voice: VoiceId) -> str:
    """The speaking model `voice` speaks with."""
    return _MODELS[voice]


def sample_rate_before_load(voice: VoiceId) -> int:
    """The rate `voice` reports before it loads; 0 where its loaded model sets it."""
    return _SAMPLE_RATES_BEFORE_LOAD.get(voice, 0)


def missing_artifacts(settings: Settings, voice: VoiceId) -> tuple[str, ...]:
    """The names of the files `voice` needs that are absent; empty when it is installed.

    Inspect only exact artifacts; this does not read credentials or the network.
    """
    if voice is VoiceId.PIPER:
        return tuple(
            path.name
            for path in piper_voice_paths(settings.voice_cache, DEFAULT_SPEAKING_MODEL)
            if not path.is_file()
        )
    missing = [
        filename
        for filename in QWEN_ARTIFACTS
        if not _is_cached(settings.huggingface_cache, filename)
    ]
    entrypoint = provider_entrypoint(settings.provider_root, ProviderId.QWEN)
    if not is_executable_file(entrypoint):
        missing.append(
            entrypoint.name if entrypoint is not None else "SPEECH_PROVIDER_ROOT"
        )
    return tuple(missing)


def piper_voice_paths(directory: Path, model: str) -> tuple[Path, Path]:
    """The `.onnx` and `.onnx.json` paths of the Piper voice `model` in `directory`."""
    return directory / f"{model}.onnx", directory / f"{model}.onnx.json"


def _is_cached(cache: Path, filename: str) -> bool:
    return isinstance(
        try_to_load_from_cache(
            QWEN_MODEL_ID, filename, cache_dir=cache, revision=QWEN_REVISION
        ),
        str,
    )
