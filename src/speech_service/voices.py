"""What the service knows of a catalogue voice: its start, its files, Qwen's fetch."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, assert_never

from huggingface_hub import snapshot_download, try_to_load_from_cache
from speech_provider_contract import (
    QWEN_ARTIFACTS,
    QWEN_MODEL_ID,
    QWEN_REVISION,
    QWEN_SAMPLE_RATE,
)

from speech_client.wire import VOICES, SpeakingModel, VoiceId
from speech_service.provider_process import (
    ProviderId,
    is_executable_file,
    provider_entrypoint,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from speech_service.config import Settings

PIPER_VOICES: Final = tuple(
    voice for voice in VoiceId if VOICES[voice].model is SpeakingModel.PIPER
)
"""The catalogue's Piper voices, in catalogue order."""

_SAMPLE_RATES_BEFORE_LOAD: Final[dict[SpeakingModel, int]] = {
    SpeakingModel.QWEN: QWEN_SAMPLE_RATE
}


def loads_at_start(voice: VoiceId, *, gpu_voices: bool = False) -> bool:
    """Whether the start loads `voice` rather than its first request.

    A Piper voice always loads at start; a GPU voice only when `gpu_voices`.
    """
    return gpu_voices or VOICES[voice].model is SpeakingModel.PIPER


def sample_rate_before_load(voice: VoiceId) -> int:
    """The rate `voice` reports before it loads; 0 where its loaded model sets it."""
    return _SAMPLE_RATES_BEFORE_LOAD.get(VOICES[voice].model, 0)


def missing_artifacts(settings: Settings, voice: VoiceId) -> tuple[str, ...]:
    """The names of the files `voice` needs that are absent; empty when it is installed.

    Inspect only exact artifacts; this does not read credentials or the network.
    """
    match VOICES[voice].model:
        case SpeakingModel.PIPER:
            return tuple(
                path.name
                for path in piper_voice_paths(settings.voice_cache, voice.value)
                if not path.is_file()
            )
        case SpeakingModel.QWEN:
            return _missing_qwen_artifacts(settings)
        case unreachable:
            assert_never(unreachable)


def _missing_qwen_artifacts(settings: Settings) -> tuple[str, ...]:
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


def fetch_qwen(cache: Path, download: Callable[..., str] = snapshot_download) -> Path:
    """Download Qwen's pinned files into the Hub cache `cache`; return their snapshot.

    No token is sent. `download` is the Hugging Face snapshot downloader.
    """
    return Path(
        download(
            QWEN_MODEL_ID,
            revision=QWEN_REVISION,
            cache_dir=cache,
            allow_patterns=list(QWEN_ARTIFACTS),
            token=False,
        )
    )


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
