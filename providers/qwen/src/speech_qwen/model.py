"""The pinned Qwen3-TTS 1.7B voice, spoken through a vLLM-Omni server child."""

from __future__ import annotations

import importlib.resources
import importlib.util
import json
import sys
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

from speech_provider_contract import (
    MAX_PCM_BYTES,
    QWEN_ARTIFACTS,
    QWEN_DEVICE,
    QWEN_MODEL_ID,
    QWEN_REVISION,
    QwenSpeaker,
)

from speech_qwen.server import Server, fresh_socket_path, start_server

if TYPE_CHECKING:
    from collections.abc import Iterator
    from http.client import HTTPResponse
    from importlib.resources.abc import Traversable

_LANGUAGE_NAMES = {"de": "German", "en": "English"}
_RESPONSE_SECONDS = 120.0
"""How long the server may stay silent in the middle of a response."""
_SAMPLE_BYTES = 2
# The deploy config reserves 0.3 of the card for each stage, 14.4 GB on a
# 24 GB card, while both stages together measured 9.4 GB resident. The talker
# keeps its share; the decoder takes half of its own.
_STAGE_MEMORY_FRACTIONS = {"0": 0.3, "1": 0.15}


class SpeechRequestError(RuntimeError):
    """The server refused a speech request or ended its audio inside a sample."""


def load_model(device: str, cache: Path) -> Server:
    """Start the server on the pinned checkpoint in the local Hub cache."""
    if device != QWEN_DEVICE:
        message = "unsupported Qwen device"
        raise ValueError(message)

    from huggingface_hub import snapshot_download

    checkpoint = snapshot_download(
        repo_id=QWEN_MODEL_ID,
        repo_type="model",
        revision=QWEN_REVISION,
        allow_patterns=QWEN_ARTIFACTS,
        local_files_only=True,
        cache_dir=cache,
    )
    socket_path = fresh_socket_path()
    with importlib.resources.as_file(_deploy_config()) as deploy_config:
        return start_server(
            _server_command(Path(checkpoint), deploy_config, socket_path), socket_path
        )


def _deploy_config() -> Traversable:
    # Importing vllm_omni would load torch and patch vllm in a process that
    # never runs a model; its spec locates the package without running it.
    spec = importlib.util.find_spec("vllm_omni")
    if spec is None:
        raise ModuleNotFoundError(name="vllm_omni")
    package = importlib.util.module_from_spec(spec)
    return importlib.resources.files(package) / "deploy" / "qwen3_tts.yaml"


def _server_command(
    checkpoint: Path, deploy_config: Path, socket_path: Path
) -> list[str]:
    stage_overrides = {
        stage: {"gpu_memory_utilization": fraction}
        for stage, fraction in _STAGE_MEMORY_FRACTIONS.items()
    }
    return [
        str(Path(sys.executable).with_name("vllm-omni")),
        "serve",
        str(checkpoint),
        "--omni",
        "--deploy-config",
        str(deploy_config),
        "--stage-overrides",
        json.dumps(stage_overrides),
        "--uds",
        str(socket_path),
    ]


def pcm_chunks(
    server: Server,
    text: str,
    language: str,
    instructions: str | None = None,
    *,
    speaker: QwenSpeaker | str,
) -> Iterator[bytes]:
    """Yield signed PCM frames as the server streams them, none past `MAX_PCM_BYTES`."""
    language_name = _LANGUAGE_NAMES.get(language)
    if language_name is None:
        message = "unsupported Qwen language"
        raise ValueError(message)
    request: dict[str, object] = {
        "input": text,
        "voice": QwenSpeaker(speaker).value,
        "language": language_name,
        "response_format": "pcm",
        "stream": True,
        "stream_format": "audio",
    }
    if instructions is not None:
        request["instructions"] = instructions
    connection = server.connect(_RESPONSE_SECONDS)
    try:
        connection.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(request),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != HTTPStatus.OK:
            raise SpeechRequestError
        yield from _whole_samples(response)
    finally:
        connection.close()


def _whole_samples(response: HTTPResponse) -> Iterator[bytes]:
    """Re-cut what arrives into frames of whole samples, each as it arrives."""
    carried = b""
    while arrived := response.read1(MAX_PCM_BYTES - len(carried)):
        available = carried + arrived
        whole = len(available) - len(available) % _SAMPLE_BYTES
        carried = available[whole:]
        if whole:
            yield available[:whole]
    if carried:
        raise SpeechRequestError
