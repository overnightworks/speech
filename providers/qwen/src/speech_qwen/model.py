"""The pinned Qwen3-TTS 1.7B voice, served by a vLLM-Omni server."""

from __future__ import annotations

import importlib.resources
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from speech_provider_contract import QWEN_ARTIFACTS, QWEN_MODEL_ID, QWEN_REVISION

from speech_qwen.server import Server, fresh_socket_path

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from importlib.resources.abc import Traversable

LANGUAGE_NAMES = {"de": "German", "en": "English"}
# The deploy config reserves 0.3 of the card for each stage, 14.4 GB on a
# 24 GB card, while both stages together measured 9.4 GB resident. The talker
# keeps its share; the decoder takes half of its own.
_STAGE_MEMORY_FRACTIONS = {"0": 0.3, "1": 0.15}


def pinned_server_command(cache: Path, socket_path: Path) -> list[str]:
    """Return the command that serves the pinned checkpoint of the local Hub `cache`.

    The checkpoint resolves offline: a snapshot the cache lacks fails here.
    """
    from huggingface_hub import snapshot_download

    checkpoint = snapshot_download(
        repo_id=QWEN_MODEL_ID,
        repo_type="model",
        revision=QWEN_REVISION,
        allow_patterns=QWEN_ARTIFACTS,
        local_files_only=True,
        cache_dir=cache,
    )
    with importlib.resources.as_file(_deploy_config()) as deploy_config:
        return _server_command(Path(checkpoint), deploy_config, socket_path)


def start_on_checkpoint(
    checkpoint: Path, start: Callable[[Sequence[str], Path], Server]
) -> Server:
    """Start the vLLM-Omni server with `start` on the local `checkpoint`."""
    socket_path = fresh_socket_path()
    with importlib.resources.as_file(_deploy_config()) as deploy_config:
        return start(
            _server_command(checkpoint, deploy_config, socket_path), socket_path
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
