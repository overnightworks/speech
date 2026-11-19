"""The launcher serves the pinned checkpoint from the local Hub cache."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest
from speech_provider_contract import QWEN_MODEL_ID, QWEN_REVISION

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


@pytest.fixture
def model() -> ModuleType:
    from speech_qwen import model as module

    return module


def _fake_vllm_omni(directory: Path) -> Path:
    deploy_config = directory / "vllm_omni" / "deploy" / "qwen3_tts.yaml"
    deploy_config.parent.mkdir(parents=True)
    deploy_config.write_text("stages: []\n", encoding="utf-8")
    deploy_config.parents[1].joinpath("__init__.py").write_text("raise SystemExit\n")
    return deploy_config


def _command(
    model: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Return the command for the fake package and a snapshot the Hub finds."""
    _fake_vllm_omni(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **_: str(tmp_path))
    return model.pinned_server_command(tmp_path, tmp_path / "server.sock")


def test_the_command_serves_the_pinned_local_snapshot_on_the_given_socket(
    model: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "snapshot"
    deploy_config = _fake_vllm_omni(tmp_path)
    snapshots: list[dict[str, object]] = []

    def snapshot_download(**arguments: object) -> str:
        snapshots.append(arguments)
        return str(checkpoint)

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    command = model.pinned_server_command(tmp_path, tmp_path / "server.sock")

    assert snapshots == [
        {
            "repo_id": QWEN_MODEL_ID,
            "repo_type": "model",
            "revision": QWEN_REVISION,
            "allow_patterns": model.QWEN_ARTIFACTS,
            "local_files_only": True,
            "cache_dir": tmp_path,
        }
    ]
    assert command[1:3] == ["serve", str(checkpoint)]
    assert command[command.index("--uds") + 1] == str(tmp_path / "server.sock")
    assert command[command.index("--deploy-config") + 1] == str(deploy_config)
    assert "--trust-remote-code" not in command
    assert "vllm_omni" not in sys.modules


def test_the_cards_reservation_leaves_room_for_another_user(
    model: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = _command(model, tmp_path, monkeypatch)

    stages = json.loads(command[command.index("--stage-overrides") + 1])

    reserved = sum(stage["gpu_memory_utilization"] for stage in stages.values())
    assert 0 < reserved < 1


def test_a_command_without_vllm_omni_installed_fails(
    model: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **_: str(tmp_path))
    monkeypatch.setitem(sys.modules, "vllm_omni", None)

    with pytest.raises(ModuleNotFoundError):
        model.pinned_server_command(tmp_path, tmp_path / "server.sock")
