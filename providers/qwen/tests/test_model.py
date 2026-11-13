"""The Qwen adapter starts a closed server and streams whole-sample PCM from it."""

from __future__ import annotations

import json
import sys
import threading
from typing import TYPE_CHECKING

import pytest
from speech_provider_contract import (
    MAX_PCM_BYTES,
    QWEN_DEVICE,
    QWEN_MODEL_ID,
    QWEN_REVISION,
)

from support.scripted_server import HANG_UP

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import ModuleType

    from support.scripted_server import ScriptedSpeechServer


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


class _Started:
    """What a stand-in server starter saw, and the server it hands back."""

    def __init__(self) -> None:
        self.server = object()
        self.calls: list[tuple[list[str], Path]] = []

    def __call__(self, command: list[str], socket_path: Path) -> object:
        self.calls.append((command, socket_path))
        return self.server


def _load(
    model: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _Started:
    """Load with the fake package and a stand-in starter, which is returned."""
    started = _Started()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **_: str(tmp_path))
    assert model.load_model(QWEN_DEVICE, tmp_path, start=started) is started.server
    return started


def test_load_starts_the_server_on_the_pinned_local_snapshot_and_a_fresh_socket(
    model: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "snapshot"
    deploy_config = _fake_vllm_omni(tmp_path)
    snapshots: list[dict[str, object]] = []

    def snapshot_download(**arguments: object) -> str:
        snapshots.append(arguments)
        return str(checkpoint)

    started = _Started()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    assert model.load_model(QWEN_DEVICE, tmp_path, start=started) is started.server

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
    ((command, socket_path),) = started.calls
    assert command[1:3] == ["serve", str(checkpoint)]
    assert command[command.index("--uds") + 1] == str(socket_path)
    assert command[command.index("--deploy-config") + 1] == str(deploy_config)
    assert "--trust-remote-code" not in command
    assert "vllm_omni" not in sys.modules


def test_each_load_listens_on_its_own_socket(
    model: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_vllm_omni(tmp_path)
    started = _load(model, tmp_path, monkeypatch)

    model.load_model(QWEN_DEVICE, tmp_path, start=started)

    sockets = [socket_path for _command, socket_path in started.calls]
    assert len(sockets) == len(set(sockets)) > 1


def test_the_cards_reservation_leaves_room_for_another_user(
    model: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_vllm_omni(tmp_path)
    ((command, _socket_path),) = _load(model, tmp_path, monkeypatch).calls

    stages = json.loads(command[command.index("--stage-overrides") + 1])

    reserved = sum(stage["gpu_memory_utilization"] for stage in stages.values())
    assert 0 < reserved < 1


def test_load_without_vllm_omni_installed_fails(
    model: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **_: str(tmp_path))
    monkeypatch.setitem(sys.modules, "vllm_omni", None)

    with pytest.raises(ModuleNotFoundError):
        model.load_model(QWEN_DEVICE, tmp_path)


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps"])
def test_load_rejects_unsupported_devices_before_the_hub_import(
    model: ModuleType, tmp_path: Path, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)

    with pytest.raises(ValueError, match="unsupported Qwen device"):
        model.load_model(device, tmp_path)

    assert "huggingface_hub" not in sys.modules


def _spoken(
    model: ModuleType, speech_server: ScriptedSpeechServer, **named: object
) -> Iterator[bytes]:
    return model.pcm_chunks(
        speech_server.server, "Hallo", "de", speaker="Ryan", **named
    )


def test_a_chunk_reaches_the_caller_before_the_next_arrives(
    model: ModuleType, speech_server: ScriptedSpeechServer
) -> None:
    second_may_follow = threading.Event()
    speech_server.parts = [b"\x01\x02", second_may_follow, b"\x03\x04"]
    frames = _spoken(model, speech_server)

    assert next(frames) == b"\x01\x02"
    assert speech_server.sent == 1

    second_may_follow.set()
    assert list(frames) == [b"\x03\x04"]


def test_an_odd_trailing_byte_joins_the_next_frame(
    model: ModuleType, speech_server: ScriptedSpeechServer
) -> None:
    speech_server.parts = [b"\x01", b"\x02\x03", b"\x04\x05\x06\x07", b"\x08"]

    frames = list(_spoken(model, speech_server))

    assert b"".join(frames) == bytes(range(1, 9))
    assert all(len(frame) % 2 == 0 for frame in frames)


def test_an_odd_remainder_at_the_end_fails_after_its_whole_samples(
    model: ModuleType, speech_server: ScriptedSpeechServer
) -> None:
    speech_server.parts = [b"\x01\x02\x03"]
    frames = _spoken(model, speech_server)

    assert next(frames) == b"\x01\x02"
    with pytest.raises(model.SpeechRequestError):
        next(frames)


def test_a_chunk_larger_than_a_frame_is_split_into_bounded_frames(
    model: ModuleType, speech_server: ScriptedSpeechServer
) -> None:
    audio = bytes(range(256)) * ((2 * MAX_PCM_BYTES + 2) // 256 + 1)
    audio = audio[: 2 * MAX_PCM_BYTES + 2]
    speech_server.parts = [audio]

    frames = list(_spoken(model, speech_server))

    assert b"".join(frames) == audio
    assert len(frames) >= len(audio) // MAX_PCM_BYTES
    assert all(0 < len(frame) <= MAX_PCM_BYTES for frame in frames)


@pytest.mark.parametrize(
    ("language", "language_name"), [("de", "German"), ("en", "English")]
)
def test_the_request_names_the_voice_and_language(
    model: ModuleType,
    speech_server: ScriptedSpeechServer,
    language: str,
    language_name: str,
) -> None:
    speech_server.parts = [b"\x01\x02"]

    assert list(
        model.pcm_chunks(speech_server.server, "Hallo", language, speaker="Sohee")
    )

    (request,) = speech_server.requests
    assert request["input"] == "Hallo"
    assert request["voice"] == "Sohee"
    assert request["language"] == language_name
    assert "instructions" not in request


def test_named_instructions_reach_the_server_as_instructions_for_that_utterance(
    model: ModuleType, speech_server: ScriptedSpeechServer
) -> None:
    speech_server.parts = [b"\x01\x02"]

    assert list(_spoken(model, speech_server, instructions="Sprich zornig"))
    assert list(_spoken(model, speech_server))

    spoken_with, spoken_without = speech_server.requests
    assert spoken_with["instructions"] == "Sprich zornig"
    assert "instructions" not in spoken_without


def test_an_unsupported_language_is_refused_before_any_request(
    model: ModuleType, speech_server: ScriptedSpeechServer
) -> None:
    with pytest.raises(ValueError, match="unsupported Qwen language"):
        list(model.pcm_chunks(speech_server.server, "Bonjour", "fr", speaker="Ryan"))

    assert speech_server.requests == []


def test_a_speaker_outside_the_closed_set_is_refused(
    model: ModuleType, speech_server: ScriptedSpeechServer
) -> None:
    with pytest.raises(ValueError, match="not a valid QwenSpeaker"):
        list(model.pcm_chunks(speech_server.server, "Hallo", "de", speaker="Nobody"))

    assert speech_server.requests == []


def test_a_refused_request_fails(
    model: ModuleType, speech_server: ScriptedSpeechServer
) -> None:
    speech_server.status = 400

    with pytest.raises(model.SpeechRequestError):
        list(_spoken(model, speech_server))


def test_closing_the_stream_early_hangs_up_on_the_server(
    model: ModuleType, speech_server: ScriptedSpeechServer
) -> None:
    speech_server.parts = [b"\x01\x02", HANG_UP]
    frames = _spoken(model, speech_server)
    assert next(frames) == b"\x01\x02"

    frames.close()

    assert speech_server.hung_up.wait(5)
