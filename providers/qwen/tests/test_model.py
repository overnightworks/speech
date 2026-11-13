"""The Qwen adapter stays local, closed, and bounds completed PCM."""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from speech_provider_contract import (
    MAX_PCM_BYTES,
    QWEN_DEVICE,
    QWEN_MODEL_ID,
    QWEN_REVISION,
)

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


@pytest.fixture
def model() -> ModuleType:
    from speech_qwen import model as module

    return module


class _GeneratedModel:
    def __init__(self, waveform: np.ndarray, rate: int = 24_000) -> None:
        self._waveform = waveform
        self._rate = rate
        self.calls: list[dict[str, object]] = []

    def generate_custom_voice(
        self, **arguments: object
    ) -> tuple[list[np.ndarray], int]:
        self.calls.append(arguments)
        return [self._waveform], self._rate


def test_load_uses_the_pinned_local_snapshot_and_closed_cuda_mapping(
    model: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}
    checkpoint = tmp_path / "snapshot"
    loaded = object()

    def snapshot_download(**arguments: object) -> str:
        observed["snapshot"] = arguments
        return str(checkpoint)

    class FakeQwen:
        @classmethod
        def from_pretrained(cls, path: str, **arguments: object) -> object:
            observed["model"] = (path, arguments)
            return loaded

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setitem(
        sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=FakeQwen)
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bfloat16"))

    assert model.load_model(QWEN_DEVICE, tmp_path) is loaded
    assert observed == {
        "snapshot": {
            "repo_id": QWEN_MODEL_ID,
            "repo_type": "model",
            "revision": QWEN_REVISION,
            "allow_patterns": model.QWEN_ARTIFACTS,
            "local_files_only": True,
            "cache_dir": tmp_path,
        },
        "model": (
            str(checkpoint),
            {
                "device_map": "cuda:0",
                "dtype": "bfloat16",
                "local_files_only": True,
            },
        ),
    }


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps"])
def test_load_rejects_unsupported_devices_before_sdk_import(
    model: ModuleType, tmp_path: Path, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "qwen_tts", raising=False)

    with pytest.raises(ValueError, match="unsupported Qwen device"):
        model.load_model(device, tmp_path)

    assert "qwen_tts" not in sys.modules


def test_complete_waveform_is_clipped_and_split_into_even_bounded_pcm(
    model: ModuleType,
) -> None:
    waveform = np.linspace(-2.0, 2.0, (MAX_PCM_BYTES // 2) + 17)
    generated = _GeneratedModel(waveform)

    chunks = list(model.pcm_chunks(generated, "Hallo", "de", speaker="Ryan"))

    assert generated.calls == [
        {"text": "Hallo", "language": "German", "speaker": "Ryan"}
    ]
    assert chunks
    assert all(len(chunk) <= MAX_PCM_BYTES and len(chunk) % 2 == 0 for chunk in chunks)
    expected = np.clip(waveform * 32767.0, -32768, 32767).astype("<i2").tobytes()
    assert b"".join(chunks) == expected


def test_english_is_explicit_and_instruction_is_absent(model: ModuleType) -> None:
    generated = _GeneratedModel(np.array([0.25]))

    assert list(model.pcm_chunks(generated, "Hello", "en", speaker="Sohee"))

    assert generated.calls == [
        {"text": "Hello", "language": "English", "speaker": "Sohee"}
    ]


def test_named_instructions_reach_the_model_as_instruct_for_that_utterance(
    model: ModuleType,
) -> None:
    generated = _GeneratedModel(np.array([0.25]))

    assert list(
        model.pcm_chunks(generated, "Hallo", "de", "Sprich zornig", speaker="Ryan")
    )
    assert list(model.pcm_chunks(generated, "Hallo", "de", speaker="Ryan"))

    assert generated.calls == [
        {
            "text": "Hallo",
            "language": "German",
            "speaker": "Ryan",
            "instruct": "Sprich zornig",
        },
        {"text": "Hallo", "language": "German", "speaker": "Ryan"},
    ]


def test_generation_rejects_an_unexpected_sample_rate(model: ModuleType) -> None:
    generated = _GeneratedModel(np.array([0.25]), rate=16_000)

    with pytest.raises(ValueError, match="Qwen sample rate"):
        list(model.pcm_chunks(generated, "Hallo", "de", speaker="Ryan"))


def test_the_pinned_model_keeps_the_instructions_the_sdk_receives() -> None:
    import torch
    from qwen_tts import Qwen3TTSModel

    parameter_count = re.search(r"-(\d+)\.(\d+)B-", QWEN_MODEL_ID)
    assert parameter_count is not None
    model_size = f"{parameter_count[1]}b{parameter_count[2]}"
    received: dict[str, object] = {}

    def generate(**arguments: object) -> tuple[list[object], None]:
        received.update(arguments)
        return [object()], None

    sdk_model = SimpleNamespace(
        tts_model_type="custom_voice",
        tts_model_size=model_size,
        device="cpu",
        generate=generate,
        speech_tokenizer=SimpleNamespace(
            decode=lambda _codes: ([np.array([0.25])], 24_000)
        ),
    )
    speaking = Qwen3TTSModel(
        sdk_model, lambda **_text: {"input_ids": torch.tensor([1, 2])}
    )

    speaking.generate_custom_voice(
        text="Hallo", language="German", speaker="Ryan", instruct="Sprich zornig"
    )

    assert received["instruct_ids"][0] is not None
