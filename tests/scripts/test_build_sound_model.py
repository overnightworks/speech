"""The sound-model build switches telemetry off, refuses bad inputs and builds."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import types
import wave
from typing import TYPE_CHECKING

import numpy as np
import pytest

from speech_service import pcm, sound_events
from tests.support.script_support import script_module

if TYPE_CHECKING:
    from pathlib import Path


class _EnvironmentAtImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Stands in for onnxruntime and records the variable when it is imported."""

    def __init__(self) -> None:
        self.seen: str | None = None

    def find_spec(
        self, name: str, _path: object, _target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        if name != "onnxruntime":
            return None
        self.seen = os.environ.get("ORT_DISABLE_TELEMETRY")
        return importlib.machinery.ModuleSpec(name, self)

    def create_module(
        self, _spec: importlib.machinery.ModuleSpec
    ) -> types.ModuleType | None:
        return None

    def exec_module(self, _module: types.ModuleType) -> None:
        return None


@pytest.mark.parametrize(
    ("before", "expected"),
    [(None, "1"), ("0", "0")],
    ids=["unset_is_switched_off", "operator_value_stays"],
)
def test_build_script_sets_telemetry_off_before_importing_onnxruntime(
    monkeypatch: pytest.MonkeyPatch, before: str | None, expected: str
) -> None:
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", before or "restored")
    if before is None:
        monkeypatch.delenv("ORT_DISABLE_TELEMETRY")
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    monkeypatch.delitem(sys.modules, "onnxruntime")
    finder = _EnvironmentAtImport()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    script_module("build_sound_model")
    assert finder.seen == expected


def test_build_script_windows_match_the_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "restored")
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY")
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    script = script_module("build_sound_model")
    assert script.SAMPLE_RATE == pcm.HEARING_SAMPLE_RATE
    assert script.WINDOW_SAMPLES == sound_events.WINDOW_SAMPLES
    assert script.HOP_SAMPLES == sound_events.HOP_SAMPLES


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    return script_module("build_sound_model")


_CLASSES = ["laughter", "cough"]


@pytest.fixture
def upstream(
    script: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """A small upstream tree served over file://, its bytes pinned by their hashes."""
    root = tmp_path / "upstream"
    served = {path: path.encode() for path in script.SOURCE_SHA256}
    served["LICENSE"] = b"licence"
    served["data_util/audioset_classes.py"] = (
        f"as_strong_train_classes = {_CLASSES!r}\n".encode()
    )
    for path, content in served.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(content)
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    pins = {
        path: hashlib.sha256(content).hexdigest() for path, content in served.items()
    }
    monkeypatch.setattr(script, "SOURCE_URL", f"{root.as_uri()}/{{path}}")
    monkeypatch.setattr(script, "SOURCE_SHA256", pins)
    monkeypatch.setattr(script, "CHECKPOINT_URL", checkpoint.as_uri())
    monkeypatch.setattr(
        script, "CHECKPOINT_SHA256", hashlib.sha256(b"weights").hexdigest()
    )
    return root


@pytest.fixture
def faked_build(
    script: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> types.ModuleType:
    """The script with its torch half faked, parity left open."""

    def convert(_model: object, onnx: Path) -> None:
        onnx.write_bytes(b"onnx")
        onnx.with_name(f"{onnx.name}.data").write_bytes(b"weights")

    monkeypatch.setattr(script, "pytorch_model", lambda _source: object())
    monkeypatch.setattr(script, "convert", convert)
    monkeypatch.setattr(
        script, "worst_difference", lambda _model, _onnx, _windows, _labels: 0.0
    )
    return script


@pytest.fixture
def destination(tmp_path: Path) -> Path:
    """The model directory to build, inside an empty `models` directory."""
    models = tmp_path / "models"
    models.mkdir()
    return models / "model"


def test_build_writes_the_model_directory_when_the_parity_holds(
    faked_build: types.ModuleType, upstream: Path, destination: Path
) -> None:
    faked_build.build(destination)
    sound_events.check_model_directory(destination)
    assert json.loads((destination / "labels.json").read_text()) == _CLASSES
    assert (destination / "LICENSE").read_bytes() == (upstream / "LICENSE").read_bytes()


@pytest.mark.usefixtures("upstream")
def test_build_refuses_an_onnx_that_differs_from_pytorch(
    faked_build: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    destination: Path,
) -> None:
    monkeypatch.setattr(
        faked_build,
        "worst_difference",
        lambda _model, _onnx, _windows, _labels: faked_build.TOLERANCE * 2,
    )
    with pytest.raises(RuntimeError, match="differs from PyTorch"):
        faked_build.build(destination)
    assert list(destination.parent.iterdir()) == []


def test_build_refuses_an_existing_destination(
    faked_build: types.ModuleType, destination: Path
) -> None:
    destination.mkdir()
    with pytest.raises(RuntimeError, match="exists"):
        faked_build.build(destination)
    assert list(destination.parent.iterdir()) == [destination]
    assert list(destination.iterdir()) == []


def test_build_refuses_a_checkpoint_that_differs_from_its_pin(
    faked_build: types.ModuleType, upstream: Path, destination: Path
) -> None:
    (upstream / "checkpoint.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="is not the pinned"):
        faked_build.build(destination)
    assert list(destination.parent.iterdir()) == []


@pytest.mark.usefixtures("upstream")
def test_build_refuses_a_clip_that_is_not_16_khz_mono_pcm16(
    faked_build: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clip = tmp_path / "clip.wav"
    with wave.open(str(clip), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(faked_build.SAMPLE_RATE // 2)
        writer.writeframes(b"\0\0" * 8)
    monkeypatch.setattr(faked_build, "REFERENCE_CLIP", clip)
    absent = tmp_path / "absent" / "model"
    with pytest.raises(RuntimeError, match="is not 16 kHz mono PCM16"):
        faked_build.build(absent)
    assert not absent.parent.exists()


def test_the_clips_windows_are_the_services_windows_of_the_clip(
    script: types.ModuleType,
) -> None:
    stream = sound_events.SoundStream({})
    with wave.open(str(script.REFERENCE_CLIP), "rb") as reader:
        pcm_bytes = reader.readframes(reader.getnframes())
    hop_bytes = sound_events.HOP_SAMPLES * 2
    served = []
    for start in range(0, len(pcm_bytes), hop_bytes):
        stream.add(pcm_bytes[start : start + hop_bytes])
        window = stream.next_window()
        if window is not None:
            served.append(window.samples[0])
            stream.scored(window, np.zeros((1, 1, 1), dtype=np.float32))
    windows = script.reference_windows()[script.NOISE_WINDOWS :]
    assert len(windows) == len(served) > 0
    assert all(
        np.array_equal(found, expected)
        for found, expected in zip(windows, served, strict=True)
    )


def _scores(*values: float, shape: tuple[int, ...] = (1, 2)) -> np.ndarray:
    return np.array(values, dtype=np.float32).reshape(shape)


@pytest.mark.parametrize(
    ("found", "expected", "message"),
    [
        (_scores(0, 0), _scores(0, 0, 0, shape=(1, 3)), "the ONNX scores are"),
        (
            _scores(0, 0, 0, shape=(1, 3)),
            _scores(0, 0, 0, shape=(1, 3)),
            "the ONNX scores are",
        ),
        (_scores(0, np.nan), _scores(0, 0), "non-finite"),
        (_scores(0, np.inf), _scores(0, 0), "non-finite"),
    ],
    ids=[
        "scores_of_another_shape_than_pytorchs_are_refused",
        "scores_for_another_number_of_labels_are_refused",
        "a_non_finite_score_is_refused",
        "an_infinite_score_is_refused",
    ],
)
def test_score_difference_refuses_unsound_scores(
    script: types.ModuleType, found: np.ndarray, expected: np.ndarray, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        script.score_difference(found, expected, 2)


@pytest.mark.parametrize(
    ("found", "expected", "gap"),
    [
        (_scores(0.5, 0.25), _scores(0.25, 0.5), 0.25),
        (_scores(0.5, 0.75), _scores(0.5, 0.25), 0.5),
    ],
    ids=["the_gap_is_absolute", "the_gap_is_the_largest_column"],
)
def test_score_difference_is_the_largest_gap(
    script: types.ModuleType, found: np.ndarray, expected: np.ndarray, gap: float
) -> None:
    assert script.score_difference(found, expected, 2) == pytest.approx(gap)
