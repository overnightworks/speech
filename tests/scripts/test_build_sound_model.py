"""The sound-model build switches onnxruntime telemetry off before it loads."""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.machinery
import io
import os
import sys
import types
import wave
from typing import TYPE_CHECKING

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


@pytest.fixture
def faked_build(
    script: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> types.ModuleType:
    """The script with its download and conversion faked, parity left open."""

    def fetch_source(source: Path) -> None:
        source.mkdir()
        (source / "LICENSE").write_bytes(b"licence")

    monkeypatch.setattr(script, "fetch_source", fetch_source)
    monkeypatch.setattr(script, "class_names", lambda _source: ["laughter"])
    monkeypatch.setattr(script, "pytorch_model", lambda _source: object())
    monkeypatch.setattr(script, "convert", lambda _model, _onnx: None)
    monkeypatch.setattr(script, "worst_difference", lambda _model, _onnx, _labels: 0.0)
    return script


def test_build_writes_the_model_directory_when_the_parity_holds(
    faked_build: types.ModuleType, tmp_path: Path
) -> None:
    destination = tmp_path / "model"
    faked_build.build(destination)
    assert (destination / "labels.json").is_file()


def test_build_refuses_an_onnx_that_differs_from_pytorch(
    faked_build: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        faked_build,
        "worst_difference",
        lambda _model, _onnx, _labels: faked_build.TOLERANCE * 2,
    )
    destination = tmp_path / "model"
    with pytest.raises(RuntimeError, match="differs from PyTorch"):
        faked_build.build(destination)
    assert not destination.exists()


def test_build_refuses_an_existing_destination(
    faked_build: types.ModuleType, tmp_path: Path
) -> None:
    destination = tmp_path / "model"
    destination.mkdir()
    with pytest.raises(RuntimeError, match="exists"):
        faked_build.build(destination)
    assert list(destination.iterdir()) == []


def test_build_refuses_a_download_that_differs_from_the_pin(
    script: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        script.urllib.request,
        "urlopen",
        lambda *_a, **_k: contextlib.nullcontext(io.BytesIO(b"tampered")),
    )
    destination = tmp_path / "model"
    with pytest.raises(RuntimeError, match="is not the pinned"):
        script.build(destination)
    assert not destination.exists()


def test_reference_windows_refuse_a_clip_that_is_not_the_pinned_format(
    script: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = tmp_path / "clip.wav"
    with wave.open(str(clip), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(script.SAMPLE_RATE // 2)
        writer.writeframes(b"\0\0" * 8)
    monkeypatch.setattr(script, "REFERENCE_CLIP", clip)
    with pytest.raises(RuntimeError, match="is not 16 kHz mono PCM16"):
        script.reference_windows()
