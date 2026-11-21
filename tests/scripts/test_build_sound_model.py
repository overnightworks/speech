"""The sound-model build switches onnxruntime telemetry off before it loads."""

import importlib.abc
import importlib.machinery
import os
import sys
from typing import TYPE_CHECKING

import pytest

from tests.support.script_support import script_module

if TYPE_CHECKING:
    import types


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
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    finder = _EnvironmentAtImport()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    script_module("build_sound_model")
    assert finder.seen == expected
