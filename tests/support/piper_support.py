"""A fake `piper` package whose voice loader a test names."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest


def install_fake_piper(
    monkeypatch: pytest.MonkeyPatch, load: Callable[..., object]
) -> None:
    """Replace `piper` for the test, with `PiperVoice.load` being `load`."""
    # Imported first so `from piper.config import ...` still finds the real submodule.
    importlib.import_module("piper.config")
    monkeypatch.setitem(
        sys.modules, "piper", SimpleNamespace(PiperVoice=SimpleNamespace(load=load))
    )
