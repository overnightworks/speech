"""The vLLM engine the model loads installs one OpenCV build, the headless one."""

from __future__ import annotations

import tomllib
from pathlib import Path

_OPENCV_PREFIX = "opencv-python"
_QWEN_PROJECT = Path(__file__).resolve().parents[1]


def test_the_qwen_lock_installs_only_the_headless_opencv_build() -> None:
    lock = tomllib.loads((_QWEN_PROJECT / "uv.lock").read_text())
    builds = {
        package["name"]
        for package in lock["package"]
        if package["name"].startswith(_OPENCV_PREFIX)
    }

    assert builds == {"opencv-python-headless"}
