"""The Qwen environment installs one OpenCV build, the headless one."""

from __future__ import annotations

import tomllib

from tests.support.browser_support import REPOSITORY

_OPENCV_PREFIX = "opencv-python"


def test_the_qwen_lock_installs_only_the_headless_opencv_build() -> None:
    lock = tomllib.loads((REPOSITORY / "providers/qwen/uv.lock").read_text())
    builds = {
        package["name"]
        for package in lock["package"]
        if package["name"].startswith(_OPENCV_PREFIX)
    }

    assert builds == {"opencv-python-headless"}
