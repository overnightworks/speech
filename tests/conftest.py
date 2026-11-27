"""The environment every test starts in: telemetry off.

A failed player browser cell leaves the record of what its page showed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from speech_service.onnxruntime_telemetry import disable_onnxruntime_telemetry

if TYPE_CHECKING:
    from collections.abc import Generator

disable_onnxruntime_telemetry()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """A player cell that failed leaves the record of what its page showed."""
    report = yield
    if call.when == "call" and item.get_closest_marker("browser"):
        from tests.support.browser_player_support import take_recording  # noqa: PLC0415 - only a browser cell needs the stage

        recording = take_recording()
        if recording is not None and report.failed:
            recording.write(item.nodeid)
    return report
