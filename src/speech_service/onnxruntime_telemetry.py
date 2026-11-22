"""The process's onnxruntime telemetry switch."""

import os
from typing import Final

ONNXRUNTIME_TELEMETRY_SWITCH: Final = "ORT_DISABLE_TELEMETRY"


def disable_onnxruntime_telemetry() -> None:
    """Switch telemetry off unless the environment names a value."""
    # onnxruntime reads the switch once, at import, so it is set before any import.
    os.environ.setdefault(ONNXRUNTIME_TELEMETRY_SWITCH, "1")
