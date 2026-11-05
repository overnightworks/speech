"""Report, as JSON, the GPU voice's running workers in the service container."""

# ruff: noqa: T201 - this script's output is the report the proof reads

import json
import sys
from pathlib import Path

from speech_runtime_service.config import load_settings
from speech_runtime_service.provider_process import ProviderId, provider_entrypoint

entrypoint = provider_entrypoint(load_settings().provider_root, ProviderId.QWEN)
if entrypoint is None:
    sys.exit("SPEECH_PROVIDER_ROOT is not set in the service container")


def runs_the_worker(process: Path) -> bool:
    """Whether the process `process` names in /proc runs the GPU voice's worker."""
    try:
        return str(entrypoint).encode() in (process / "cmdline").read_bytes()
    except OSError:
        return False


print(
    json.dumps(
        {
            "workers": sorted(
                int(process.name)
                for process in Path("/proc").iterdir()
                if process.name.isdigit() and runs_the_worker(process)
            )
        }
    )
)
