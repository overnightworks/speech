"""Report, as JSON, the GPU voice's workers in the service container.

`workers` are the pids of its running workers, `directories` the private
directories of provider workers still in the temporary directory.
"""

# ruff: noqa: T201 - this script's output is the report the proof reads

import json
import sys
import tempfile
from pathlib import Path

from speech_runtime_service.config import load_settings
from speech_runtime_service.provider_process import (
    WORKER_DIRECTORY_PREFIX,
    ProviderId,
    provider_entrypoint,
)

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
            ),
            "directories": sorted(
                directory.name
                for directory in Path(tempfile.gettempdir()).iterdir()
                if directory.name.startswith(WORKER_DIRECTORY_PREFIX)
            ),
        }
    )
)
