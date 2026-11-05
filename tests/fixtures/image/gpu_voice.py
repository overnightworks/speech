"""Report, as JSON, the GPU voice's workers in the service container.

With `probe` on its input, it also starts Qwen's interpreter as the service
starts its worker, in the same directory and environment, imports the model
library and asks torch whether a CUDA device is there.
"""

# ruff: noqa: T201 - this script's output is the report the proof reads

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from speech_runtime_service.config import load_settings
from speech_runtime_service.provider_process import (
    ProviderId,
    provider_directory,
    provider_entrypoint,
    provider_environment,
)

PROBE = (
    "import json, torch, qwen_tts, speech_runtime_qwen.model;"
    " print(json.dumps(torch.cuda.is_available()))"
)
PROBE_TIMEOUT_SECONDS = 300

entrypoint = provider_entrypoint(load_settings().provider_root, ProviderId.QWEN)
if entrypoint is None:
    sys.exit("SPEECH_PROVIDER_ROOT is not set in the service container")


def runs_the_worker(process: Path) -> bool:
    """Whether the process `process` names in /proc runs the GPU voice's worker."""
    try:
        return str(entrypoint).encode() in (process / "cmdline").read_bytes()
    except OSError:
        return False


report: dict[str, object] = {
    "workers": sorted(
        int(process.name)
        for process in Path("/proc").iterdir()
        if process.name.isdigit() and runs_the_worker(process)
    ),
    "cuda": None,
    "error": None,
}
if json.load(sys.stdin)["probe"]:
    directory = provider_directory(entrypoint)
    with tempfile.TemporaryDirectory() as scratch:
        probe = subprocess.run(  # noqa: S603, RUF100 - the image's Qwen interpreter runs a fixed probe
            [str(entrypoint.with_name("python")), "-c", PROBE],
            cwd=directory,
            env=provider_environment(directory, Path(scratch)),
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    if probe.returncode == 0:
        report["cuda"] = json.loads(probe.stdout.splitlines()[-1])
    else:
        report["error"] = (probe.stderr.strip().splitlines() or ["no output"])[-1]
print(json.dumps(report))
