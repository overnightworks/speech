"""Report what /tmp holds, as JSON: the bytes of its files and its tmpfs size."""

# ruff: noqa: T201 - this script's output is the report the proof reads

import json
import shutil
from pathlib import Path

TMP = Path("/tmp")  # noqa: S108 - the service's own tmpfs, which the proof inspects

print(
    json.dumps(
        {
            "file_bytes": sum(
                path.stat().st_size for path in TMP.rglob("*") if path.is_file()
            ),
            "size_bytes": shutil.disk_usage(TMP).total,
        }
    )
)
