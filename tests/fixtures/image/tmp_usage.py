"""Report the filesystem's own count of /tmp, as JSON: bytes in use and its size."""

# ruff: noqa: T201 - this script's output is the report the proof reads

import json
import shutil

TMP = "/tmp"  # noqa: S108 - the service's own tmpfs which the proof inspects

usage = shutil.disk_usage(TMP)
print(json.dumps({"used_bytes": usage.used, "size_bytes": usage.total}))
