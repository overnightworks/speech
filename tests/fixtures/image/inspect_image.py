"""Report what the image holds, as JSON, for the proof to judge."""

# ruff: noqa: T201 - this script's output is the report the proof reads

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import sysconfig
from pathlib import Path

APPLICATION = Path.cwd()
VOICES = Path(sys.argv[1])
EXECUTABLES = sys.argv[2:]
SITE_PACKAGES = Path(sysconfig.get_paths()["purelib"])
PRIVATE_KEY_MARK = b"PRIVATE KEY"

files = sorted(path for path in APPLICATION.rglob("*") if path.is_file())
print(
    json.dumps(
        {
            "distributions": sorted(
                distribution.metadata["Name"]
                for distribution in importlib.metadata.distributions()
            ),
            "user": os.getuid(),
            "application_files": [str(path.relative_to(APPLICATION)) for path in files],
            "application_files_with_private_keys": [
                str(path.relative_to(APPLICATION))
                for path in files
                if PRIVATE_KEY_MARK in path.read_bytes()
            ],
            "voice_hashes": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(VOICES.iterdir())
            },
            "executables": {name: shutil.which(name) for name in EXECUTABLES},
            "nvidia_directories": sorted(
                path.name for path in SITE_PACKAGES.glob("nvidia*")
            ),
        }
    )
)
