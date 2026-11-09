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

from speech_service.hearing_models import (
    HearingModel,
    HearingModelMissingError,
    model_directory,
)

APPLICATION = Path.cwd()
VOICES = Path(sys.argv[1])
HEARING_CACHE = Path(sys.argv[2])
HEARING_MODEL = sys.argv[3]
EXECUTABLES = sys.argv[4:]
SITE_PACKAGES = Path(sysconfig.get_paths()["purelib"])
PRIVATE_KEY_MARK = b"PRIVATE KEY"

SOURCE_SUFFIXES = {".py", ".pyc"}


def hearing_model_directory() -> str | None:
    """Where the service finds the pinned hearing model offline, if it does."""
    try:
        return str(model_directory(HearingModel(HEARING_MODEL), HEARING_CACHE))
    except HearingModelMissingError:
        return None


files = sorted(path for path in APPLICATION.rglob("*") if path.is_file())
package_files = sorted(
    path
    for package in SITE_PACKAGES.glob("speech_*")
    if package.is_dir() and not package.name.endswith(".dist-info")
    for path in package.rglob("*")
    if path.is_file()
)
stray_package_files = [
    path
    for path in package_files
    if path.suffix not in SOURCE_SUFFIXES and path.name != "py.typed"
]
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
            "package_stray_files": [
                str(path.relative_to(SITE_PACKAGES)) for path in stray_package_files
            ],
            "package_files_with_private_keys": [
                str(path.relative_to(SITE_PACKAGES))
                for path in package_files
                if PRIVATE_KEY_MARK in path.read_bytes()
            ],
            "voice_hashes": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(VOICES.iterdir())
            },
            "hearing_model_directory": hearing_model_directory(),
            "executables": {name: shutil.which(name) for name in EXECUTABLES},
            "nvidia_directories": sorted(
                path.name for path in SITE_PACKAGES.glob("nvidia*")
            ),
        }
    )
)
