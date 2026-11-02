"""Check that the three distributions stand at one version, and at the released one.

The release workflow runs this before it creates the tag. Without a version it
checks only that the three agree, which the default test run does on every pull
request.
"""

# ruff: noqa: T201 - the release workflow reads the faults from this command's output

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parent.parent
DISTRIBUTION_PROJECTS: Final = (
    "pyproject.toml",
    "packages/client/pyproject.toml",
    "providers/provider-contract/pyproject.toml",
)


def release_faults(root: Path, expected: str | None) -> list[str]:
    """One line per distribution whose version is off, each naming its project file."""
    versions = {
        project: tomllib.loads((root / project).read_text(encoding="utf-8"))["project"][
            "version"
        ]
        for project in DISTRIBUTION_PROJECTS
    }
    if expected is not None:
        return [
            f"{project} stands at {version}, the release is {expected}"
            for project, version in versions.items()
            if version != expected
        ]
    if len(set(versions.values())) > 1:
        return [
            f"{project} stands at {version}" for project, version in versions.items()
        ]
    return []


def main(argv: list[str]) -> int:
    """Print every fault and return 1, or return 0 when the versions hold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?", help="the version the release creates")
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    arguments = parser.parse_args(argv)
    faults = release_faults(arguments.root, arguments.version)
    for fault in faults:
        print(fault)
    return 1 if faults else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
