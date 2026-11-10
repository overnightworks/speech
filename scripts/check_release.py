"""Check that the release's artefacts stand at one version, and at the released one.

The release set is the root project plus each local project the root declares in
`[tool.uv.workspace] members` and as a `path` in `[tool.uv.sources]`, and each
`packages/*/package.json`. The distributions' release workflow runs this before
it creates the tag, and reads its rows with `--rows`. Without a version the
check only requires that the set agrees, which the default test run does on
every pull request.
"""

# ruff: noqa: T201 - the distributions' release workflow reads this command's output

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DistributionRow:
    """One distribution: its name, its directory under the root, its import package."""

    name: str
    directory: str
    package: str


def _read_project(root: Path, directory: str) -> dict[str, Any]:
    return tomllib.loads(
        (root / directory / "pyproject.toml").read_text(encoding="utf-8")
    )


def _release_directories(root: Path) -> list[str]:
    """The root and the local projects it declares, each once, in declaration order."""
    declared = _read_project(root, ".").get("tool", {}).get("uv", {})
    candidates = [
        match.relative_to(root).as_posix()
        for member in declared.get("workspace", {}).get("members", [])
        for match in sorted(root.glob(member))
    ]
    candidates += [
        Path(source["path"]).as_posix()
        for source in declared.get("sources", {}).values()
        if "path" in source
    ]
    return list(dict.fromkeys([".", *candidates]))


def release_rows(root: Path) -> list[DistributionRow]:
    """One row per distribution of the release set."""
    rows = []
    for directory in _release_directories(root):
        project = _read_project(root, directory)
        wheel = project["tool"]["hatch"]["build"]["targets"]["wheel"]
        rows.append(
            DistributionRow(
                name=project["project"]["name"],
                directory=directory,
                package=Path(wheel["packages"][0]).name,
            )
        )
    return rows


def release_faults(root: Path, expected: str | None) -> list[str]:
    """One line per artefact whose version is off, each naming its version file."""
    versions = {
        (Path(directory) / "pyproject.toml").as_posix(): _read_project(root, directory)[
            "project"
        ]["version"]
        for directory in _release_directories(root)
    }
    versions |= {
        package_file.relative_to(root).as_posix(): json.loads(
            package_file.read_text(encoding="utf-8")
        )["version"]
        for package_file in sorted(root.glob("packages/*/package.json"))
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
    parser.add_argument(
        "--rows",
        action="store_true",
        help="print name|directory|package per distribution",
    )
    arguments = parser.parse_args(argv)
    if arguments.rows:
        for row in release_rows(REPOSITORY_ROOT):
            print(f"{row.name}|{row.directory}|{row.package}")
        return 0
    faults = release_faults(REPOSITORY_ROOT, arguments.version)
    for fault in faults:
        print(fault)
    return 1 if faults else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
