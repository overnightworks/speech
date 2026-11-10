"""The release check names the version fields that are off, package.json included."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from check_release import REPOSITORY_ROOT, main, release_faults, release_rows

RELEASE = "3.4.5"
OTHER = "9.9.9"


def _write_project(root: Path, directory: str, name: str, version: str) -> None:
    project = root / directory
    project.mkdir(parents=True, exist_ok=True)
    package = name.replace("-", "_")
    (project / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "{version}"\n\n'
        f'[tool.hatch.build.targets.wheel]\npackages = ["src/{package}"]\n'
    )


def _write_package(root: Path, directory: str, version: str) -> None:
    package = root / directory
    package.mkdir(parents=True, exist_ok=True)
    (package / "package.json").write_text(json.dumps({"version": version}))


def _write_root(
    root: Path,
    version: str,
    *,
    members: tuple[str, ...] = (),
    sources: tuple[str, ...] = (),
) -> None:
    _write_project(root, ".", "root-service", version)
    declared = (root / "pyproject.toml").read_text()
    declared += f"\n[tool.uv.workspace]\nmembers = {list(members)!r}\n".replace(
        "'", '"'
    )
    declared += "\n[tool.uv.sources]\n"
    declared += "".join(
        f'{path.replace("/", "-")} = {{ path = "{path}", editable = true }}\n'
        for path in sources
    )
    (root / "pyproject.toml").write_text(declared)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    _write_root(
        tmp_path,
        RELEASE,
        members=("libs/client",),
        sources=("libs/contract",),
    )
    _write_project(tmp_path, "libs/client", "the-client", RELEASE)
    _write_project(tmp_path, "libs/contract", "the-contract", RELEASE)
    _write_package(tmp_path, "packages/web", RELEASE)
    return tmp_path


def test_the_distributions_of_this_repository_stand_in_lockstep() -> None:
    assert release_faults(REPOSITORY_ROOT, None) == []


def test_a_wrong_input_fails_the_command_for_this_repository(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([OTHER]) == 1
    assert "pyproject.toml" in capsys.readouterr().out


def test_the_rows_of_this_repository_name_a_package_that_exists() -> None:
    rows = release_rows(REPOSITORY_ROOT)

    assert len(rows) > 1
    for row in rows:
        assert (REPOSITORY_ROOT / row.directory / "src" / row.package).is_dir()


def test_the_rows_print_for_the_workflow(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--rows"]) == 0

    printed = [line.split("|") for line in capsys.readouterr().out.splitlines()]
    assert printed == [
        [row.name, row.directory, row.package] for row in release_rows(REPOSITORY_ROOT)
    ]


def test_a_tree_in_lockstep_passes_for_its_own_version(tree: Path) -> None:
    assert release_faults(tree, RELEASE) == []
    assert release_faults(tree, None) == []


def test_the_rows_hold_the_root_and_every_declared_project_once(tree: Path) -> None:
    assert {(row.name, row.directory, row.package) for row in release_rows(tree)} == {
        ("root-service", ".", "root_service"),
        ("the-client", "libs/client", "the_client"),
        ("the-contract", "libs/contract", "the_contract"),
    }


def test_a_project_declared_both_ways_is_one_row(tree: Path) -> None:
    _write_root(
        tree,
        RELEASE,
        members=("libs/client",),
        sources=("libs/client", "libs/contract"),
    )

    names = [row.name for row in release_rows(tree)]
    assert sorted(names) == sorted(set(names))
    assert "the-client" in names


@pytest.mark.parametrize("off", [".", "libs/client", "libs/contract", "packages/web"])
def test_a_field_that_is_off_is_named(tree: Path, off: str) -> None:
    if off == ".":
        _write_root(tree, OTHER, members=("libs/client",), sources=("libs/contract",))
    elif off.startswith("packages/"):
        _write_package(tree, off, OTHER)
    else:
        _write_project(tree, off, Path(off).name, OTHER)

    project_file = (
        f"{off}/package.json"
        if off.startswith("packages/")
        else ("pyproject.toml" if off == "." else f"{off}/pyproject.toml")
    )
    for expected in (None, RELEASE):
        faults = release_faults(tree, expected)
        assert any(fault.startswith(f"{project_file} ") for fault in faults)


def test_a_wrong_input_names_every_field(tree: Path) -> None:
    faults = release_faults(tree, OTHER)

    assert all(
        any(fault.startswith(f"{project} ") for fault in faults)
        for project in (
            "pyproject.toml",
            "libs/client/pyproject.toml",
            "libs/contract/pyproject.toml",
            "packages/web/package.json",
        )
    )


@pytest.mark.parametrize("declaration", ["workspace member", "path source"])
def test_a_further_project_at_another_version_is_named_and_listed(
    tree: Path, declaration: str
) -> None:
    members = (
        ("libs/client", "libs/extra")
        if declaration == "workspace member"
        else ("libs/client",)
    )
    sources = (
        ("libs/contract",)
        if declaration == "workspace member"
        else ("libs/contract", "libs/extra")
    )
    _write_root(tree, RELEASE, members=members, sources=sources)
    _write_project(tree, "libs/extra", "the-extra", OTHER)

    assert any(
        "libs/extra/pyproject.toml" in fault for fault in release_faults(tree, None)
    )
    assert any(
        "libs/extra/pyproject.toml" in fault for fault in release_faults(tree, RELEASE)
    )
    assert "the-extra" in {row.name for row in release_rows(tree)}


def test_a_package_json_without_a_version_fails_loud(tree: Path) -> None:
    (tree / "packages/web/package.json").write_text("{}")

    with pytest.raises(KeyError):
        release_faults(tree, None)
