"""The release check names the version fields that are off."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from check_release import DISTRIBUTION_PROJECTS, REPOSITORY_ROOT, main

if TYPE_CHECKING:
    from pathlib import Path

RELEASE = "3.4.5"


def _write_tree(root: Path, versions: dict[str, str]) -> None:
    for project, version in versions.items():
        path = root / project
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'[project]\nname = "x"\nversion = "{version}"\n')


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    _write_tree(tmp_path, dict.fromkeys(DISTRIBUTION_PROJECTS, RELEASE))
    return tmp_path


def test_the_three_distributions_of_this_repository_stand_in_lockstep() -> None:
    assert main(["--root", str(REPOSITORY_ROOT)]) == 0


def test_a_tree_in_lockstep_passes_for_its_own_version(tree: Path) -> None:
    assert main([RELEASE, "--root", str(tree)]) == 0


@pytest.mark.parametrize("off", DISTRIBUTION_PROJECTS)
def test_a_field_that_is_off_is_named(
    tree: Path, off: str, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_tree(tree, {off: "9.9.9"})

    assert main(["--root", str(tree)]) == 1
    assert off in capsys.readouterr().out

    assert main([RELEASE, "--root", str(tree)]) == 1
    printed = capsys.readouterr().out
    assert off in printed
    assert len(printed.splitlines()) == 1


def test_a_wrong_input_names_every_field(
    tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["9.9.9", "--root", str(tree)]) == 1

    printed = capsys.readouterr().out
    assert all(project in printed for project in DISTRIBUTION_PROJECTS)
