"""The generator keeps the browser package's hop module in step with its owners."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from generate_browser_wire import OUTPUT, main

if TYPE_CHECKING:
    from pathlib import Path


def _written(root: Path) -> dict[Path, str]:
    return {path: path.read_text() for path in root.rglob("*") if path.is_file()}


def test_check_on_the_committed_module_exits_clean() -> None:
    assert main(["--check"]) == 0


def test_a_write_without_check_writes_the_module_under_the_root(tmp_path: Path) -> None:
    assert main([], tmp_path) == 0
    assert list(_written(tmp_path)) == [tmp_path / OUTPUT]


def test_check_on_an_in_sync_module_exits_clean_and_writes_nothing(
    tmp_path: Path,
) -> None:
    main([], tmp_path)
    before = _written(tmp_path)

    assert main(["--check"], tmp_path) == 0
    assert _written(tmp_path) == before


def test_check_on_a_drifted_module_exits_with_out_of_date(tmp_path: Path) -> None:
    main([], tmp_path)
    module = tmp_path / OUTPUT
    module.write_text(module.read_text() + "// hand edit\n")

    with pytest.raises(SystemExit, match="out of date"):
        main(["--check"], tmp_path)


def test_check_with_the_module_missing_exits_and_creates_nothing(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="out of date"):
        main(["--check"], tmp_path)

    assert list(tmp_path.iterdir()) == []
