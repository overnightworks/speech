"""The bundled CUDA library lookup finds regular and namespace NVIDIA wheels."""

from __future__ import annotations

import importlib
import os
import sys
from typing import TYPE_CHECKING

import pytest

from speech_service.cuda_libs import cuda_library_dirs, prepare_cuda_libraries

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import ModuleType


def _nvidia_modules() -> dict[str, ModuleType]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name.split(".")[0] == "nvidia"
    }


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An empty import path that no installed nvidia package can shadow."""
    directory = tmp_path / "site"
    directory.mkdir()
    original_modules = _nvidia_modules()
    for name in original_modules:
        del sys.modules[name]
    monkeypatch.setattr(sys, "path", [str(directory)])
    importlib.invalidate_caches()
    yield directory
    for name in _nvidia_modules():
        del sys.modules[name]
    sys.modules.update(original_modules)
    importlib.invalidate_caches()


@pytest.mark.parametrize("init_files", [False, True], ids=["namespace", "regular"])
def test_finds_the_cublas_and_cudnn_library_directories(
    site: Path, *, init_files: bool
) -> None:
    expected = []
    for package in ("cublas", "cudnn"):
        library = site / "nvidia" / package / "lib"
        library.mkdir(parents=True)
        expected.append(library.resolve())
        if init_files:
            (site / "nvidia" / "__init__.py").touch()
            (site / "nvidia" / package / "__init__.py").touch()

    assert cuda_library_dirs() == expected


@pytest.mark.parametrize(
    "package_present",
    [False, True],
    ids=["no_package", "package_without_lib_directory"],
)
def test_a_package_without_a_library_directory_yields_none(
    site: Path, *, package_present: bool
) -> None:
    """A package without a lib directory yields none, as a missing package does."""
    if package_present:
        (site / "nvidia" / "cublas").mkdir(parents=True)

    assert cuda_library_dirs() == []


@pytest.mark.parametrize(
    ("before", "suffix"),
    [(None, ""), ("/already/there", ":/already/there")],
    ids=["unset_path", "existing_path_follows"],
)
def test_prepare_puts_the_library_directories_first_on_the_loader_path(
    site: Path,
    monkeypatch: pytest.MonkeyPatch,
    before: str | None,
    suffix: str,
) -> None:
    library = site / "nvidia" / "cublas" / "lib"
    library.mkdir(parents=True)
    if before is None:
        monkeypatch.setenv("LD_LIBRARY_PATH", "")
        monkeypatch.delenv("LD_LIBRARY_PATH")
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH", before)

    prepare_cuda_libraries()

    assert os.environ["LD_LIBRARY_PATH"] == f"{library.resolve()}{suffix}"


def test_prepare_skips_a_library_the_loader_cannot_open(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = site / "nvidia" / "cudnn" / "lib"
    library.mkdir(parents=True)
    (library / "libcudnn.so.9").write_bytes(b"not an ELF file")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")

    prepare_cuda_libraries()

    assert os.environ["LD_LIBRARY_PATH"] == str(library.resolve())


@pytest.mark.usefixtures("site")
def test_prepare_leaves_the_loader_path_alone_without_bundled_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/already/there")

    prepare_cuda_libraries()

    assert os.environ["LD_LIBRARY_PATH"] == "/already/there"
