"""A provider process sees its own CUDA libraries and no parent secret."""

from __future__ import annotations

import os
import secrets
from typing import TYPE_CHECKING

import pytest

from speech_service.provider_process import provider_environment

if TYPE_CHECKING:
    from pathlib import Path


def _cuda_library_directory(
    provider_directory: Path, python_directory: str, package: str
) -> Path:
    directory = (
        provider_directory
        / ".venv"
        / "lib"
        / python_directory
        / "site-packages"
        / "nvidia"
        / package
        / "lib"
    )
    directory.mkdir(parents=True)
    return directory


def test_no_cuda_packages_leave_the_library_path_unset(tmp_path: Path) -> None:
    assert "LD_LIBRARY_PATH" not in provider_environment(tmp_path, tmp_path / "scratch")


@pytest.mark.parametrize("python_directory", ["python3.12", "python3.14"])
def test_library_path_names_the_cuda_packages_of_any_python_three_venv(
    tmp_path: Path, python_directory: str
) -> None:
    cublas = _cuda_library_directory(tmp_path, python_directory, "cublas")
    cudnn = _cuda_library_directory(tmp_path, python_directory, "cudnn")

    environment = provider_environment(tmp_path, tmp_path / "scratch")

    assert environment["LD_LIBRARY_PATH"] == os.pathsep.join([str(cublas), str(cudnn)])


@pytest.mark.parametrize("variable", ["NUMBA_CACHE_DIR", "TMPDIR", "TRITON_CACHE_DIR"])
def test_writable_directories_are_the_workers_scratch_directory(
    tmp_path: Path, variable: str
) -> None:
    scratch = tmp_path / "scratch"

    assert provider_environment(tmp_path, scratch)[variable] == str(scratch)


@pytest.mark.parametrize("visible_devices", ["", "0,1"])
def test_visible_devices_are_passed_through_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, visible_devices: str
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)

    assert (
        provider_environment(tmp_path, tmp_path / "scratch")["CUDA_VISIBLE_DEVICES"]
        == visible_devices
    )


def test_visible_devices_stay_unset_when_the_parent_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert "CUDA_VISIBLE_DEVICES" not in provider_environment(
        tmp_path, tmp_path / "scratch"
    )


def test_a_parent_token_does_not_reach_the_provider_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bearer = secrets.token_urlsafe(32)
    monkeypatch.setenv("SPEECH_PERSON_TOKEN", bearer)
    _cuda_library_directory(tmp_path, "python3.14", "cublas")

    environment = provider_environment(tmp_path, tmp_path / "scratch")

    assert set(environment) <= {
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "PYTHONUNBUFFERED",
        "PATH",
        "NUMBA_CACHE_DIR",
        "TMPDIR",
        "TRITON_CACHE_DIR",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
    }
    assert all(bearer not in value for value in environment.values())
