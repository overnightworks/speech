"""The image proof's entry point and the files it keeps honest."""

from __future__ import annotations

import json
import re
import runpy
import sys
from typing import TYPE_CHECKING

import pytest

from tests.support.script_support import script_module

if TYPE_CHECKING:
    from pathlib import Path

PROVE = script_module("prove_image")
INSPECT_IMAGE = PROVE.FIXTURES / "inspect_image.py"
PRIVATE_KEY = b"-----BEGIN PRIVATE KEY-----"


def test_every_job_the_quality_gate_waits_for_is_named_in_its_needs() -> None:
    workflow = (PROVE.REPOSITORY / ".github" / "workflows" / "ci.yml").read_text()
    jobs, _, gate = workflow.partition("\n  quality:\n")
    jobs = jobs.partition("\njobs:\n")[2]
    needs = set(re.findall(r"^      - (\S+)$", gate.split("if:")[0], re.MULTILINE))

    assert needs == set(re.findall(r"^  ([a-z-]+):$", jobs, re.MULTILINE))


def test_the_build_context_admits_only_files_the_image_copies() -> None:
    admitted = [
        line.removeprefix("!")
        for line in (PROVE.REPOSITORY / ".dockerignore").read_text().splitlines()
        if line.startswith("!")
    ]
    dockerfile = (PROVE.REPOSITORY / "Dockerfile").read_text()
    copied = {
        source
        for line in re.findall(r"^COPY (?!--from)(.+)$", dockerfile, re.MULTILINE)
        for source in line.split()[:-1]
    }

    assert all(path.partition("/**/")[0] in copied for path in admitted)


def dockerignore_lines() -> list[str]:
    lines = (PROVE.REPOSITORY / ".dockerignore").read_text().splitlines()
    return [line for line in lines if line and not line.startswith("#")]


def source_admissions_beyond_python(admissions: list[str]) -> list[str]:
    """The admissions into a `src` tree that take more than Python source."""
    return [
        admission
        for admission in admissions
        if "src" in admission.removeprefix("!").split("/")
        and not admission.endswith(("/*.py", "/py.typed"))
    ]


@pytest.mark.parametrize(
    ("admissions", "beyond"),
    [
        pytest.param(["!src/**/*.py"], [], id="python source"),
        pytest.param(["!packages/client/src/**/py.typed"], [], id="py.typed"),
        pytest.param(["!src/**"], ["!src/**"], id="the whole top tree"),
        pytest.param(["!src"], ["!src"], id="the tree itself"),
        pytest.param(["!pkg/src/**/*.pem"], ["!pkg/src/**/*.pem"], id="a key suffix"),
    ],
)
def test_an_admission_into_a_source_tree_is_refused_unless_it_is_python_source(
    admissions: list[str], beyond: list[str]
) -> None:
    assert source_admissions_beyond_python(admissions) == beyond


def test_the_source_trees_admit_python_source_and_py_typed_and_nothing_else() -> None:
    lines = dockerignore_lines()

    assert any("src" in line.split("/") for line in lines)
    assert not source_admissions_beyond_python(
        [line for line in lines if line.startswith("!")]
    )


def test_a_private_key_in_a_python_file_of_the_packages_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    site_packages = tmp_path / "site-packages"
    package = site_packages / "speech_service"
    package.mkdir(parents=True)
    (package / "leaked.py").write_bytes(b'KEY = """' + PRIVATE_KEY + b'"""')
    (package / "clean.py").write_text("VALUE = 1")
    voices = tmp_path / "voices"
    voices.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sysconfig.get_paths", lambda: {"purelib": str(site_packages)})
    monkeypatch.setattr(
        sys, "argv", [str(INSPECT_IMAGE), str(voices), str(tmp_path / "hub"), "small"]
    )

    runpy.run_path(str(INSPECT_IMAGE), run_name="__main__")

    report = json.loads(capsys.readouterr().out)
    assert report["package_files_with_private_keys"] == ["speech_service/leaked.py"]
    assert report["package_stray_files"] == []


def test_a_failed_removal_of_the_temporary_files_fails_the_proof_loudly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class StoppedCompose:
        def __init__(self, *_: object) -> None:
            pass

        def run(self, *_: object, **__: object) -> object:
            message = "stopped"
            raise SystemExit(message)

    directory = tmp_path / "proof"
    stuck = directory / "stuck"
    stuck.mkdir(parents=True)
    (stuck / "file").write_text("x")
    stuck.chmod(0o500)
    monkeypatch.setenv(PROVE.PROJECT_ENV, "a-run")
    monkeypatch.setattr(PROVE, "wait_for_load", lambda: None)
    monkeypatch.setattr(PROVE.tempfile, "mkdtemp", lambda **_: str(directory))
    monkeypatch.setattr(PROVE, "Compose", StoppedCompose)
    monkeypatch.setattr(
        PROVE, "teardown", lambda *_, **__: PROVE.Teardown(tail=None, left=[])
    )
    try:
        with pytest.raises(PermissionError):
            PROVE._prove()  # noqa: SLF001 - the whole run is the entry point under test
    finally:
        stuck.chmod(0o700)

    assert "PROOF_OK" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("outcome", "stdout_starts", "stderr"),
    [
        pytest.param(PROVE.Teardown(tail=None, left=[]), "torn_down", "", id="clean"),
        pytest.param(
            PROVE.Teardown(tail="a log line\nan error", left=[]),
            "a log line\nan error\ntorn_down",
            "",
            id="a failed proof prints its log tail before torn_down",
        ),
        pytest.param(
            PROVE.Teardown(tail=None, left=["compose down failed: busy"]),
            "",
            "teardown left something:\ncompose down failed: busy\n",
            id="a removal that failed is named on stderr",
        ),
    ],
)
def test_the_teardown_prints_its_outcome_in_order(
    capsys: pytest.CaptureFixture[str],
    outcome: object,
    stdout_starts: str,
    stderr: str,
) -> None:
    PROVE.print_teardown(outcome, "a-run")

    captured = capsys.readouterr()
    assert captured.out.startswith(stdout_starts)
    assert ("torn_down" in captured.out) is (not stderr)
    assert captured.err == stderr


def test_the_image_built_for_a_run_is_removed_when_the_drive_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed: list[str] = []
    built: list[str] = []
    monkeypatch.setattr(PROVE, "cell_build", built.append)
    monkeypatch.setattr(PROVE, "remove_image", removed.append)

    def refused(*_: object, **__: object) -> int:
        message = "probe-run is not on PATH"
        raise SystemExit(message)

    monkeypatch.setattr(PROVE, "run_under_probe_run", refused)

    with pytest.raises(SystemExit):
        PROVE.build_then_drive([])

    assert removed == built


def test_two_runs_never_share_a_project_name() -> None:
    assert PROVE.new_project() != PROVE.new_project()
