"""The image proof's driving: what teardown removes and what it names when it cannot."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

import image_stack
from image_stack import REPOSITORY
from tests.support.docker_support import DockerOnPath
from tests.support.nvidia_support import FakeNvidiaSmi

if TYPE_CHECKING:
    from pathlib import Path


class FakeDocker:
    """The process boundary: every command ends with the code its name is given."""

    def __init__(self, failing: set[str]) -> None:
        """Fail the commands named in `failing`."""
        self.failing = failing
        self.commands: list[str] = []

    def __call__(
        self, command: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        """Answer one command with the code its name was given."""
        name = next(word for word in ("image", "down", "logs") if word in command)
        self.commands.append(name)
        code = 1 if name in self.failing else 0
        return subprocess.CompletedProcess(command, code, "", f"{name} broke")


@pytest.mark.parametrize(
    ("failing", "left"),
    [
        pytest.param(set(), [], id="clean"),
        pytest.param({"down"}, ["compose down"], id="compose down fails"),
        pytest.param({"image"}, ["docker image rm"], id="image removal fails"),
        pytest.param({"down", "image"}, ["compose down", "docker image rm"], id="both"),
    ],
)
def test_a_teardown_tries_both_removals_and_names_each_that_failed(
    monkeypatch: pytest.MonkeyPatch,
    failing: set[str],
    left: list[str],
) -> None:
    fake = FakeDocker(failing)
    monkeypatch.setattr(image_stack.subprocess, "run", fake)

    result = image_stack.teardown(
        image_stack.Compose((REPOSITORY / "compose.yaml",), {}), failed=False
    )

    assert fake.commands == ["down", "image"]
    assert [line.split(" failed")[0] for line in result.left] == left
    assert result.tail is None


@pytest.mark.parametrize("size", [4096, 25 * 1024 * 1024, 25 * 1024 * 1024 + 1])
def test_a_form_of_a_size_has_a_body_of_exactly_that_many_bytes(size: int) -> None:
    form = image_stack.form_of_size(size, "de")

    assert len(form.body) == size
    assert form.content_type.startswith("multipart/form-data; boundary=")


def test_a_recording_form_carries_the_language_it_is_given_even_a_foreign_one() -> None:
    form = image_stack.recording_form(b"audio", "fr")

    assert b'name="language"\r\n\r\nfr' in form.body
    assert b'name="file"; filename="recording.wav"' in form.body


def test_a_machine_without_the_card_is_refused_before_anything_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    DockerOnPath(tmp_path, monkeypatch, {"info": '{"runc": {}}'})
    FakeNvidiaSmi(tmp_path, monkeypatch).fail()

    with pytest.raises(SystemExit) as stopped:
        image_stack.require_card()

    assert "no card" in str(stopped.value.code)
    assert "no nvidia runtime" in str(stopped.value.code)
    assert "--no-card" in str(stopped.value.code)


def test_a_machine_with_the_card_and_the_runtime_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    DockerOnPath(tmp_path, monkeypatch, {"info": '{"runc": {}, "nvidia": {}}'})
    FakeNvidiaSmi(tmp_path, monkeypatch).use(0)

    image_stack.require_card()


def test_the_pids_of_a_container_are_read_from_its_process_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    DockerOnPath(
        tmp_path,
        monkeypatch,
        {"ps --quiet": "container", "top container": "PID\n4242\n4300"},
    )

    pids = image_stack.container_pids(
        image_stack.Compose((REPOSITORY / "compose.yaml",), {}), "speech-runtime"
    )

    assert pids == {4242, 4300}
