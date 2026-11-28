"""The image proof's driving: what teardown removes and what it names when it cannot."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

import image_stack
from image_judgement import baked_caller
from image_stack import REPOSITORY
from prove_speech import SENTENCES
from shipped_catalogue import CATALOGUE, first_language
from speech_client.wire import SpeechRequest
from speech_service.config import load_caller_config

if TYPE_CHECKING:
    from pathlib import Path

    from proof_identity import Minting
    from speech_client.wire import VoiceId


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
    ("failing", "failed", "commands", "left"),
    [
        pytest.param(set(), False, ["down", "image"], [], id="clean"),
        pytest.param(
            {"down"},
            False,
            ["down", "image"],
            ["compose down"],
            id="compose down fails",
        ),
        pytest.param(
            {"image"},
            False,
            ["down", "image"],
            ["docker image rm"],
            id="image removal fails",
        ),
        pytest.param(
            {"down", "image"},
            False,
            ["down", "image"],
            ["compose down", "docker image rm"],
            id="both",
        ),
        pytest.param(
            set(), True, ["logs", "down", "image"], [], id="a failed proof logs first"
        ),
    ],
)
def test_a_teardown_tries_both_removals_and_names_each_that_failed(
    monkeypatch: pytest.MonkeyPatch,
    failing: set[str],
    *,
    failed: bool,
    commands: list[str],
    left: list[str],
) -> None:
    fake = FakeDocker(failing)
    monkeypatch.setattr(image_stack.subprocess, "run", fake)

    result = image_stack.teardown(
        image_stack.Compose((REPOSITORY / "compose.yaml",), {}, "run"), failed=failed
    )

    assert fake.commands == commands
    assert [line.split(" failed")[0] for line in result.left] == left
    assert (result.tail is not None) == failed
    if failed:
        assert "logs broke" in (result.tail or "")


@pytest.mark.parametrize("size", [4096, 25 * 1024 * 1024, 25 * 1024 * 1024 + 1])
def test_a_form_of_a_size_has_a_body_of_exactly_that_many_bytes(size: int) -> None:
    form = image_stack.form_of_size(size, "de")

    assert len(form.body) == size
    assert form.content_type.startswith("multipart/form-data; boundary=")


def test_a_recording_form_carries_the_language_it_is_given_even_a_foreign_one() -> None:
    form = image_stack.recording_form(b"audio", "fr")

    assert b'name="language"\r\n\r\nfr' in form.body
    assert b'name="file"; filename="recording.wav"' in form.body


@dataclass(frozen=True)
class RecordingCompose(image_stack.Compose):
    """The process boundary of a caller container: it records what it was sent."""

    specifications: list[dict[str, str]] = field(default_factory=list)

    def run(
        self, *_: str, stdin: str | None = None, **__: object
    ) -> subprocess.CompletedProcess[str]:
        """Keep the request specification and answer with an empty 200."""
        assert stdin is not None
        self.specifications.append(json.loads(stdin))
        answer = {"status": 200, "body": "", "headers": {}}
        return subprocess.CompletedProcess([], 0, json.dumps(answer), "")


@pytest.mark.parametrize("voice", list(CATALOGUE))
def test_the_session_speaks_each_voice_in_a_language_the_service_accepts(
    minting: Minting, voice: VoiceId
) -> None:
    config = load_caller_config(REPOSITORY / "deploy" / "speech.toml")
    compose = RecordingCompose((), {}, "run")
    session = image_stack.Session(
        compose,
        config,
        baked_caller(config, next(iter(config.apps))),
        minting,
        tokens=[],
        heard=[],
        logs=[],
    )

    session.speak({}, voice)

    (specification,) = compose.specifications
    spoken = SpeechRequest.model_validate_json(base64.b64decode(specification["body"]))
    assert spoken.spoken_language == first_language(voice)
    assert spoken.input == SENTENCES[spoken.spoken_language]


def test_the_kills_for_memory_are_read_from_a_cgroups_events() -> None:
    kills = 2
    events = f"low 0\nmax 9\noom 3\noom_kill {kills}\noom_group_kill 0\n"

    assert image_stack.oom_kills(events) == kills


def test_a_session_over_another_stack_or_app_changes_only_that_part(
    minting: Minting,
) -> None:
    config = load_caller_config(REPOSITORY / "deploy" / "speech.toml")
    app = next(iter(config.apps))
    session = image_stack.Session(
        image_stack.Compose((), {}, "run"),
        config,
        baked_caller(config, app),
        minting,
        tokens=["token"],
        heard=["heard"],
        logs=["log"],
    )
    override = REPOSITORY / "compose.yaml"

    stacked = session.with_compose(session.compose.with_files((override,)))
    spoken_as = session.as_app(app)

    assert stacked.compose.files == (override,)
    assert stacked.compose.project == session.compose.project
    assert (stacked.baked, stacked.signing, stacked.tokens) == (
        session.baked,
        session.signing,
        session.tokens,
    )
    assert spoken_as.baked == baked_caller(config, app)
    assert (spoken_as.compose, spoken_as.heard, spoken_as.logs) == (
        session.compose,
        session.heard,
        session.logs,
    )


def test_a_service_started_with_an_override_keeps_the_log_and_the_proofs_files(
    monkeypatch: pytest.MonkeyPatch, minting: Minting
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "the log", "")

    monkeypatch.setattr(image_stack.subprocess, "run", run)
    config = load_caller_config(REPOSITORY / "deploy" / "speech.toml")
    app = next(iter(config.apps))
    session = image_stack.Session(
        image_stack.Compose((REPOSITORY / "compose.yaml",), {}, "run"),
        config,
        baked_caller(config, app),
        minting,
        tokens=[],
        heard=[],
        logs=[],
    )
    override = REPOSITORY / "override.yaml"

    stack = image_stack.start_with(session, override)

    assert stack.files == (*session.compose.files, override)
    assert session.logs == ["the log"]
    assert "--force-recreate" in commands[-1]
    assert commands[-1][-1] == image_stack.SERVICE


@pytest.fixture
def docker_with_image_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A docker whose image carries two settings and whose compose renders one."""
    fake = tmp_path / "docker"
    fake.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = compose ]; then\n'
        '  echo \'{"services": {"speech": {"environment": {"IMAGE_SETTING": "c"}}}}\'\n'
        "else\n"
        '  echo \'["PATH=/bin:/usr/bin","IMAGE_SETTING=a=b","OTHER_SETTING=d"]\'\n'
        "fi\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")


@pytest.mark.usefixtures("docker_with_image_environment")
def test_an_image_environment_is_read_from_the_image_itself() -> None:
    environment = image_stack.image_environment("image-under-proof")

    assert environment == {
        "PATH": "/bin:/usr/bin",
        "IMAGE_SETTING": "a=b",
        "OTHER_SETTING": "d",
    }


def test_a_docker_command_that_fails_ends_the_proof_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "docker"
    fake.write_text("#!/bin/sh\necho 'no daemon' >&2\nexit 1\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(SystemExit, match="docker image failed: no daemon"):
        image_stack.image_environment("image-under-proof")


@pytest.mark.usefixtures("docker_with_image_environment")
def test_a_service_runs_with_its_image_environment_under_the_compose_settings() -> None:
    compose = image_stack.Compose(
        (REPOSITORY / "compose.yaml",), {}, "image-under-proof"
    )

    environment = image_stack.service_config(compose)["environment"]

    assert environment == {
        "PATH": "/bin:/usr/bin",
        "IMAGE_SETTING": "c",
        "OTHER_SETTING": "d",
    }
