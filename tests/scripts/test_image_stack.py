"""The image proof's driving: what teardown removes and what it names when it cannot."""

from __future__ import annotations

import base64
import json
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

import image_stack
from image_judgement import baked_caller
from image_stack import REPOSITORY
from prove_speech import SENTENCES
from speech_client.wire import VOICES, SpeechRequest, VoiceId
from speech_service.config import load_caller_config

if TYPE_CHECKING:
    from pathlib import Path

    from proof_identity import Minting


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


@pytest.mark.parametrize("voice", list(VoiceId))
def test_the_session_speaks_each_voice_in_a_language_the_service_accepts(
    minting: Minting, voice: VoiceId
) -> None:
    config = load_caller_config(REPOSITORY / "deploy" / "speech.toml")
    compose = RecordingCompose((), {})
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
    assert spoken.spoken_language == VOICES[voice].languages[0]
    assert spoken.input == SENTENCES[spoken.spoken_language]


def test_every_baked_app_has_a_proof_caller_on_a_network_the_service_shares(
    tmp_path: Path,
) -> None:
    config = load_caller_config(REPOSITORY / "deploy" / "speech.toml")
    compose = image_stack.Compose(
        (
            REPOSITORY / "compose.yaml",
            REPOSITORY / "tests/fixtures/image/compose.proof.yaml",
        ),
        {
            "PROOF_KEY_SET_PATH": "key-set",
            "PROOF_KEY_SET_HOST": "key-set.example",
            "PROOF_DIRECTORY": str(tmp_path),
        },
    )

    services = json.loads(compose.run("config", "--format", "json").stdout)["services"]

    service_networks = set(services[image_stack.SERVICE]["networks"])
    for app in config.apps:
        caller_networks = set(services[image_stack.caller_service(app)]["networks"])
        assert caller_networks
        assert caller_networks <= service_networks
