"""The image proof's driving: what teardown removes and what it names when it cannot."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

import image_stack
from image_stack import REPOSITORY
from speech_client.wire import VOICES, SpeechRequest, VoiceId

if TYPE_CHECKING:
    from collections.abc import Mapping


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


@pytest.mark.parametrize("voice", list(VoiceId))
def test_the_session_speaks_each_voice_in_a_language_the_service_accepts(
    monkeypatch: pytest.MonkeyPatch, voice: VoiceId
) -> None:
    posted: list[bytes] = []

    def send(
        _compose: object,
        _caller: str,
        _url: str,
        _headers: Mapping[str, str],
        body: bytes | None,
    ) -> None:
        posted.append(body or b"")

    monkeypatch.setattr(image_stack, "send", send)
    session = SimpleNamespace(
        compose=None, caller="caller", service_url="http://service"
    )

    image_stack.Session.speak(session, {}, voice)  # type: ignore[arg-type]

    (body,) = posted
    spoken = SpeechRequest.model_validate_json(body)
    assert spoken.spoken_language == VOICES[voice].languages[0]
