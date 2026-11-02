"""What the image proof drives: Docker, Compose and the caller container."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from image_judgement import minting_for
from proof_identity import mint_pair
from prove_speech import SENTENCE
from speech_runtime_client.wire import (
    SPEECH_PATH,
    speech_request_body,
    speech_request_headers,
)
from speech_runtime_service.config import Settings

if TYPE_CHECKING:
    from collections.abc import Mapping

    from image_judgement import BakedCaller
    from proof_identity import Minting, TokenPair
    from speech_runtime_service.config import CallerConfig
    from speech_runtime_service.voices import VoiceId

REPOSITORY: Final = Path(__file__).resolve().parents[1]
FIXTURES: Final = REPOSITORY / "tests" / "fixtures" / "image"
PROJECT: Final = "speech-runtime-proof"
IMAGE: Final = "speech-runtime-proof"
SERVICE: Final = "speech-runtime"
KEY_SERVER: Final = "key-server"
CALLER: Final = "caller"
HEALTHY_TIMEOUT_SECONDS: Final = 120
STOP_TIMEOUT_SECONDS: Final = 5


@dataclass(frozen=True)
class Reply:
    """One answer, or the lack of it, as the caller container saw it."""

    status: int
    body: bytes


@dataclass(frozen=True)
class Compose:
    """`docker compose` for the proof's project over some files."""

    files: tuple[Path, ...]
    environment: Mapping[str, str]

    def command(self, *arguments: str) -> list[str]:
        """The full argv of one compose subcommand."""
        files = [argument for file in self.files for argument in ("-f", str(file))]
        return [
            "docker",
            "compose",
            "--project-directory",
            str(REPOSITORY),
            "-p",
            PROJECT,
            *files,
            *arguments,
        ]

    def run(
        self, *arguments: str, stdin: str | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Run one compose subcommand; with `check`, a failure ends the proof."""
        completed = subprocess.run(  # noqa: S603 - docker with this repository's files and fixed arguments
            self.command(*arguments),
            input=stdin,
            env={**os.environ, **self.environment},
            capture_output=True,
            text=True,
            check=False,
        )
        if check and completed.returncode != 0:
            message = (
                f"compose {arguments[0]} failed: {completed.stderr.strip()[-600:]}"
            )
            raise SystemExit(message)
        return completed

    def up(self, *services: str, recreate: bool = False) -> None:
        """Start `services` (all, when none) and wait until they are healthy."""
        flags = ["--detach", "--no-build", "--wait"]
        flags += ["--wait-timeout", str(HEALTHY_TIMEOUT_SECONDS)]
        flags += ["--no-deps"] if services else []
        flags += ["--force-recreate"] if recreate else []
        self.run("up", *flags, *services)


def docker(*arguments: str, stdin: str | None = None) -> str:
    """Run one docker command; a failure ends the proof."""
    completed = subprocess.run(  # noqa: S603 - docker with fixed arguments
        ["docker", *arguments],  # noqa: S607 - docker is the host's own client
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = f"docker {arguments[0]} failed: {completed.stderr.strip()[-600:]}"
        raise SystemExit(message)
    return completed.stdout


def container_of(compose: Compose, service: str) -> str:
    """The id of the one container of `service`."""
    return compose.run("ps", "--quiet", service).stdout.strip()


def send(
    compose: Compose, url: str, headers: Mapping[str, str], body: bytes | None
) -> Reply:
    """Send one request from inside the caller container; no answer ends the proof."""
    specification = {
        "url": url,
        "method": "GET" if body is None else "POST",
        "headers": dict(headers),
        "body": base64.b64encode(body or b"").decode("ascii"),
    }
    script = (FIXTURES / "caller.py").read_text()
    output = compose.run(
        "exec", "-T", CALLER, "python", "-", json.dumps(specification), stdin=script
    ).stdout
    answer = json.loads(output)
    if "error" in answer:
        message = f"no answer from {url}: {answer['error']}"
        raise SystemExit(message)
    return Reply(answer["status"], base64.b64decode(answer["body"]))


@dataclass
class Session:
    """What the cells share: the stack, the baked caller and the tokens spoken with."""

    compose: Compose
    config: CallerConfig
    baked: BakedCaller
    signing: Minting
    tokens: list[str]
    logs: list[str]

    @property
    def service_url(self) -> str:
        """The service as the caller container reaches it."""
        return f"http://{SERVICE}:{Settings.model_fields['port'].default}"

    def pair(self, minting: Minting | None = None) -> TokenPair:
        """Fresh tokens, remembered for the log check."""
        pair = mint_pair(minting or minting_for(self.config, self.baked, self.signing))
        self.tokens += [pair.workload, pair.person]
        return pair

    def speak(self, headers: Mapping[str, str], voice: VoiceId) -> Reply:
        """Post one sentence from the caller container."""
        return send(
            self.compose,
            self.service_url + SPEECH_PATH,
            headers,
            speech_request_body(SENTENCE, voice.value),
        )

    def admitted_headers(self) -> dict[str, str]:
        """The headers of a call the service admits."""
        pair = self.pair()
        return speech_request_headers(
            workload_token=pair.workload, person_token=pair.person
        )

    def health(self) -> dict[str, dict[str, object]]:
        """`/health` as the caller container reads it."""
        reply = send(self.compose, self.service_url + "/health", {}, None)
        return json.loads(reply.body)

    def collect_log(self) -> None:
        """Keep the service's log so far; a recreated container loses its own."""
        self.logs.append(self.compose.run("logs", "--no-color", SERVICE).stdout)


@dataclass(frozen=True)
class Teardown:
    """What teardown did: a failed proof's log tail and the removals that failed."""

    tail: str | None
    left: list[str]


def teardown(compose: Compose, *, failed: bool) -> Teardown:
    """Remove the proof's containers, networks, volumes and image.

    Every removal is tried; the ones that failed are named.
    """
    tail = None
    if failed:
        logs = compose.run("logs", "--no-color", "--tail", "30", check=False)
        tail = f"{logs.stdout}\n{logs.stderr}"
    attempts = {
        "compose down": compose.run(
            "down",
            "--volumes",
            "--remove-orphans",
            "--timeout",
            str(STOP_TIMEOUT_SECONDS),
            check=False,
        ),
        "docker image rm": subprocess.run(  # noqa: S603 - docker with fixed arguments
            ["docker", "image", "rm", "--force", IMAGE],  # noqa: S607 - docker is the host's own client
            capture_output=True,
            text=True,
            check=False,
        ),
    }
    left = [
        f"{what} failed: {done.stderr.strip()[-300:]}"
        for what, done in attempts.items()
        if done.returncode != 0
    ]
    return Teardown(tail=tail, left=left)
