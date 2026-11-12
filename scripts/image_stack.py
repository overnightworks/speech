"""What the image proof drives: Docker, Compose and the caller container."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import httpx2

import hearing_figures as figures
from image_judgement import (
    TmpUsage,
    baked_caller,
    minting_for,
    require_marked_wav,
    seconds_to_healthy,
)
from proof_identity import TokenPair, mint_pair
from prove_speech import SENTENCES
from speech_client.wire import (
    CONTENT_TYPE_HEADER,
    MODEL,
    SPEECH_PATH,
    TRANSCRIPTIONS_PATH,
    VOICES,
    Language,
    ResponseFormat,
    TranscriptionRequest,
    speech_request_body,
    speech_request_headers,
)
from speech_service.config import Settings
from speech_service.pcm import LIVE_INPUT_RATE, pcm_from_wav, wav_header

if TYPE_CHECKING:
    from collections.abc import Mapping

    from image_judgement import BakedCaller
    from proof_identity import Minting
    from speech_client.wire import VoiceId
    from speech_service.config import CallerConfig

REPOSITORY: Final = Path(__file__).resolve().parents[1]
FIXTURES: Final = REPOSITORY / "tests" / "fixtures" / "image"
PROJECT: Final = "speech-proof"
IMAGE: Final = "speech-proof"
SERVICE: Final = "speech"
KEY_SERVER: Final = "key-server"
LIVE_PACE_SECONDS: Final = 0.05
LIVE_WAIT_SECONDS: Final = 120
MEMORY_PEAK_FILE: Final = "/sys/fs/cgroup/memory.peak"
HEALTHY_TIMEOUT_SECONDS: Final = 120
STOP_TIMEOUT_SECONDS: Final = 5
HEARING_LANGUAGE: Final = Language.DE.value


def caller_service(app: str) -> str:
    """The proof's caller container for `app`, on that app's network."""
    return f"caller-{app}"


@dataclass(frozen=True)
class Reply:
    """One answer, or the lack of it, as the caller container saw it."""

    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


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


def run_in(
    compose: Compose, container: str, fixture: str, specification: object
) -> str:
    """Run a fixture script in `container` with `specification` on its input."""
    script = (FIXTURES / fixture).read_text()
    return compose.run(
        "exec", "-T", container, "python", "-c", script,
        stdin=json.dumps(specification),
    ).stdout  # fmt: skip


def send(
    compose: Compose,
    caller: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
) -> Reply:
    """Send one request from inside a caller container; no answer ends the proof."""
    specification = {
        "url": url,
        "method": "GET" if body is None else "POST",
        "headers": dict(headers),
        "body": base64.b64encode(body or b"").decode("ascii"),
    }
    answer = json.loads(run_in(compose, caller, "caller.py", specification))
    if "error" in answer:
        message = f"no answer from {url}: {answer['error']}"
        raise SystemExit(message)
    return Reply(answer["status"], base64.b64decode(answer["body"]), answer["headers"])


@dataclass(frozen=True)
class RecordingForm:
    """A transcription request's multipart body and the header that names it."""

    content_type: str
    body: bytes


def recording_form(wav: bytes, language: str) -> RecordingForm:
    """The multipart request of `wav` in `language`; the language is sent as given."""
    fields = TranscriptionRequest(model=MODEL).model_dump(mode="json")
    request = httpx2.Request(
        "POST",
        "http://form.invalid",
        data={**fields, "language": language},
        files={"file": ("recording.wav", wav, "audio/wav")},
    )
    return RecordingForm(request.headers["content-type"], request.read())


def form_of_size(body_bytes: int, language: str) -> RecordingForm:
    """A request whose body is `body_bytes` long, around a silent marked WAV."""
    empty = len(recording_form(b"", language).body)
    header = wav_header(LIVE_INPUT_RATE)
    audio_bytes = body_bytes - empty - len(header)
    wav = wav_header(LIVE_INPUT_RATE, audio_bytes) + bytes(audio_bytes)
    return recording_form(wav, language)


def live_frames(wav: bytes) -> list[str]:
    """A speak answer as the base64 frames a live client sends, with closing silence."""
    rate, pcm = pcm_from_wav(wav)
    frames = figures.frames_with_closing_silence(figures.to_wire_rate(pcm, rate))
    return [base64.b64encode(frame).decode("ascii") for frame in frames]


@dataclass(frozen=True)
class LiveOutcome:
    """How a live turn ended: its final and partials, or the error that stopped it."""

    final: str | None
    partials: tuple[str, ...]
    error: str | None
    language: str | None = None


@dataclass
class Session:
    """What the cells share: the stack, the baked callers and what the log must lack.

    `baked` is the caller the cells speak as; `tokens` and `heard` are what the log
    must not hold.
    """

    compose: Compose
    config: CallerConfig
    baked: BakedCaller
    signing: Minting
    tokens: list[str]
    heard: list[str]
    logs: list[str]

    def as_app(self, app: str) -> Session:
        """The session speaking as another baked app."""
        return replace(self, baked=baked_caller(self.config, app))

    @property
    def caller(self) -> str:
        """The caller container of the session's app."""
        return caller_service(self.baked.app)

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
        """Post the proof sentence of the voice's language from the caller container."""
        language = VOICES[voice].languages[0]
        return send(
            self.compose,
            self.caller,
            self.service_url + SPEECH_PATH,
            headers,
            speech_request_body(
                SENTENCES[language], voice, ResponseFormat.WAV, language
            ),
        )

    def admitted_headers(self) -> dict[str, str]:
        """The headers of a call the service admits."""
        pair = self.pair()
        return speech_request_headers(
            workload_token=pair.workload, person_token=pair.person
        )

    def health(self) -> dict[str, dict[str, object]]:
        """`/health` as the caller container reads it."""
        reply = send(self.compose, self.caller, self.service_url + "/health", {}, None)
        return json.loads(reply.body)

    def post_recording(self, headers: Mapping[str, str], form: RecordingForm) -> Reply:
        """Post one recording form to the file route from the caller container."""
        no_json = {k: v for k, v in headers.items() if k != CONTENT_TYPE_HEADER}
        return send(
            self.compose,
            self.caller,
            self.service_url + TRANSCRIPTIONS_PATH,
            {**no_json, CONTENT_TYPE_HEADER: form.content_type},
            form.body,
        )

    def hear(self, headers: Mapping[str, str], wav: bytes, language: str) -> Reply:
        """Post `wav` to the file route."""
        return self.post_recording(headers, recording_form(wav, language))

    def hear_live(
        self,
        tokens: TokenPair | None,
        wav: bytes,
        languages: tuple[Language, ...] = (Language.DE,),
    ) -> LiveOutcome:
        """One live turn in a session of `languages`, from the caller container.

        Without `tokens` the session offers empty ones, as a caller with none would.
        """
        offered = tokens or TokenPair(workload="", person="")
        specification = {
            "base_url": self.service_url,
            "workload_token": offered.workload,
            "person_token": offered.person,
            "languages": [language.value for language in languages],
            "frames": live_frames(wav),
            "pace_seconds": LIVE_PACE_SECONDS,
            "wait_seconds": LIVE_WAIT_SECONDS,
        }
        answer = json.loads(run_in(self.compose, self.caller, "live.py", specification))
        return LiveOutcome(
            answer.get("final"),
            tuple(answer.get("partials", ())),
            answer.get("error"),
            answer.get("language"),
        )

    def tmp_usage(self) -> TmpUsage:
        """The filesystem's own count of the service's /tmp now."""
        output = run_in(self.compose, SERVICE, "tmp_usage.py", None)
        return TmpUsage(**json.loads(output))

    def memory_peak_mib(self) -> int:
        """The service container's peak memory since it started, from its cgroup."""
        peak = self.compose.run("exec", "-T", SERVICE, "cat", MEMORY_PEAK_FILE)
        return int(peak.stdout) // 2**20

    def collect_log(self) -> None:
        """Keep the service's log so far; a recreated container loses its own."""
        self.logs.append(self.compose.run("logs", "--no-color", SERVICE).stdout)


def service_config(compose: Compose) -> dict[str, Any]:
    """The service as `compose` renders it."""
    rendered = json.loads(compose.run("config", "--format", "json").stdout)
    return rendered["services"][SERVICE]


def speak_marked(session: Session, voice: VoiceId | None = None) -> bytes:
    """One admitted speak of `voice` or the baked one; the WAV carries the AI mark."""
    reply = session.speak(session.admitted_headers(), voice or session.baked.voice)
    if reply.status != HTTPStatus.OK:
        message = f"the admitted speak answered {reply.status}"
        raise SystemExit(message)
    require_marked_wav(reply.body)
    return reply.body


def start_with(session: Session, override: Path) -> Compose:
    """Recreate the service with `override` over the proof's files, without waiting."""
    session.collect_log()
    stack = replace(session.compose, files=(*session.compose.files, override))
    stack.run("up", "--detach", "--no-build", "--no-deps", "--force-recreate", SERVICE)
    return stack


def healthcheck_passes(compose: Compose, healthcheck: Mapping[str, list[str]]) -> bool:
    """Whether the rendered healthcheck, run in the service container, passes."""
    kind, *command = healthcheck["test"]
    if kind != "CMD":
        message = f"the healthcheck is not an exec form: {kind}"
        raise SystemExit(message)
    return compose.run("exec", "-T", SERVICE, *command, check=False).returncode == 0


def seconds_until_healthy(compose: Compose) -> float:
    """From the service container's start to its first passing healthcheck."""
    state = json.loads(docker("inspect", container_of(compose, SERVICE)))[0]["State"]
    return seconds_to_healthy(state["StartedAt"], state["Health"]["Log"])


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
