"""Image proof: build the image and run the real compose file on isolated networks.

`tests/fixtures/image/compose.proof.yaml` is laid over `compose.yaml`: the
external networks become internal ones, a test CA takes the secret's place, and
a key-set server and a caller container join. The tokens carry the issuer,
audience, group and caller of the baked `deploy/speech.toml`. The proof takes
/tmp/probe-stack.lock like `prove_speech.py` and removes everything it made.
"""

# ruff: noqa: T201 - the proof prints its cell lines as the command line output

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
from dataclasses import dataclass, replace
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from packaging.markers import Marker

from proof_identity import Minting, TokenPair, load_minting, mint_pair, proof_identity
from prove_speech import (
    LOCK_HELD_ENV,
    LOCK_PATH,
    SENTENCE,
    parse_arguments,
    prove_under_lock,
    run_under_lock,
    spoken_in_log,
    wait_for_load,
)
from speech_runtime_client.wire import (
    REFUSAL_STATUS,
    SPEECH_PATH,
    Refusal,
    RefusalCode,
    speech_request_body,
    speech_request_headers,
)
from speech_runtime_service.config import CallerConfig, Settings, load_caller_config
from speech_runtime_service.pcm import AI_MARK
from speech_runtime_service.piper_voices import PINS, PinnedVoice
from speech_runtime_service.voices import DEFAULT_SPEAKING_MODEL, VoiceId

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

REPOSITORY: Final = Path(__file__).resolve().parents[1]
THIS_SCRIPT: Final = Path(__file__).resolve()
COMPOSE_FILE: Final = REPOSITORY / "compose.yaml"
BAKED_CONFIG: Final = REPOSITORY / "deploy" / "speech.toml"
FIXTURES: Final = REPOSITORY / "tests" / "fixtures" / "image"
PROOF_OVERRIDE: Final = FIXTURES / "compose.proof.yaml"
LOOPBACK_OVERRIDE: Final = FIXTURES / "compose.loopback.yaml"
PROJECT: Final = "speech-runtime-proof"
IMAGE: Final = "speech-runtime-proof"
SERVICE: Final = "speech-runtime"
KEY_SERVER: Final = "key-server"
CALLER: Final = "caller"
# hal builds with a cap of 1800 s; a build past half of it is a finding.
BUILD_LIMIT_SECONDS: Final = 900
HEALTHY_TIMEOUT_SECONDS: Final = 120
STOP_TIMEOUT_SECONDS: Final = 5
CA_NAME: Final = "proof CA"
CA_KEY_USAGE: Final = x509.KeyUsage(
    digital_signature=False, content_commitment=False, key_encipherment=False,
    data_encipherment=False, key_agreement=False, key_cert_sign=True,
    crl_sign=False, encipher_only=False, decipher_only=False,
)  # fmt: skip
PUBLIC_ADDRESSES: Final = ("1.1.1.1:443", "huggingface.co:443")
FORBIDDEN_EXECUTABLES: Final = ("git", "uv", "gcc", "cc")
FORBIDDEN_PATH_PARTS: Final = frozenset({".git", "tests", "providers"})
KEY_FILE_SUFFIXES: Final = (".pem", ".key")
KEY_FILE_PREFIXES: Final = (".env", "id_")
MARK_CHUNK: Final = b"LIST"


@dataclass(frozen=True)
class BakedCaller:
    """What `deploy/speech.toml` says about the one caller the proof speaks as."""

    app: str
    voice: VoiceId
    other_voice: VoiceId
    key_set_host: str
    key_set_port: int
    key_set_path: str


@dataclass(frozen=True)
class ImageReport:
    """What `inspect_image.py` found inside the image."""

    distributions: list[str]
    user: int
    application_files: list[str]
    application_files_with_private_keys: list[str]
    voice_hashes: dict[str, str]
    executables: dict[str, str | None]


@dataclass(frozen=True)
class Reply:
    """One answer, or the lack of it, as the caller container saw it."""

    status: int
    body: bytes


def baked_caller(config: CallerConfig) -> BakedCaller:
    """The caller, its voices and the key set's address, read from the baked table."""
    (app,) = config.apps
    voices = sorted(config.apps[app].voices)
    (other_voice, *_) = sorted(set(VoiceId) - set(voices))
    key_set = urlsplit(config.identity.jwks_url)
    assert key_set.hostname is not None  # noqa: S101 - the owner validated an https URL
    return BakedCaller(
        app=app,
        voice=voices[0],
        other_voice=other_voice,
        key_set_host=key_set.hostname,
        key_set_port=key_set.port or 443,
        key_set_path=key_set.path,
    )


def minting_for(config: CallerConfig, baked: BakedCaller, signing: Minting) -> Minting:
    """The proof's signing key, speaking as the baked table's caller and issuer."""
    identity = config.identity
    return replace(
        signing,
        issuer=identity.issuer,
        audience=identity.audience,
        client_id=config.apps[baked.app].workload_client,
        workload_role=identity.workload_role,
        group_claim=identity.group_claim,
        app_group=f"/apps/{baked.app}",
        service_group=identity.service_group,
    )


def _chunks(wav: bytes) -> Iterator[tuple[bytes, int, int]]:
    """The id, start and end of each RIFF chunk after the header."""
    position = 12
    while position + 8 <= len(wav):
        size = int.from_bytes(wav[position + 4 : position + 8], "little")
        end = position + 8 + size + size % 2
        yield wav[position : position + 4], position, end
        position = end


def ai_mark_of(wav: bytes) -> str | None:
    """The RIFF INFO comment before the `data` chunk, or None."""
    for chunk_id, start, end in _chunks(wav):
        if chunk_id == b"data":
            return None
        if chunk_id == MARK_CHUNK and wav[start + 8 : start + 12] == b"INFO":
            return wav[start + 20 : end].split(b"\0", 1)[0].decode("ascii", "replace")
    return None


def without_mark(wav: bytes) -> bytes:
    """The same WAV with its INFO chunk cut out."""
    for chunk_id, start, end in _chunks(wav):
        if chunk_id == MARK_CHUNK:
            return wav[:start] + wav[end:]
    return wav


def require_marked_wav(wav: bytes) -> None:
    """Fail unless `wav` is a WAV that carries the AI mark and some audio."""
    if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        message = "the answer is not a WAV"
        raise SystemExit(message)
    if ai_mark_of(wav) != AI_MARK:
        message = "the WAV does not carry the AI mark"
        raise SystemExit(message)
    if len(wav) <= wav.find(b"data") + 8:
        message = "the WAV holds no audio"
        raise SystemExit(message)


def require_refusal(reply: Reply, expected: RefusalCode) -> None:
    """Fail unless `reply` is the wire refusal `expected`, with its status."""
    try:
        refusal = Refusal.model_validate_json(reply.body)
    except ValueError:
        message = f"status {reply.status} is no wire refusal"
        raise SystemExit(message) from None
    code = refusal.error.code
    if code is not expected or reply.status != REFUSAL_STATUS[code]:
        message = f"status {reply.status} refused as {code}, expected {expected}"
        raise SystemExit(message)


def require_fails(action: Callable[[], object], what: str) -> None:
    """Fail unless `action` ends the proof's way, with a SystemExit."""
    try:
        action()
    except SystemExit:
        return
    message = f"the proof did not fail on {what}"
    raise SystemExit(message)


def distribution_name(name: str) -> str:
    """The normalised form in which package names compare."""
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_distributions(export: str, project_name: Callable[[str], str]) -> set[str]:
    """The names `uv export` lists for this platform.

    `project_name` reads an editable entry's path.
    """
    names = set()
    for line in export.splitlines():
        requirement, _, marker = line.partition(";")
        if marker.strip() and not Marker(marker.strip()).evaluate():
            continue
        if requirement.startswith("-e "):
            path = requirement.removeprefix("-e ").strip()
            names.add(distribution_name(project_name(path)))
        elif match := re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(==|@)", requirement):
            names.add(distribution_name(match.group(1)))
    return names


def image_faults(
    report: ImageReport, locked: set[str], voice_hashes: dict[str, str]
) -> list[str]:
    """What the image holds that it must not, or lacks; empty when it is as locked."""
    installed = {distribution_name(name) for name in report.distributions}
    faults = [f"missing {name}" for name in sorted(locked - installed)]
    faults += [f"extra {name}" for name in sorted(installed - locked)]
    if report.user == 0:
        faults.append("runs as root")
    for path in map(Path, report.application_files):
        forbidden = (
            FORBIDDEN_PATH_PARTS & set(path.parts)
            or path.name.startswith(KEY_FILE_PREFIXES)
            or path.suffix in KEY_FILE_SUFFIXES
        )
        if forbidden:
            faults.append(f"holds {path}")
    faults += [
        f"holds a private key in {name}"
        for name in report.application_files_with_private_keys
    ]
    faults += [f"carries {name}" for name, found in report.executables.items() if found]
    if report.voice_hashes != voice_hashes:
        faults.append("the voice files are not the owner's pinned bytes")
    return faults


def log_faults(log: str, spoken: Iterable[str], tokens: Iterable[str]) -> list[str]:
    """The spoken texts and tokens a service log holds."""
    leaked = [f"spoken text {text[:12]!r}" for text in spoken_in_log(log, spoken)]
    return leaked + ["a token" for token in tokens if token in log]


def external_networks(
    compose_config: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> list[str]:
    """The names of the networks a rendered compose file takes from outside."""
    return [
        str(network["name"])
        for network in compose_config["networks"].values()
        if network.get("external")
    ]


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


def cell_build(compose: Compose) -> None:
    """Cell 1: a cold build, timed."""
    started = time.monotonic()
    compose.run("build", SERVICE)
    seconds = time.monotonic() - started
    size = int(docker("image", "inspect", "--format", "{{.Size}}", IMAGE))
    print(f"cell 1: build_seconds={seconds:.0f} image_mib={size // 2**20}")
    if seconds > BUILD_LIMIT_SECONDS:
        message = f"the build took {seconds:.0f} s, past {BUILD_LIMIT_SECONDS} s"
        raise SystemExit(message)


def _project_name(path: str) -> str:
    return tomllib.loads((REPOSITORY / path / "pyproject.toml").read_text())["project"][
        "name"
    ]


def _locked_set() -> set[str]:
    export = subprocess.run(
        ["uv", "export", "--locked", "--no-dev", "--no-hashes"],  # noqa: S607 - the repository's own tool
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return locked_distributions(export, _project_name)


def cell_contents(plain: Compose) -> None:
    """Cell 2: the image holds the locked set and the pinned voice, no more."""
    environment = json.loads(plain.run("config", "--format", "json").stdout)[
        "services"
    ][SERVICE]["environment"]
    pin = PINS[PinnedVoice(DEFAULT_SPEAKING_MODEL)]
    voice_hashes = {
        Path(file.path).name: file.sha256 for file in (pin.model, pin.metadata)
    }
    script = (FIXTURES / "inspect_image.py").read_text()
    output = docker(
        "run", "--rm", "--interactive", "--network", "none", "--entrypoint", "python",
        IMAGE, "-", environment["SPEECH_VOICE_CACHE"], *FORBIDDEN_EXECUTABLES,
        stdin=script,
    )  # fmt: skip
    report = ImageReport(**json.loads(output))
    faults = image_faults(report, _locked_set(), voice_hashes)
    if faults:
        message = "the image is not as locked: " + "; ".join(faults)
        raise SystemExit(message)
    print(f"cell 2: held; user={report.user} distributions={len(report.distributions)}")


def require_networks_absent(names: list[str]) -> None:
    """A network of that name belongs to someone else, and a start would join it."""
    existing = docker("network", "ls", "--format", "{{.Name}}").split()
    present = sorted(set(names) & set(existing))
    if present:
        message = f"the host already has {present}; the proof will not touch them"
        raise SystemExit(message)


def cell_absent_networks(plain: Compose, names: list[str]) -> None:
    """Cell 4: without its external networks the start fails loud and nothing runs."""
    started = plain.run("up", "--detach", "--no-build", check=False)
    refused = started.returncode != 0 and any(n in started.stderr for n in names)
    if not refused or plain.run("ps", "--all", "--quiet").stdout.strip():
        message = (
            f"compose up did not refuse the absent networks: {started.stderr[-300:]}"
        )
        raise SystemExit(message)
    print("cell 4: held; compose up refused the absent networks")


def cell_start(session: Session) -> None:
    """Cell 3: the stack starts healthy, read-only and non-root, and writes nothing."""
    compose = session.compose
    compose.up()
    container = container_of(compose, SERVICE)
    inspected = json.loads(docker("inspect", container))[0]
    user = compose.run("exec", "-T", SERVICE, "id", "-u").stdout.strip()
    ready = session.health()["speaking"]["ready"]
    writable = can_write_root(compose)
    if not (
        inspected["State"]["Health"]["Status"] == "healthy"
        and inspected["HostConfig"]["ReadonlyRootfs"] is True
        and user != "0"
        and ready is True
        and not writable
    ):
        message = f"start is not as shipped: user={user} ready={ready} {writable=}"
        raise SystemExit(message)
    print(f"cell 3: held; healthy read_only user={user} ready={ready} root_writable=no")


def can_write_root(compose: Compose) -> bool:
    """Whether the service's user can create a file in the root directory."""
    attempt = compose.run(
        "exec", "-T", SERVICE, "python", "-c", "open('/probe', 'w')", check=False
    )
    return attempt.returncode == 0


def speak_marked(session: Session) -> bytes:
    """One admitted speak; the WAV it returns, which must carry the AI mark."""
    reply = session.speak(session.admitted_headers(), session.baked.voice)
    if reply.status != HTTPStatus.OK:
        message = f"the admitted speak answered {reply.status}"
        raise SystemExit(message)
    require_marked_wav(reply.body)
    return reply.body


def cell_speak(session: Session) -> bytes:
    """Cell 5: one admitted speak returns a marked WAV."""
    wav = speak_marked(session)
    print(f"cell 5: held; status=200 wav_bytes={len(wav)} mark=present")
    return wav


def cell_refusals(session: Session) -> None:
    """Cell 6: no token, a token outside the group and a foreign voice are refused."""
    outside = replace(
        minting_for(session.config, session.baked, session.signing),
        service_group=f"{session.config.identity.service_group}-outside",
    )
    pair = session.pair(outside)
    cases = [
        ("no token", {}, session.baked.voice, RefusalCode.UNAUTHORIZED),
        (
            "a token outside the group",
            speech_request_headers(
                workload_token=pair.workload, person_token=pair.person
            ),
            session.baked.voice,
            RefusalCode.FORBIDDEN,
        ),
        (
            "a voice outside the table",
            session.admitted_headers(),
            session.baked.other_voice,
            RefusalCode.FORBIDDEN,
        ),
    ]
    for what, headers, voice, code in cases:
        require_refusal(session.speak(headers, voice), code)
        print(f"cell 6: held; {what} refused as {code}")
    session.collect_log()


def cell_key_server_stopped(session: Session) -> None:
    """Cell 7: with the key server stopped and no cached key, the call is refused."""
    compose = session.compose
    compose.run("stop", "--timeout", str(STOP_TIMEOUT_SECONDS), KEY_SERVER)
    compose.up(SERVICE, recreate=True)
    reply = session.speak(session.admitted_headers(), session.baked.voice)
    require_refusal(reply, RefusalCode.UNAVAILABLE)
    if session.health()["speaking"]["ready"] is not True:
        message = "/health does not stay honest while the key server is away"
        raise SystemExit(message)
    session.collect_log()
    compose.up(KEY_SERVER)
    print("cell 7: held; refused as unavailable, no audio, health ready")


def cell_restart(session: Session) -> None:
    """Cell 9: a restart with no way to a registry or hub ends healthy and unchanged."""
    compose = session.compose
    compose.run("restart", "--timeout", str(STOP_TIMEOUT_SECONDS), SERVICE)
    compose.up(SERVICE)
    speak_marked(session)
    if can_write_root(compose):
        message = "the restarted service can write outside tmpfs"
        raise SystemExit(message)
    print("cell 9: held; healthy again, speaks, root_writable=no")


def cell_no_egress(session: Session) -> None:
    """Cell 8: from inside the service only the key server answers."""
    key_server = f"{session.baked.key_set_host}:{session.baked.key_set_port}"
    output = session.compose.run(
        "exec", "-T", SERVICE, "python", "-", key_server, *PUBLIC_ADDRESSES,
        stdin=(FIXTURES / "reach.py").read_text(),
    ).stdout  # fmt: skip
    reached = json.loads(output)
    outside = [
        address
        for address, answered in reached.items()
        if answered and address != key_server
    ]
    if not reached[key_server] or outside:
        message = f"reach is not the key server alone: {reached}"
        raise SystemExit(message)
    print(f"cell 8: held; only {key_server} answers")


def cell_negatives(session: Session, wav: bytes) -> None:
    """Cell 10: a loopback-only bind and an unmarked WAV each fail the proof."""
    require_fails(
        lambda: require_marked_wav(without_mark(wav)), "a WAV without the mark"
    )
    session.collect_log()
    loopback = replace(
        session.compose, files=(*session.compose.files, LOOPBACK_OVERRIDE)
    )
    loopback.up(SERVICE, recreate=True)
    require_fails(
        lambda: speak_marked(replace(session, compose=loopback)), "a loopback-only bind"
    )
    print("cell 10: held; both negatives failed the proof")


def cell_log(session: Session) -> None:
    """Cell 11: the logs hold no spoken text and no token."""
    session.collect_log()
    faults = log_faults("\n".join(session.logs), [SENTENCE], session.tokens)
    if faults:
        message = "the service log holds " + ", ".join(faults)
        raise SystemExit(message)
    print("cell 11: held; the log holds no spoken text and no token")


def _certificate(
    subject: str,
    public_key: ec.EllipticCurvePublicKey,
    signer: ec.EllipticCurvePrivateKey,
    *,
    host: str | None,
) -> x509.Certificate:
    """A CA certificate when `host` is None, else a server certificate for `host`."""
    now = dt.datetime.now(tz=dt.UTC)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)])
    authority = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, CA_NAME)])
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(authority)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False
        )
        .add_extension(
            x509.BasicConstraints(ca=host is None, path_length=None), critical=True
        )
    )
    if host is None:
        builder = builder.add_extension(CA_KEY_USAGE, critical=True)
    else:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(host)]), critical=False
        ).add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(signer.public_key()),
            critical=False,
        )
    return builder.sign(signer, hashes.SHA256())


def issue_certificates(directory: Path, host: str, key_set: bytes) -> None:
    """Write a test CA, the key server's certificate for `host`, and the key set."""
    authority_key = ec.generate_private_key(ec.SECP256R1())
    server_key = ec.generate_private_key(ec.SECP256R1())
    authority = _certificate(
        CA_NAME, authority_key.public_key(), authority_key, host=None
    )
    server = _certificate(host, server_key.public_key(), authority_key, host=host)
    pem = serialization.Encoding.PEM
    files = {
        directory / "ca" / "root_ca.crt": authority.public_bytes(pem),
        directory / "key-server" / "tls.crt": server.public_bytes(pem),
        directory / "key-server" / "tls.key": server_key.private_bytes(
            pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ),
        directory / "key-server" / "key-set.json": key_set,
    }
    for path, content in files.items():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        # The container's non-root user reads these throwaway files.
        path.chmod(0o644)
        path.parent.chmod(0o755)


def fetch_key_set(url: str, trust: Path) -> bytes:
    """The key set the proof identity serves, read once from the host."""
    context = ssl.create_default_context(cafile=str(trust))
    with urllib.request.urlopen(url, context=context, timeout=10) as response:  # noqa: S310 - the proof's own https URL
        return response.read()


def teardown(compose: Compose, *, failed: bool) -> None:
    """Remove the proof's containers, networks, volumes and image."""
    if failed:
        tail = compose.run("logs", "--no-color", "--tail", "30", check=False)
        print(tail.stdout, tail.stderr, sep="\n")
    compose.run(
        "down",
        "--volumes",
        "--remove-orphans",
        "--timeout",
        str(STOP_TIMEOUT_SECONDS),
        check=False,
    )
    subprocess.run(  # noqa: S603 - docker with fixed arguments
        ["docker", "image", "rm", "--force", IMAGE],  # noqa: S607 - docker is the host's own client
        capture_output=True,
        check=False,
    )
    print("torn_down", PROJECT)


def _prove() -> None:
    wait_for_load()
    config = load_caller_config(BAKED_CONFIG)
    baked = baked_caller(config)
    directory = Path(tempfile.mkdtemp(prefix="speech-image-"))
    environment = {
        "PROOF_DIRECTORY": str(directory),
        "PROOF_KEY_SET_PATH": baked.key_set_path,
        "PROOF_KEY_SET_HOST": baked.key_set_host,
    }
    plain = Compose((COMPOSE_FILE,), {})
    compose = Compose((COMPOSE_FILE, PROOF_OVERRIDE), environment)
    failed = True
    try:
        names = external_networks(
            json.loads(plain.run("config", "--format", "json").stdout)
        )
        require_networks_absent(names)
        with proof_identity(directory, baked.voice.value) as identity:
            signing = load_minting(identity.minting_path)
            issue_certificates(
                directory,
                baked.key_set_host,
                fetch_key_set(identity.jwks_url, identity.ca_path),
            )
            session = Session(compose, config, baked, signing, [], [])
            cell_build(compose)
            cell_contents(plain)
            cell_absent_networks(plain, names)
            cell_start(session)
            wav = cell_speak(session)
            cell_refusals(session)
            cell_key_server_stopped(session)
            cell_restart(session)
            cell_no_egress(session)
            cell_negatives(session, wav)
            cell_log(session)
        print("PROOF_OK")
        failed = False
    finally:
        teardown(compose, failed=failed)
        shutil.rmtree(directory, ignore_errors=True)


def main() -> None:
    """Run the image proof, taking the probe-stack lock through the fleet form."""
    parse_arguments(sys.argv[1:])
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if LOCK_HELD_ENV not in os.environ:
        print("waiting_for_lock", LOCK_PATH)
        raise SystemExit(run_under_lock(sys.argv[1:], script=THIS_SCRIPT))
    prove_under_lock(os.environ, _prove)


if __name__ == "__main__":
    main()
