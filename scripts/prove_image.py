"""Image proof: build the image and run the real compose file on isolated networks.

`tests/fixtures/image/compose.proof.yaml` is laid over `compose.yaml`: the
external networks become internal ones, a test CA takes the secret's place, and
a key-set server and a caller container join. The tokens carry the issuer,
audience, group and caller of the baked `deploy/speech.toml`. The proof takes
/tmp/probe-stack.lock like `prove_speech.py` and removes everything it made.
`image_judgement` holds what the proof judges and `image_stack` what it drives.
"""

# ruff: noqa: T201 - the proof prints its cell lines as the command line output

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from typing import Final

from image_judgement import (
    ImageReport,
    baked_caller,
    external_networks,
    image_faults,
    locked_distributions,
    log_faults,
    minting_for,
    require_fails,
    require_marked_wav,
    require_refusal,
    without_mark,
)
from image_stack import (
    FIXTURES,
    IMAGE,
    KEY_SERVER,
    REPOSITORY,
    SERVICE,
    STOP_TIMEOUT_SECONDS,
    Compose,
    Session,
    container_of,
    docker,
    teardown,
)
from proof_identity import (
    fetch_key_set,
    issue_certificates,
    load_minting,
    proof_identity,
)
from prove_speech import (
    LOCK_HELD_ENV,
    LOCK_PATH,
    SENTENCE,
    parse_arguments,
    prove_under_lock,
    run_under_lock,
    wait_for_load,
)
from speech_runtime_client.wire import RefusalCode, speech_request_headers
from speech_runtime_service.config import load_caller_config
from speech_runtime_service.piper_voices import PINS, PinnedVoice
from speech_runtime_service.voices import DEFAULT_SPEAKING_MODEL

THIS_SCRIPT: Final = Path(__file__).resolve()
COMPOSE_FILE: Final = REPOSITORY / "compose.yaml"
BAKED_CONFIG: Final = REPOSITORY / "deploy" / "speech.toml"
PROOF_OVERRIDE: Final = FIXTURES / "compose.proof.yaml"
LOOPBACK_OVERRIDE: Final = FIXTURES / "compose.loopback.yaml"

THIS_SCRIPT: Final = Path(__file__).resolve()
COMPOSE_FILE: Final = REPOSITORY / "compose.yaml"
BAKED_CONFIG: Final = REPOSITORY / "deploy" / "speech.toml"
PROOF_OVERRIDE: Final = FIXTURES / "compose.proof.yaml"
LOOPBACK_OVERRIDE: Final = FIXTURES / "compose.loopback.yaml"
# hal builds with a cap of 1800 s; a build past half of it is a finding.
BUILD_LIMIT_SECONDS: Final = 900
PUBLIC_ADDRESSES: Final = ("1.1.1.1:443", "huggingface.co:443")
FORBIDDEN_EXECUTABLES: Final = ("git", "uv", "gcc", "cc")
# What caller.py reports for a connection nothing listens on.
REFUSED_CONNECTION: Final = "ConnectionRefusedError"


def cell_build(compose: Compose) -> None:
    """Cell 1: a build with no layer cache, timed."""
    started = time.monotonic()
    compose.run("build", "--no-cache", SERVICE)
    seconds = time.monotonic() - started
    size = int(docker("image", "inspect", "--format", "{{.Size}}", IMAGE))
    print(
        f"cell 1: build_seconds={seconds:.0f} image_mib={size // 2**20}"
        " layer_cache=none (images already pulled on the host are still reused)"
    )
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
    print(
        f"cell 2: held; user={report.user} distributions={len(report.distributions)}"
        " no_nvidia=yes"
    )


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
    """Cell 8: from the service the key server answers and public hosts do not."""
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
    print(
        f"cell 8: held; from the service {key_server} answers and "
        f"{', '.join(PUBLIC_ADDRESSES)} do not"
    )


def cell_negatives(session: Session, wav: bytes) -> None:
    """Cell 10: a loopback-only bind and an unmarked WAV each fail the proof."""
    require_fails(
        lambda: require_marked_wav(without_mark(wav)),
        "a WAV without the mark",
        naming="AI mark",
    )
    session.collect_log()
    loopback = replace(
        session.compose, files=(*session.compose.files, LOOPBACK_OVERRIDE)
    )
    loopback.up(SERVICE, recreate=True)
    require_fails(
        lambda: speak_marked(replace(session, compose=loopback)),
        "a loopback-only bind",
        naming=REFUSED_CONNECTION,
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
            failed = False
        finally:
            left = teardown(compose, failed=failed)
    finally:
        shutil.rmtree(directory)
    if left:
        message = "teardown left something: " + "; ".join(left)
        raise SystemExit(message)
    print("PROOF_OK")


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
