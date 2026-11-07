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
import threading
import time
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Final

from image_gpu_voice import cell_gpu_voice_not_offered, cell_gpu_voice_offered
from image_judgement import (
    ImageReport,
    TmpUsage,
    baked_caller,
    external_networks,
    heard_words,
    image_faults,
    is_speaking_alone,
    live_texts,
    locked_distributions,
    minting_for,
    require_fails,
    require_final,
    require_heard,
    require_hearing_ready,
    require_limit_answer,
    require_log_clean,
    require_marked_wav,
    require_ready_within,
    require_refusal,
    require_tmp_clean,
    require_transcript,
    without_mark,
)
from image_stack import (
    FIXTURES,
    HEARING_LANGUAGE,
    IMAGE,
    KEY_SERVER,
    PROJECT,
    REPOSITORY,
    SERVICE,
    STOP_TIMEOUT_SECONDS,
    Compose,
    LiveOutcome,
    Reply,
    Session,
    Teardown,
    container_of,
    docker,
    form_of_size,
    healthcheck_passes,
    seconds_until_healthy,
    service_config,
    speak_marked,
    start_with,
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
from speech_runtime_client.wire import (
    RefusalCode,
    refusal,
    speech_request_headers,
)
from speech_runtime_service.config import Settings, load_caller_config
from speech_runtime_service.piper_voices import PINS, PinnedVoice
from speech_runtime_service.voices import DEFAULT_SPEAKING_MODEL

THIS_SCRIPT: Final = Path(__file__).resolve()
COMPOSE_FILE: Final = REPOSITORY / "compose.yaml"
BAKED_CONFIG: Final = REPOSITORY / "deploy" / "speech.toml"
PROOF_OVERRIDE: Final = FIXTURES / "compose.proof.yaml"
LOOPBACK_OVERRIDE: Final = FIXTURES / "compose.loopback.yaml"
NO_MODEL_OVERRIDE: Final = FIXTURES / "compose.no-model.yaml"
MISSING_MODEL_OVERRIDE: Final = FIXTURES / "compose.missing-model.yaml"

# hal builds with a cap of 1800 s; a build past half of it is a finding.
BUILD_LIMIT_SECONDS: Final = 900
PUBLIC_ADDRESSES: Final = ("1.1.1.1:443", "huggingface.co:443")
FORBIDDEN_EXECUTABLES: Final = ("git", "uv", "gcc", "cc")
# What caller.py reports for a connection nothing listens on.
REFUSED_CONNECTION: Final = "ConnectionRefusedError"
# What live.py reports for an upgrade the service refused.
REFUSED_UPGRADE: Final = "InvalidStatus"
FOREIGN_LANGUAGE: Final = "fr"
# The setting a missing model is reported under, in the environment and the log.
HEARING_MODEL_SETTING: Final = (
    Settings.model_config["env_prefix"] + "hearing_model"
).upper()
# Recordings at the byte limit sent at once: one decodes, the waiting places fill,
# and the rest are told busy; more would only lengthen CI.
BURST_SIZE: Final = 5
ANSWER_WAIT_SECONDS: Final = 60


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
    """Cell 2: the image holds the locked set, the pinned voice and hearing model."""
    environment = service_config(plain)["environment"]
    pin = PINS[PinnedVoice(DEFAULT_SPEAKING_MODEL)]
    voice_hashes = {
        Path(file.path).name: file.sha256 for file in (pin.model, pin.metadata)
    }
    script = (FIXTURES / "inspect_image.py").read_text()
    output = docker(
        "run", "--rm", "--interactive", "--network", "none", "--entrypoint", "python",
        IMAGE, "-", environment["SPEECH_VOICE_CACHE"],
        environment["SPEECH_HUGGINGFACE_CACHE"], environment[HEARING_MODEL_SETTING],
        *FORBIDDEN_EXECUTABLES,
        stdin=script,
    )  # fmt: skip
    report = ImageReport(**json.loads(output))
    faults = image_faults(report, _locked_set(), voice_hashes)
    if faults:
        message = "the image is not as locked: " + "; ".join(faults)
        raise SystemExit(message)
    print(
        f"cell 2: held; user={report.user} distributions={len(report.distributions)}"
        " no_nvidia=yes hearing_model=pinned_revision"
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


def cell_start(session: Session) -> float:
    """Cell 3: the stack starts healthy, read-only and non-root, and writes nothing.

    Returns the seconds from the service's start to its first passing check.
    """
    compose = session.compose
    compose.up()
    seconds = seconds_until_healthy(compose)
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
    return seconds


def can_write_root(compose: Compose) -> bool:
    """Whether the service's user can create a file in the root directory."""
    attempt = compose.run(
        "exec", "-T", SERVICE, "python", "-c", "open('/probe', 'w')", check=False
    )
    return attempt.returncode == 0


def cell_speak(session: Session) -> bytes:
    """Cell 5: one admitted speak returns a marked WAV."""
    wav = speak_marked(session)
    print(f"cell 5: held; status=200 wav_bytes={len(wav)} mark=present")
    return wav


def cell_refusals(session: Session) -> None:
    """Cell 6: no token and a token outside the group are refused.

    Every app in the baked table lists every voice, so no voice is forbidden.
    """
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
    print(
        "cell 9: held; healthy again after"
        f" {seconds_until_healthy(compose):.0f}s, speaks, root_writable=no"
    )


def reached(session: Session, addresses: list[str]) -> dict[str, bool]:
    """Which of `addresses` the service container can open a connection to."""
    output = session.compose.run(
        "exec", "-T", SERVICE, "python", "-", *addresses,
        stdin=(FIXTURES / "reach.py").read_text(),
    ).stdout  # fmt: skip
    return json.loads(output)


def cell_no_egress(session: Session) -> None:
    """Cell 8: from the service the key server answers and public hosts do not."""
    key_server = f"{session.baked.key_set_host}:{session.baked.key_set_port}"
    reach = reached(session, [key_server, *PUBLIC_ADDRESSES])
    outside = [
        address
        for address, answered in reach.items()
        if answered and address != key_server
    ]
    if not reach[key_server] or outside:
        message = f"reach is not the key server alone: {reach}"
        raise SystemExit(message)
    print(
        f"cell 8: held; from the service {key_server} answers and "
        f"{', '.join(PUBLIC_ADDRESSES)} do not"
    )


def cell_hearing_ready(session: Session, plain: Compose, seconds: float) -> None:
    """Cell 13: hearing is ready, the healthcheck passes, and it did so in time.

    `seconds` is cell 3's measure of the first start.
    """
    service = service_config(plain)
    model = service["environment"][HEARING_MODEL_SETTING]
    passes = healthcheck_passes(session.compose, service["healthcheck"])
    require_hearing_ready(session.health(), model, healthcheck_passes=passes)
    start_period = service["healthcheck"]["start_period"]
    require_ready_within(seconds, start_period)
    print(
        f"cell 13: held; hearing.ready model={model} healthcheck=passes"
        f" healthy_after={seconds:.0f}s start_period={start_period}"
    )


def cell_recording(session: Session, wav: bytes) -> None:
    """Cell 14: every baked app gets text from one recording."""
    for app in sorted(session.config.apps):
        caller = session.as_app(app)
        reply = caller.hear(caller.admitted_headers(), wav, HEARING_LANGUAGE)
        text = require_transcript(reply, SENTENCE)
        session.heard.append(text)
        print(
            f"cell 14: held; {app} status=200"
            f" spoken_words_heard={len(heard_words(text, SENTENCE))}"
        )


def cell_hearing_refusals(session: Session, wav: bytes) -> None:
    """Cell 16: no token, a foreign group and a foreign language are refused."""
    admitted = session.admitted_headers()
    outside = replace(
        minting_for(session.config, session.baked, session.signing),
        service_group=f"{session.config.identity.service_group}-outside",
    )
    pair = session.pair(outside)
    cases = [
        ("no token", {}, HEARING_LANGUAGE, RefusalCode.UNAUTHORIZED),
        (
            "a token outside the group",
            speech_request_headers(
                workload_token=pair.workload, person_token=pair.person
            ),
            HEARING_LANGUAGE,
            RefusalCode.FORBIDDEN,
        ),
        (
            "a language outside de and en",
            admitted,
            FOREIGN_LANGUAGE,
            RefusalCode.INVALID_REQUEST,
        ),
    ]
    for what, headers, language, code in cases:
        reply = session.hear(headers, wav, language)
        require_refusal(reply, code)
        print(f"cell 16: held; {what} refused as {code}")


def cell_live_turn(session: Session, wav: bytes) -> None:
    """Cell 17: a live turn over the WebSocket ends in a final."""
    outcome = session.hear_live(session.admitted_headers(), wav)
    require_final(outcome, SENTENCE)
    session.heard.extend(live_texts(outcome))
    print(f"cell 17: held; final arrived after {len(outcome.partials)} partials")


def cell_recording_limit(session: Session, wav: bytes) -> None:
    """Cell 18: recordings at the size limit never become a 500 or a spool left over."""
    limit = Settings.model_fields["max_recording_bytes"].default
    at_limit_form = form_of_size(limit, HEARING_LANGUAGE)
    headers = session.admitted_headers()
    before = session.tmp_usage()
    at_limit = session.post_recording(headers, at_limit_form)
    require_limit_answer(at_limit)
    over = session.post_recording(headers, form_of_size(limit + 1, HEARING_LANGUAGE))
    require_refusal(over, RefusalCode.TOO_LARGE)
    leave_together = threading.Barrier(BURST_SIZE)

    def post_when_all_are_ready(_: int) -> Reply:
        leave_together.wait()
        return session.post_recording(headers, at_limit_form)

    with ThreadPoolExecutor(BURST_SIZE) as pool:
        burst = list(pool.map(post_when_all_are_ready, range(BURST_SIZE)))
    for reply in burst:
        require_limit_answer(reply, busy_allowed=True)
    after = session.tmp_usage()
    require_tmp_clean(before, after, limit)
    session.heard.append(
        require_transcript(session.hear(headers, wav, HEARING_LANGUAGE), SENTENCE)
    )
    answers = dict(sorted(Counter(reply.status for reply in burst).items()))
    print(
        f"cell 18: held; at the limit answered {at_limit.status}, one byte over"
        f" {over.status}, a burst of {BURST_SIZE} answered {answers},"
        f" tmp_used {before.used_bytes}B before and {after.used_bytes}B after"
        f" of {after.size_bytes // 2**20}MiB, the next recording heard"
    )


def cell_hearing_restart(session: Session, wav: bytes, plain: Compose) -> None:
    """Cell 19: a restart with no way to a hub ends with both ready and hearing."""
    compose = session.compose
    peak_before = session.memory_peak_mib()
    session.collect_log()
    compose.run("restart", "--timeout", str(STOP_TIMEOUT_SECONDS), SERVICE)
    compose.up(SERVICE)
    service = service_config(plain)
    require_hearing_ready(
        session.health(),
        service["environment"][HEARING_MODEL_SETTING],
        healthcheck_passes=healthcheck_passes(compose, service["healthcheck"]),
    )
    health = session.health()
    reply = session.hear(session.admitted_headers(), wav, HEARING_LANGUAGE)
    session.heard.append(require_transcript(reply, SENTENCE))
    if any(reached(session, [PUBLIC_ADDRESSES[1]]).values()) or can_write_root(compose):
        message = "after the restart the service can reach the hub or write its root"
        raise SystemExit(message)
    print(
        f"cell 19: held; healthy_after={seconds_until_healthy(compose):.0f}s"
        f" speaking_ready={health['speaking']['ready']}"
        f" hearing_ready={health['hearing']['ready']} heard, hub_reachable=no,"
        f" root_writable=no, memory_peak_mib before_restart={peak_before}"
        f" after_restart={session.memory_peak_mib()}"
    )


def wait_until_speaking_alone(session: Session) -> None:
    """Wait until the voice is ready and hearing is not, so hearing alone fails."""
    for _ in range(ANSWER_WAIT_SECONDS):
        try:
            if is_speaking_alone(session.health()):
                return
        except SystemExit:
            pass
        time.sleep(1)
    message = "the service never showed speaking ready with hearing not ready"
    raise SystemExit(message)


def cell_hearing_negatives(session: Session, wav: bytes) -> None:
    """Cell 20: each hearing check fails, for its own reason, on a wrong subject."""
    require_fails(
        lambda: require_heard("Guten Tag zusammen", SENTENCE),
        "a text that holds no word of the sentence",
        naming="no word",
    )
    wrong_code = refusal(RefusalCode.FORBIDDEN).model_dump_json().encode()
    require_fails(
        lambda: require_refusal(Reply(403, wrong_code), RefusalCode.UNAUTHORIZED),
        "a refusal with another code",
        naming="expected",
    )
    server_error = Reply(500, b"internal server error")
    too_large_without_field = Reply(
        413, refusal(RefusalCode.TOO_LARGE).model_dump_json().encode()
    )
    require_fails(
        lambda: require_limit_answer(server_error),
        "a 500 at the size limit",
        naming="no wire refusal",
    )
    require_fails(
        lambda: require_limit_answer(too_large_without_field),
        "the byte limit refusing a body of exactly the limit",
        naming="byte limit",
    )
    require_fails(
        lambda: require_tmp_clean(
            TmpUsage(used_bytes=0, size_bytes=2**40),
            TmpUsage(used_bytes=1, size_bytes=2**40),
            1,
        ),
        "a spool left in /tmp",
        naming="spool",
    )
    require_fails(
        lambda: require_final(session.hear_live({}, wav), SENTENCE),
        "a live turn without tokens",
        naming=REFUSED_UPGRADE,
    )
    require_hearing_model_negatives(session)
    print("cell 20: held; each check failed on the wrong subject it was shown")


def require_hearing_model_negatives(session: Session) -> None:
    """A compose file without the model and an image without it each fail."""
    plain = replace(session.compose, files=session.compose.files[:1])
    model = service_config(plain)["environment"][HEARING_MODEL_SETTING]
    without = start_with(session, NO_MODEL_OVERRIDE)
    wait_until_speaking_alone(session)
    passes = healthcheck_passes(without, service_config(without)["healthcheck"])
    require_fails(
        lambda: require_hearing_ready(
            session.health(), model, healthcheck_passes=passes
        ),
        "a compose file without the model",
        naming="hearing is not ready",
    )
    if passes:
        message = "the healthcheck passes without a hearing model"
        raise SystemExit(message)
    missing = start_with(session, MISSING_MODEL_OVERRIDE)
    require_fails(
        lambda: missing.up(SERVICE),
        "a hearing model missing from the image",
        naming="unhealthy",
    )
    if HEARING_MODEL_SETTING not in missing.run("logs", "--no-color", SERVICE).stdout:
        message = "the service did not stop for want of the hearing model"
        raise SystemExit(message)


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
    require_log_clean("\n".join(session.logs), [SENTENCE], [], session.tokens)
    print("cell 11: held; the log holds no spoken text and no token")


def cell_hearing_log(session: Session) -> None:
    """Cell 21: the log after the hearing cells holds no heard text either."""
    log = "\n".join(session.logs)
    require_log_clean(log, [SENTENCE], session.heard, session.tokens)
    require_fails(
        lambda: require_log_clean(
            f"{log}\n{session.heard[0]}", [SENTENCE], session.heard, session.tokens
        ),
        "a log with a heard text",
        naming="heard text",
    )
    final = session.heard[-1]
    partial_turn = LiveOutcome(final, (final[:-1],), None)
    require_fails(
        lambda: require_log_clean(
            f"{log}\n{partial_turn.partials[0]}",
            [SENTENCE],
            live_texts(partial_turn),
            session.tokens,
        ),
        "a log with a live partial that differs from the final",
        naming="heard text",
    )
    print(
        f"cell 21: held; the log of {len(session.logs)} runs holds no heard text"
        " (finals and live partials), no spoken text and no token"
    )


def print_teardown(outcome: Teardown) -> None:
    """Print the log tail of a failed proof, then what teardown left or `torn_down`."""
    if outcome.tail is not None:
        print(outcome.tail)
    if outcome.left:
        print("teardown left something:", *outcome.left, sep="\n", file=sys.stderr)
    else:
        print("torn_down", PROJECT)


def hearing_cells(
    session: Session, plain: Compose, wav: bytes, started_after: float
) -> None:
    """Cells 13 to 19: hearing in the image, after a start `started_after` long."""
    cell_hearing_ready(session, plain, started_after)
    cell_recording(session, wav)
    cell_hearing_refusals(session, wav)
    cell_live_turn(session, wav)
    cell_recording_limit(session, wav)
    cell_hearing_restart(session, wav, plain)


def _prove() -> None:
    wait_for_load()
    config = load_caller_config(BAKED_CONFIG)
    baked = baked_caller(config, min(config.apps))
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
                session = Session(compose, config, baked, signing, [], [], [])
                cell_build(compose)
                cell_contents(plain)
                cell_absent_networks(plain, names)
                started_after = cell_start(session)
                wav = cell_speak(session)
                cell_refusals(session)
                cell_gpu_voice_not_offered(session)
                cell_key_server_stopped(session)
                cell_restart(session)
                cell_no_egress(session)
                hearing_cells(session, plain, wav, started_after)
                cell_hearing_negatives(session, wav)
                cell_negatives(session, wav)
                cell_gpu_voice_offered(session, wav)
                cell_log(session)
                cell_hearing_log(session)
            failed = False
        finally:
            outcome = teardown(compose, failed=failed)
            print_teardown(outcome)
    finally:
        shutil.rmtree(directory)
    if outcome.left:
        message = "teardown left something: " + "; ".join(outcome.left)
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
