"""Stage proof: start the service, drive health and speak, then the contract client.

Drives itself under `probe-run`, which waits for the load, runs one heavy step at a
time, and leaves no process behind. Model weights stay in the Hugging Face and
Piper caches.
"""

# ruff: noqa: T201 - the proof prints its stage lines as the command line output

from __future__ import annotations

import argparse
import contextlib
import functools
import http.client
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

import speaking_figures as figures
from hearing_client import Service, hear_file
from hearing_figures import speech_only, to_wire_rate
from proof_identity import (
    Minting,
    ProofIdentity,
    load_minting,
    mint_pair,
    proof_identity,
)
from speech_client.health import fetch_health
from speech_client.wire import (
    SPEECH_PATH,
    VOICES,
    Health,
    Language,
    ResponseFormat,
    VoiceId,
    speech_request_body,
    speech_request_headers,
)
from speech_service.config import load_settings
from speech_service.pcm import pcm_from_wav
from speech_service.voices import PIPER_VOICES, loads_at_start

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
    from types import FrameType
    from typing import IO

    from speech_client.health import HealthOutcome
    from speech_service.hearing_models import HearingModel

SENTENCE = (
    "Die Kamera zeigt den Vortragenden, während im Hintergrund "
    "die nächste Folie automatisch vorbereitet wird."
)
SECOND = "Die nächste Folie bitte."
ENGLISH_SENTENCE = (
    "The camera shows the speaker, while in the background "
    "the next slide is being prepared automatically."
)
SENTENCES: Final = {Language.DE: SENTENCE, Language.EN: ENGLISH_SENTENCE}
THIS_SCRIPT = Path(__file__).resolve()
SETPRIV = "setpriv"
PROBE_RUN = "probe-run"
UNDER_PROBE_RUN_ENV = "SPEECH_PROOF_UNDER_PROBE_RUN"
# The cap and the kill grace are the numbers this script owns inside the fleet
# probe-run form; together they stay within 1830 s. The grace covers the teardown
# `speech_server` runs after TERM: terminate, wait up to 10 s, then kill and wait
# up to 5 s, so 15 s at worst, with the providers stopped through the server.
# The hearing proof adds the card holder it stops after its server: up to 10 s
# more, so 25 s at worst.
# The proof speaks the GPU voice SPEECH_VOICES names, else the first Piper voice.
# A CPU Piper run needs only a small fraction of the cap, and the
# server exits within the grace after the proof body ends.
CAP_SECONDS = 1800
KILL_GRACE_SECONDS = 30
SPEAK_TIMEOUT_SECONDS = 120
CONCURRENT_EXTRA_WAIT_SECONDS = 8.0
CONTRACT_MARKERS = ("contract_health", "contract_speak")
LOGGER = logging.getLogger(__name__)
READY_TIMEOUT_SECONDS = 300.0
HEALTH_DEADLINE_SECONDS = 5.0
HEALTH_CAP_BYTES = 64 * 1024
PIPER_FIRST_BYTE_LIMIT_S = 0.5
PIPER_PROOF_VOICE: Final = PIPER_VOICES[0]
PIPER_CACHE = Path.home() / ".cache" / "piper"
LOOPBACK_HOST = "127.0.0.1"
PROVIDER_ROOT = Path(__file__).resolve().parents[1] / "providers"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def fleet_command(arguments: list[str], script: Path = THIS_SCRIPT) -> list[str]:
    """`script` under the fleet's one form for driving a probe stack."""
    return [
        PROBE_RUN,
        "-k",
        str(KILL_GRACE_SECONDS),
        str(CAP_SECONDS),
        sys.executable,
        str(script),
        *arguments,
    ]


def run_under_probe_run(
    arguments: list[str],
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    script: Path = THIS_SCRIPT,
    which: Callable[[str], str | None] = shutil.which,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Run the proof under `probe-run`; without it the proof does not start."""
    if which(PROBE_RUN) is None:
        message = (
            f"{PROBE_RUN} is not on PATH; install the fleet's probe-run,"
            " the proof does not run without it"
        )
        raise SystemExit(message)
    child_environment = {**os.environ, **(environment or {}), UNDER_PROBE_RUN_ENV: "1"}
    command = fleet_command(arguments, script)
    return run(command, env=child_environment, check=False).returncode


def prove_under_probe_run(prove: Callable[[], None]) -> None:
    """Turn TERM into an exit and run the proof body `probe-run` started."""
    exit_on_terminate()
    prove()


def _exit_on_signal(received: int, _frame: FrameType | None) -> None:
    raise SystemExit(128 + received)


def exit_on_terminate() -> None:
    """Turn the cap's TERM into an ordinary exit, so the teardown `finally` runs."""
    signal.signal(signal.SIGTERM, _exit_on_signal)


def read_health(base_url: str) -> HealthOutcome:
    """Fetch the service's health within the proof's own deadline and cap."""
    return fetch_health(
        base_url,
        deadline_seconds=HEALTH_DEADLINE_SECONDS,
        cap_bytes=HEALTH_CAP_BYTES,
    )


def require_running(exit_status: Callable[[], int | None], last: Health | None) -> None:
    """End a wait at once when the server exited: a failed load exits it.

    Health has no failure state, and `unavailable` also shows during a normal
    load, so the exit of the process is the one final signal.
    """
    status = exit_status()
    if status is not None:
        detail = "" if last is None else f": {last}"
        message = f"the service exited with status {status} before it was ready{detail}"
        raise SystemExit(message)


def wait_for_speaking_ready(
    base_url: str,
    exit_status: Callable[[], int | None],
    fetch: Callable[[str], HealthOutcome] = read_health,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Health:
    """Poll health until speaking is ready; end the stage on exit or timeout."""
    deadline = clock() + READY_TIMEOUT_SECONDS
    last: Health | None = None
    while clock() < deadline:
        require_running(exit_status, last)
        answer = fetch(base_url)
        if isinstance(answer, Health):
            last = answer
            print(
                "health",
                f"speaking_ready={answer.speaking.ready}",
                f"card_memory_mb={answer.card_memory_mb}",
            )
            if answer.speaking.ready:
                return answer
        sleep(0.5)
    detail = "" if last is None else f": {last}"
    message = f"speaking not ready within {READY_TIMEOUT_SECONDS}s{detail}"
    raise SystemExit(message)


def speak_measured(
    base_url: str,
    text: str,
    voice: VoiceId,
    minting: Minting,
) -> tuple[bytes, float, float]:
    """Speak one request; return the whole WAV, time to first audio and total time.

    The headers leave before synthesis, so the first audio is the first body
    byte. A body the service aborted raises `IncompleteRead` in `http.client`,
    and the proof fails rather than keep the partial WAV.
    """
    pair = mint_pair(minting)
    address = urlsplit(base_url)
    connection = http.client.HTTPConnection(
        address.hostname, address.port, timeout=SPEAK_TIMEOUT_SECONDS
    )
    try:
        start = time.perf_counter()
        connection.request(
            "POST",
            SPEECH_PATH,
            speech_request_body(
                text, voice, ResponseFormat.WAV, VOICES[voice].languages[0]
            ),
            speech_request_headers(
                workload_token=pair.workload, person_token=pair.person
            ),
        )
        response = connection.getresponse()
        if response.status != http.client.OK:
            message = f"speak HTTP status: {response.status}"
            raise SystemExit(message)
        first = response.read(1)
        first_byte_at = time.perf_counter()
        body = first + response.read()
        end = time.perf_counter()
    except http.client.IncompleteRead:
        message = "speak body ended before it was complete"
        raise SystemExit(message) from None
    finally:
        connection.close()
    if not first:
        message = "speak returned no bytes"
        raise SystemExit(message)
    return body, first_byte_at - start, end - start


def proof_voice(offered: frozenset[VoiceId]) -> VoiceId:
    """The voice a proof speaks: the GPU voice the offer names, else a Piper voice."""
    return min(
        (voice for voice in offered if not loads_at_start(voice)),
        default=PIPER_PROOF_VOICE,
    )


def query_nvidia(*arguments: str) -> str:
    """Run `nvidia-smi` with `arguments`; a note starting `unavailable:` on failure."""
    nvidia = shutil.which("nvidia-smi")
    if nvidia is None:
        return "unavailable: nvidia-smi not found"
    try:
        completed = subprocess.run(  # noqa: S603 - argv is the resolved nvidia-smi and fixed arguments with no shell
            [nvidia, *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"
    return completed.stdout.strip()


def nvidia_memory() -> str:
    """The card's used and total memory as one line."""
    return query_nvidia("--query-gpu=memory.used,memory.total", "--format=csv,noheader")


def _nvidia_compute_pids() -> str:
    pids = query_nvidia("--query-compute-apps=pid", "--format=csv,noheader")
    if pids.startswith("unavailable:"):
        return pids
    return " ".join(pids.split()) or "none"


def peak_rss_mib(status: str) -> float:
    """Read `VmHWM`, the process's peak resident set, from /proc/<pid>/status."""
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    message = "no VmHWM in the process status"
    raise SystemExit(message)


@dataclass(frozen=True)
class RunningSpeech:
    """A started speech service: process, first health, texts it must not log."""

    process: subprocess.Popen[str]
    health: Health
    ready_seconds: float
    spoken_texts: list[str] = field(default_factory=list)

    def peak_rss_mib(self) -> float:
        """The service process's peak resident set in MiB."""
        status = Path(f"/proc/{self.process.pid}/status")
        return peak_rss_mib(status.read_text(encoding="utf-8"))


def start_with_parent_death(
    command: list[str], *, parent_pid: int | None = None, **options: object
) -> subprocess.Popen[str]:
    """Start `command` under `setpriv` so the kernel sends it TERM with its parent.

    The signal follows the thread that spawned the child, not the process: call
    this from the thread that lives as long as the proof, never from a worker
    thread, or the child dies when that worker returns. A proof killed outright
    then leaves no child holding the card.

    The kernel arms the signal only once `setpriv` runs, so a parent that died
    before that would leave the child alive: the shell the child starts as checks,
    with the signal armed, that its parent is still `parent_pid` (this process by
    default).
    """
    return subprocess.Popen(  # noqa: S603 - argv is setpriv with the caller's own command
        [
            SETPRIV,
            "--pdeathsig",
            "TERM",
            "sh",
            "-c",
            '[ "$PPID" = "$0" ] || exit 1; exec "$@"',
            str(os.getpid() if parent_pid is None else parent_pid),
            *command,
        ],
        **options,
    )


@contextlib.contextmanager
def speech_server(
    base_url: str,
    env: dict[str, str],
    work_directory: Path,
    start: Callable[..., subprocess.Popen[str]] = start_with_parent_death,
    wait_ready: Callable[
        [str, Callable[[], int | None]], Health
    ] = wait_for_speaking_ready,
) -> Iterator[RunningSpeech]:
    """Start speech, wait until it is ready, and stop it on the way out.

    The log must hold neither of the proof's own sentences nor any text the
    caller adds to `spoken_texts`. It goes to a file, so a service that logs more
    than a pipe holds is never blocked by the proof, which reads it after the stop.
    """
    with tempfile.TemporaryFile("w+", encoding="utf-8") as log_file:
        server = start(
            [sys.executable, "-m", "speech_service"],
            cwd=work_directory,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        yield from _run_and_stop(server, base_url, wait_ready, log_file)


def _run_and_stop(
    server: subprocess.Popen[str],
    base_url: str,
    wait_ready: Callable[[str, Callable[[], int | None]], Health],
    log_file: IO[str],
) -> Iterator[RunningSpeech]:
    proof_failed = True
    running: RunningSpeech | None = None
    try:
        started = time.monotonic()
        health = wait_ready(base_url, server.poll)
        running = RunningSpeech(server, health, time.monotonic() - started)
        yield running
        proof_failed = False
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
        log_file.seek(0)
        log = log_file.read()
        if log:
            print("server_log_tail:")
            print(log[-4000:])
        print("server_exit", server.returncode)
        texts = [SENTENCE, SECOND, *(running.spoken_texts if running else [])]
        if proof_failed:
            _report_log_text(log, texts)
        else:
            _require_log_without_text(log, texts)


def spoken_in_log(log: str, texts: Iterable[str]) -> list[str]:
    """The non-blank texts that the service log holds."""
    return [text for text in texts if text.strip() and text in log]


def _leak_message(log: str, texts: Iterable[str]) -> str | None:
    leaked = spoken_in_log(log, texts)
    if leaked:
        return f"the service log holds {len(leaked)} spoken text(s)"
    return None


def _report_log_text(log: str, texts: Iterable[str]) -> None:
    """Print a leak without hiding the failure that is already propagating."""
    message = _leak_message(log, texts)
    if message:
        print(message)


def _require_log_without_text(log: str, texts: Iterable[str]) -> None:
    message = _leak_message(log, texts)
    if message:
        raise SystemExit(message)


def _require_speaking(health: Health, voice: VoiceId) -> None:
    offered = health.speaking.voices.get(voice.value)
    if offered is None or offered.model is not VOICES[voice].model:
        message = f"speaking.voices does not offer {voice.value!r}: {health.speaking!r}"
        raise SystemExit(message)


def _prove_speak(base_url: str, voice: VoiceId, minting: Minting) -> bytes:
    """Speak German figures.MEASURE_REPETITIONS times.

    `prove_one_speak` fails the run the moment any single repetition misses
    the first-byte limit; a passing run is the proof that every
    repetition held, not just the median.
    """
    return figures.prove_speak_medians(
        functools.partial(speak_measured, base_url, minting=minting),
        SENTENCE,
        voice,
        limit=PIPER_FIRST_BYTE_LIMIT_S,
        label="speak_wire_de",
    ).wav


def _prove_concurrent_speak(base_url: str, voice: VoiceId, minting: Minting) -> None:
    limit = PIPER_FIRST_BYTE_LIMIT_S
    results: dict[str, tuple[bytes, float, float]] = {}
    errors: list[str] = []

    def run(key: str, text: str) -> None:
        try:
            results[key] = speak_measured(base_url, text, voice, minting)
        except Exception as error:
            LOGGER.exception("concurrent speak %s failed", key)
            errors.append(f"{key}: {error}")

    first = threading.Thread(target=run, args=("first", SENTENCE), daemon=True)
    second = threading.Thread(target=run, args=("second", SECOND), daemon=True)
    first.start()
    second.start()
    first.join()
    second.join()
    if errors:
        raise SystemExit("concurrent speak failed: " + "; ".join(errors))
    if set(results) != {"first", "second"}:
        message = f"concurrent speak missing a response: {sorted(results)}"
        raise SystemExit(message)
    for key, answer in results.items():
        spoken = figures.spoken_of(answer)
        ttfa = spoken.first_audio_seconds
        print(
            f"speak_concurrent_{key}",
            f"first_byte_s={ttfa:.3f}",
            f"total_s={spoken.total_seconds:.3f}",
            f"duration_s={spoken.audio_seconds:.3f}",
            f"rms={spoken.level:.4f}",
            f"rate={spoken.rate}",
        )
        if spoken.level < figures.SILENCE_RMS:
            message = f"concurrent {key} produced silence"
            raise SystemExit(message)
        if spoken.audio_seconds < figures.MIN_SPEECH_SECONDS:
            message = f"concurrent {key} produced too little audio"
            raise SystemExit(message)
        if ttfa > limit + CONCURRENT_EXTRA_WAIT_SECONDS:
            message = (
                f"concurrent {key} first byte {ttfa:.3f}s; "
                f"serialized limit is {limit}s plus "
                f"{CONCURRENT_EXTRA_WAIT_SECONDS}s wait"
            )
            raise SystemExit(message)
    if results["first"][0] == results["second"][0]:
        message = "concurrent speak returned identical bodies"
        raise SystemExit(message)


def _prove_beside(
    service: Service,
    singles: Mapping[VoiceId, figures.VoiceFigures],
    hearing_model: HearingModel | None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Two voices at once, then the hearing of one recording while they speak."""
    speak = functools.partial(speak_measured, service.base_url, minting=service.minting)
    pair = figures.voice_pair(PIPER_VOICES)
    together = figures.prove_two_voices(speak, pair, SENTENCES, singles)
    if hearing_model is None:
        print("hear_beside_speaking skipped: no hearing model named")
        return together, None
    if not service.running.health.hearing.ready:
        message = f"hearing model {hearing_model.value} did not load"
        raise SystemExit(message)
    wav, _first, _total = speak(SENTENCE, PIPER_PROOF_VOICE)
    rate, pcm = pcm_from_wav(wav)
    recording = speech_only(to_wire_rate(pcm, rate))
    hearing = figures.prove_hearing_beside_speaking(
        functools.partial(hear_file, service, Language.DE, recording),
        figures.speaking_tasks(speak, pair, SENTENCES),
        SENTENCE,
    )
    return together, hearing


def _prove_speaking(
    service: Service,
    voice: VoiceId,
    hearing_model: HearingModel | None,
    device: str,
) -> None:
    """Speak every Piper voice and the named one, then the steps that overlap them."""
    singles = figures.prove_each_voice(
        functools.partial(speak_measured, service.base_url, minting=service.minting),
        PIPER_VOICES,
        SENTENCES,
        limit=PIPER_FIRST_BYTE_LIMIT_S,
    )
    german = (
        singles[voice].wav
        if voice in singles
        else _prove_speak(service.base_url, voice, service.minting)
    )
    german_wav = Path(tempfile.mkdtemp(prefix="speech-proof-")) / "german.wav"
    german_wav.write_bytes(german)
    print("german_wav", german_wav)
    _prove_concurrent_speak(service.base_url, voice, service.minting)
    together, hearing = _prove_beside(service, singles, hearing_model)
    record = figures.speaking_record(device, singles, together, hearing)
    print("speaking_record", json.dumps(record, sort_keys=True))


def contract_client_command(base_url: str, voice: str, minting_path: Path) -> list[str]:
    """The contract client's argv: the minting file's path, never a key or a token."""
    client = Path(__file__).with_name("contract_client.py")
    return [
        sys.executable,
        str(client),
        "--minting-file",
        str(minting_path),
        base_url,
        voice,
    ]


def _prove_contract(
    base_url: str,
    env: dict[str, str],
    voice: VoiceId,
    minting_path: Path,
    work_directory: Path,
) -> None:
    print("contract_client_start")
    completed = subprocess.run(  # noqa: S603 - argv is this interpreter and this repository's client script with no shell
        contract_client_command(base_url, voice.value, minting_path),
        check=False,
        cwd=work_directory,
        env=env,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        message = f"contract client failed: {completed.returncode}"
        raise SystemExit(message)
    for marker in CONTRACT_MARKERS:
        if marker not in completed.stdout:
            message = f"contract client did not report {marker}"
            raise SystemExit(message)


def free_local_port() -> int:
    """A TCP port on 127.0.0.1 that nothing listens on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK_HOST, 0))
        return probe.getsockname()[1]


def offer_setting(offered: Collection[VoiceId]) -> str:
    """The `SPEECH_VOICES` value that offers `offered`, in catalogue order."""
    return ",".join(name.value for name in VoiceId if name in offered)


def speech_environment(
    port: int,
    voice: VoiceId,
    identity: ProofIdentity,
    device: str | None = None,
    hearing_model: str | None = None,
) -> dict[str, str]:
    """Return the process environment binding the loopback port, offering `voice`.

    `device` defaults to `SPEECH_DEVICE` from the caller's environment, else
    `cuda`; `hearing_model` is left unset when absent.
    """
    env = os.environ.copy()
    env["SPEECH_CONFIG_FILE"] = str(identity.configuration_path)
    env["SSL_CERT_FILE"] = str(identity.ca_path)
    env["SPEECH_HOST"] = LOOPBACK_HOST
    env["SPEECH_PORT"] = str(port)
    env.setdefault("SPEECH_PROVIDER_ROOT", str(PROVIDER_ROOT))
    env["SPEECH_DEVICE"] = device or os.environ.get("SPEECH_DEVICE", "cuda")
    if env["SPEECH_DEVICE"] == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["SPEECH_VOICES"] = offer_setting({PIPER_PROOF_VOICE, voice})
    if hearing_model is not None:
        env["SPEECH_HEARING_MODEL"] = hearing_model
    env["SPEECH_BUSY_WAIT_SECONDS"] = str(
        PIPER_FIRST_BYTE_LIMIT_S + CONCURRENT_EXTRA_WAIT_SECONDS
    )
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    """Read the command line before anything starts; `--help` and typos exit here."""
    parser = argparse.ArgumentParser(
        description="Stage proof of the speech service under probe-run.",
        allow_abbrev=False,
    )
    return parser.parse_args(arguments)


def _prove() -> None:
    print("gpu_before", nvidia_memory())
    work_directory = Path(tempfile.mkdtemp(prefix="speech-"))
    port = free_local_port()
    base_url = f"http://{LOOPBACK_HOST}:{port}"
    settings = load_settings()
    voice = proof_voice(settings.voices)
    try:
        with proof_identity(work_directory) as identity:
            minting = load_minting(identity.minting_path)
            env = speech_environment(port, voice, identity)
            env["SPEECH_VOICES"] = offer_setting({*PIPER_VOICES, voice})
            with speech_server(base_url, env, work_directory) as running:
                print(
                    "health_ready",
                    json.dumps(running.health.model_dump(), sort_keys=True),
                )
                print("load_to_ready_s", f"{running.ready_seconds:.1f}")
                print(
                    "gpu_compute_pids",
                    _nvidia_compute_pids(),
                    f"speech_pid={running.process.pid}",
                )
                running.spoken_texts.append(ENGLISH_SENTENCE)
                _require_speaking(running.health, voice)
                print("gpu_resident", nvidia_memory())
                print("piper_cache", PIPER_CACHE)
                _prove_speaking(
                    Service(base_url, minting, running),
                    voice,
                    settings.hearing_model,
                    env["SPEECH_DEVICE"],
                )
                _prove_contract(
                    base_url, env, voice, identity.minting_path, work_directory
                )
                print("peak_rss_mib", f"{running.peak_rss_mib():.0f}")
        print("gpu_after", nvidia_memory())
        print("caches_kept")
        print("  huggingface:", HF_CACHE)
        print("  piper:", PIPER_CACHE)
        print("PROOF_OK")
    finally:
        shutil.rmtree(work_directory)
        print("gpu_released", nvidia_memory())


def enter_proof(
    reexec: Callable[[list[str]], int],
    prove: Callable[[], None],
) -> None:
    """Read the command line; run `reexec` outside `probe-run`, `prove` inside it."""
    arguments = sys.argv[1:]
    parse_arguments(arguments)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if UNDER_PROBE_RUN_ENV not in os.environ:
        raise SystemExit(reexec(arguments))
    prove_under_probe_run(prove)


def main() -> None:
    """Run the stage proof, driving it through the fleet's `probe-run` form."""
    enter_proof(run_under_probe_run, _prove)


if __name__ == "__main__":
    main()
