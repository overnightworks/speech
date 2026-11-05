"""Stage proof: start the service, drive health and speak, then the contract client.

Takes /tmp/probe-stack.lock, checks the 1-minute load, runs one heavy step at a
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
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

import httpx2

from proof_identity import (
    Minting,
    ProofIdentity,
    load_minting,
    mint_pair,
    proof_identity,
)
from speech_runtime_client.wire import (
    SPEECH_PATH,
    speech_request_body,
    speech_request_headers,
)
from speech_runtime_service.config import load_settings
from speech_runtime_service.cuda_libs import cuda_library_dirs
from speech_runtime_service.pcm import BYTES_PER_SAMPLE
from speech_runtime_service.voices import RESIDENT_VOICE, VoiceId, model_for_voice

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping
    from types import FrameType
    from typing import IO

SENTENCE = (
    "Die Kamera zeigt den Vortragenden, während im Hintergrund "
    "die nächste Folie automatisch vorbereitet wird."
)
SECOND = "Die nächste Folie bitte."
THIS_SCRIPT = Path(__file__).resolve()
SETPRIV = "setpriv"
LOCK_PATH = Path("/tmp/probe-stack.lock")  # noqa: S108 - the one machine-wide probe lock must keep this exact path whatever TMPDIR says
LOCK_HELD_ENV = "SPEECH_PROOF_HOLDS_PROBE_LOCK"
LOCK_DIRECTORY_PREFIX = "speech-lock-"
LOCK_WAIT_SECONDS = 3600
FLOCK_TIMED_OUT_STATUS = 1
FLOCK_COMMAND_UNAVAILABLE_STATUS = 69
# The cap and the kill grace are the numbers this script owns inside the fleet
# lock form; together they stay within 1830 s. The grace covers the teardown
# `speech_server` runs after TERM: terminate, wait up to 10 s, then kill and wait
# up to 5 s, so 15 s at worst, with the providers stopped through the server.
# The hearing proof adds the card holder it stops after its server: up to 10 s
# more, so 25 s at worst.
# The proof speaks the GPU voice SPEECH_GPU_VOICES names, else Piper. A CPU
# Piper run needs only a small fraction of the cap, and the
# server exits within the grace after the proof body ends.
CAP_SECONDS = 1800
KILL_GRACE_SECONDS = 30
MEASURE_REPETITIONS = 5
SPEAK_TIMEOUT_SECONDS = 120
CONCURRENT_EXTRA_WAIT_SECONDS = 8.0
type Speak = Callable[[str, str], tuple[bytes, float, float]]
CONTRACT_MARKERS = ("contract_health", "contract_speak")
LOGGER = logging.getLogger(__name__)
SILENCE_RMS = 0.01
MIN_SPEECH_SECONDS = 0.4
LOAD_CEILING = 18.0
READY_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class VoiceFigures:
    """How a voice spoke `SENTENCE`: its load, first audio, speed and card memory.

    `real_time_factor` is audio seconds over wall seconds, so below 1 is slower
    than real time; `peak_card_mib` is the card's peak above its use before the
    voice's first request.
    """

    load_seconds: float
    first_audio_seconds: float
    real_time_factor: float
    peak_card_mib: int


# Qwen on this host's RTX 3090, loaded and driven in process before the image
# carried it, medians of five runs: the P0 plan's G0 measurement.
QWEN_G0: Final = VoiceFigures(
    load_seconds=6.9,
    first_audio_seconds=8.44,
    real_time_factor=0.78,
    peak_card_mib=3714,
)
# Qwen does not stream, so its first byte comes after the whole sentence, and
# the first request also waits for the load. G0 measured neither the HTTP hop
# nor the container, so the limit is half as much again as G0's load and first
# audio together.
QWEN_FIRST_BYTE_LIMIT_S: Final = 1.5 * (
    QWEN_G0.load_seconds + QWEN_G0.first_audio_seconds
)
FIRST_BYTE_LIMIT_S: Final = {
    RESIDENT_VOICE: 0.5,
    VoiceId.QWEN: QWEN_FIRST_BYTE_LIMIT_S,
}
PIPER_CACHE = Path.home() / ".cache" / "piper"
LOOPBACK_HOST = "127.0.0.1"
PROVIDER_ROOT = Path(__file__).resolve().parents[1] / "providers"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _load_1min() -> float:
    return float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])


def wait_for_load(
    read_load: Callable[[], float] = _load_1min,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait for the 1-minute load to fall to the ceiling, inside the lock's cap."""
    while True:
        load = read_load()
        print(f"load_1min={load:.2f}")
        if load <= LOAD_CEILING:
            return
        print(f"load {load:.2f} is above {LOAD_CEILING}, waiting")
        sleep(15)


def fleet_command(arguments: list[str], script: Path = THIS_SCRIPT) -> list[str]:
    """`script` under the fleet's one form for taking the probe lock."""
    return [
        "flock",
        "-o",
        "-w",
        str(LOCK_WAIT_SECONDS),
        str(LOCK_PATH),
        SETPRIV,
        "--pdeathsig",
        "TERM",
        "timeout",
        "-k",
        str(KILL_GRACE_SECONDS),
        str(CAP_SECONDS),
        sys.executable,
        str(script),
        *arguments,
    ]


def run_under_lock(
    arguments: list[str],
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    script: Path = THIS_SCRIPT,
) -> int:
    """Run the proof under the lock; a lock still busy after the wait fails loudly."""
    with tempfile.TemporaryDirectory(prefix=LOCK_DIRECTORY_PREFIX) as directory:
        token = Path(directory) / "acquired"
        environment = {**os.environ, LOCK_HELD_ENV: str(token)}
        status = run(
            fleet_command(arguments, script), env=environment, check=False
        ).returncode
        lock_was_taken = token.exists()
    if status == FLOCK_TIMED_OUT_STATUS and not lock_was_taken:
        message = (
            f"probe lock {LOCK_PATH} is busy: still held after {LOCK_WAIT_SECONDS} s,"
            " so the proof did not start"
        )
        raise SystemExit(message)
    if status == FLOCK_COMMAND_UNAVAILABLE_STATUS and not lock_was_taken:
        message = (
            "flock could not run setpriv, which ties the proof to the lock;"
            " install util-linux setpriv, the proof does not run without it"
        )
        raise SystemExit(message)
    return status


def announce_lock_held(environment: Mapping[str, str]) -> None:
    """Tell the process that waited for the lock that its child holds it."""
    token = Path(environment[LOCK_HELD_ENV])
    if not token.parent.name.startswith(LOCK_DIRECTORY_PREFIX):
        message = (
            f"{LOCK_HELD_ENV} must name a file in a {LOCK_DIRECTORY_PREFIX} directory"
        )
        raise SystemExit(message)
    token.touch()
    print("lock_acquired", LOCK_PATH)


def prove_under_lock(environment: Mapping[str, str], prove: Callable[[], None]) -> None:
    """Announce the held lock, turn TERM into an exit, and run the proof body."""
    announce_lock_held(environment)
    exit_on_terminate()
    prove()


def _exit_on_signal(received: int, _frame: FrameType | None) -> None:
    raise SystemExit(128 + received)


def exit_on_terminate() -> None:
    """Turn the cap's TERM into an ordinary exit, so the teardown `finally` runs."""
    signal.signal(signal.SIGTERM, _exit_on_signal)


def get_json(base_url: str, target: str) -> dict[str, object]:
    """GET a path over the speech service and decode the JSON body."""
    with httpx2.Client(base_url=base_url) as client:
        return client.get(target, timeout=5).json()


def require_running(
    exit_status: Callable[[], int | None], last: dict[str, object] | None
) -> None:
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
    fetch: Callable[[str, str], dict[str, object]] = get_json,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Poll health until speaking is ready; end the stage on exit or timeout."""
    deadline = clock() + READY_TIMEOUT_SECONDS
    last: dict[str, object] | None = None
    while clock() < deadline:
        require_running(exit_status, last)
        try:
            last = fetch(base_url, "/health")
        except httpx2.HTTPError, json.JSONDecodeError:
            sleep(0.5)
            continue
        speaking = last["speaking"]
        if not isinstance(speaking, dict):
            sleep(0.5)
            continue
        print(
            "health",
            f"speaking_ready={speaking['ready']}",
            f"card_memory_mb={last['card_memory_mb']}",
        )
        if speaking["ready"] is True:
            return last
        sleep(0.5)
    detail = "" if last is None else f": {last}"
    message = f"speaking not ready within {READY_TIMEOUT_SECONDS}s{detail}"
    raise SystemExit(message)


def _pcm_from_wav(data: bytes) -> tuple[int, bytes]:
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        message = "speak did not return a WAV"
        raise SystemExit(message)
    rate = int.from_bytes(data[24:28], "little")
    position = 12
    while position + 8 <= len(data):
        chunk_id = data[position : position + 4]
        size = int.from_bytes(data[position + 4 : position + 8], "little")
        payload = position + 8
        if chunk_id == b"data":
            return rate, data[payload:]
        position = payload + size
    message = "WAV has no data chunk"
    raise SystemExit(message)


def _rms(pcm: bytes) -> float:
    if len(pcm) < BYTES_PER_SAMPLE:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)])
    total = sum(sample * sample for sample in samples)
    return (total / len(samples)) ** 0.5 / 32768.0


def speak_measured(
    base_url: str,
    text: str,
    voice: str,
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
            speech_request_body(text, voice),
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


def proof_voice(gpu_voices: frozenset[VoiceId]) -> VoiceId:
    """The voice a proof speaks: the GPU voice the offer names, else Piper."""
    return min(gpu_voices, default=RESIDENT_VOICE)


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
    health: dict[str, object]
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
        [str, Callable[[], int | None]], dict[str, object]
    ] = wait_for_speaking_ready,
) -> Iterator[RunningSpeech]:
    """Start speech, wait until it is ready, and stop it on the way out.

    The log must hold neither of the proof's own sentences nor any text the
    caller adds to `spoken_texts`. It goes to a file, so a service that logs more
    than a pipe holds is never blocked by the proof, which reads it after the stop.
    """
    with tempfile.TemporaryFile("w+", encoding="utf-8") as log_file:
        server = start(
            [sys.executable, "-m", "speech_runtime_service"],
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
    wait_ready: Callable[[str, Callable[[], int | None]], dict[str, object]],
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


def _require_speaking(health: dict[str, object], voice: VoiceId) -> None:
    speaking = health["speaking"]
    if not isinstance(speaking, dict):
        message = f"health speaking is not an object: {speaking!r}"
        raise SystemExit(message)
    offered = speaking["voices"].get(voice.value)
    if offered is None or offered["model"] != model_for_voice(voice):
        message = f"speaking.voices does not offer {voice.value!r}: {speaking!r}"
        raise SystemExit(message)


def _prove_one_speak(
    speak: Speak,
    text: str,
    voice: str,
    *,
    limit: float,
    label: str,
) -> tuple[bytes, float, float]:
    wav, ttfa, total = speak(text, voice)
    rate, pcm = _pcm_from_wav(wav)
    duration = (len(pcm) / BYTES_PER_SAMPLE) / rate
    rtf = duration / total if total else 0.0
    print(
        label,
        f"first_byte_s={ttfa:.3f}",
        f"total_s={total:.3f}",
        f"limit_s={limit:.1f}",
        f"duration_s={duration:.3f}",
        f"rtf={rtf:.2f}",
        f"rms={_rms(pcm):.4f}",
        f"rate={rate}",
        f"bytes={len(wav)}",
    )
    if _rms(pcm) < SILENCE_RMS:
        message = f"{label} produced silence"
        raise SystemExit(message)
    if duration < 1.0:
        message = f"{label} produced too little audio to be a sentence"
        raise SystemExit(message)
    if ttfa > limit:
        message = f"{label} first byte {ttfa:.3f}s exceeded the {limit:.1f}s limit"
        raise SystemExit(message)
    return wav, ttfa, rtf


def _prove_speak_medians(
    speak: Speak,
    text: str,
    voice: str,
    *,
    limit: float,
    label: str,
) -> tuple[bytes, float, float]:
    """Speak the same sentence MEASURE_REPETITIONS times; print and return medians."""
    wavs: list[bytes] = []
    ttfas: list[float] = []
    rtfs: list[float] = []
    for rep in range(1, MEASURE_REPETITIONS + 1):
        wav, ttfa, rtf = _prove_one_speak(
            speak,
            text,
            voice,
            limit=limit,
            label=f"{label}_rep{rep}",
        )
        wavs.append(wav)
        ttfas.append(ttfa)
        rtfs.append(rtf)
    median_ttfa = statistics.median(ttfas)
    median_rtf = statistics.median(rtfs)
    print(
        f"{label}_median",
        f"first_byte_s={median_ttfa:.3f}",
        f"rtf={median_rtf:.2f}",
        f"limit_s={limit:.1f}",
        f"n={MEASURE_REPETITIONS}",
    )
    return wavs[-1], median_ttfa, median_rtf


def _prove_speak(base_url: str, voice: VoiceId, minting: Minting) -> bytes:
    """Speak German MEASURE_REPETITIONS times.

    `_prove_one_speak` fails the run the moment any single repetition misses
    the first-byte limit; a passing run is the proof that every
    repetition held, not just the median.
    """
    german, _ttfa, _rtf = _prove_speak_medians(
        functools.partial(speak_measured, base_url, minting=minting),
        SENTENCE,
        voice.value,
        limit=FIRST_BYTE_LIMIT_S[voice],
        label="speak_wire_de",
    )
    return german


def _prove_concurrent_speak(base_url: str, voice: VoiceId, minting: Minting) -> None:
    limit = FIRST_BYTE_LIMIT_S[voice]
    results: dict[str, tuple[bytes, float, float]] = {}
    errors: list[str] = []

    def run(key: str, text: str) -> None:
        try:
            results[key] = speak_measured(base_url, text, voice.value, minting)
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
    for key, (wav, ttfa, total) in results.items():
        rate, pcm = _pcm_from_wav(wav)
        duration = (len(pcm) / BYTES_PER_SAMPLE) / rate
        print(
            f"speak_concurrent_{key}",
            f"first_byte_s={ttfa:.3f}",
            f"total_s={total:.3f}",
            f"duration_s={duration:.3f}",
            f"rms={_rms(pcm):.4f}",
            f"rate={rate}",
        )
        if _rms(pcm) < SILENCE_RMS:
            message = f"concurrent {key} produced silence"
            raise SystemExit(message)
        if duration < MIN_SPEECH_SECONDS:
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


def add_cuda_library_path(env: dict[str, str]) -> None:
    """Put the bundled CUDA libraries on `env`'s loader path."""
    cuda_dirs = ":".join(str(path) for path in cuda_library_dirs())
    if cuda_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cuda_dirs}:{existing}" if existing else cuda_dirs


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
    env["SPEECH_GPU_VOICES"] = "" if voice is RESIDENT_VOICE else voice.value
    if hearing_model is not None:
        env["SPEECH_HEARING_MODEL"] = hearing_model
    env["SPEECH_BUSY_WAIT_SECONDS"] = str(
        FIRST_BYTE_LIMIT_S[voice] + CONCURRENT_EXTRA_WAIT_SECONDS
    )
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    add_cuda_library_path(env)
    return env


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    """Read the command line before anything starts; `--help` and typos exit here."""
    parser = argparse.ArgumentParser(
        description="Stage proof of the speech service under the probe-stack lock.",
        allow_abbrev=False,
    )
    return parser.parse_args(arguments)


def _prove() -> None:
    wait_for_load()
    print("gpu_before", nvidia_memory())
    work_directory = Path(tempfile.mkdtemp(prefix="speech-"))
    port = free_local_port()
    base_url = f"http://{LOOPBACK_HOST}:{port}"
    voice = proof_voice(load_settings().gpu_voices)
    try:
        with proof_identity(work_directory, voice.value) as identity:
            minting = load_minting(identity.minting_path)
            env = speech_environment(port, voice, identity)
            with speech_server(base_url, env, work_directory) as running:
                print("health_ready", json.dumps(running.health, sort_keys=True))
                print("load_to_ready_s", f"{running.ready_seconds:.1f}")
                print(
                    "gpu_compute_pids",
                    _nvidia_compute_pids(),
                    f"speech_pid={running.process.pid}",
                )
                _require_speaking(running.health, voice)
                print("gpu_resident", nvidia_memory())
                print("piper_cache", PIPER_CACHE)
                german = _prove_speak(base_url, voice, minting)
                german_wav = (
                    Path(tempfile.mkdtemp(prefix="speech-proof-")) / "german.wav"
                )
                german_wav.write_bytes(german)
                print("german_wav", german_wav)
                _prove_concurrent_speak(base_url, voice, minting)
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


def main() -> None:
    """Run the stage proof, taking the probe-stack lock through the fleet form."""
    parse_arguments(sys.argv[1:])
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if LOCK_HELD_ENV not in os.environ:
        print("waiting_for_lock", LOCK_PATH)
        raise SystemExit(run_under_lock(sys.argv[1:]))
    prove_under_lock(os.environ, _prove)


if __name__ == "__main__":
    main()
