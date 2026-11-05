"""What the image proof judges: pure checks on bytes, reports, logs and the lock."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

from packaging.markers import Marker

from hearing_figures import peak_over_baseline, words
from prove_speech import spoken_in_log
from speech_figures import VoiceFigures
from speech_runtime_client.wire import (
    REFUSAL_STATUS,
    Health,
    Refusal,
    RefusalCode,
    RequestField,
    RetryAfterError,
    Transcription,
    busy_wait,
)
from speech_runtime_service.pcm import AI_MARK, BYTES_PER_SAMPLE, pcm_from_wav
from speech_runtime_service.provider_process import ENDED_BEFORE_READY
from speech_runtime_service.voices import RESIDENT_VOICE, VoiceId

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from image_stack import LiveOutcome, Reply
    from proof_identity import Minting
    from speech_runtime_service.config import CallerConfig

FORBIDDEN_PATH_PARTS: Final = frozenset({".git", "tests", "providers"})
KEY_FILE_SUFFIXES: Final = (".pem", ".key")
KEY_FILE_PREFIXES: Final = (".env", "id_")
FORBIDDEN_DISTRIBUTION_PREFIX: Final = "nvidia-"
# torch's own words when a model is moved onto `cuda` without a card.
NO_CUDA_DEVICE: Final = "Found no NVIDIA driver on your system"
MARK_CHUNK: Final = b"LIST"
# Shorter words match by accident in any German or English text.
MIN_HEARD_WORD_LENGTH: Final = 5
DURATION_PART: Final = re.compile(r"(\d+)([hms])")
DURATION_SECONDS: Final = {"h": 3600, "m": 60, "s": 1}
# hal builds an app's image under this cap and kills a build that runs past it.
HAL_BUILD_CAP_SECONDS: Final = 1800
# hal's next build after one the cap killed takes the killed one's layers and
# build caches and fetches only what is missing, so a second build from the same
# caches must end well inside the cap. Half of it leaves room for what a killed
# build had not fetched and for this host's spread: cold builds took 1017 to
# 1512 s here.
REBUILD_LIMIT_SECONDS: Final = HAL_BUILD_CAP_SECONDS // 2
NVIDIA_RUNTIME: Final = "nvidia"


@dataclass(frozen=True)
class BakedCaller:
    """What `deploy/speech.toml` says about one caller the proof speaks as."""

    app: str
    hear: bool
    voice: VoiceId
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
    package_stray_files: list[str]
    package_files_with_private_keys: list[str]
    voice_hashes: dict[str, str]
    hearing_model_directory: str | None
    executables: dict[str, str | None]
    nvidia_directories: list[str]


def app_that(config: CallerConfig, *, hears: bool) -> str:
    """The first app, by name, whose baked entry does (or does not) allow hearing."""
    for app in sorted(config.apps):
        if config.apps[app].hear is hears:
            return app
    message = f"the baked table holds no app with hear={hears}"
    raise SystemExit(message)


@dataclass(frozen=True)
class TmpUsage:
    """The filesystem's own count of the service's /tmp: bytes in use, and its size."""

    used_bytes: int
    size_bytes: int


def baked_caller(config: CallerConfig, app: str) -> BakedCaller:
    """`app`, the resident voice it speaks and the key set's address, from the table."""
    if RESIDENT_VOICE not in config.apps[app].voices:
        message = f"the baked table does not let {app} name {RESIDENT_VOICE}"
        raise SystemExit(message)
    key_set = urlsplit(config.identity.jwks_url)
    assert key_set.hostname is not None  # noqa: S101 - the owner validated an https URL
    return BakedCaller(
        app=app,
        hear=config.apps[app].hear,
        voice=RESIDENT_VOICE,
        key_set_host=key_set.hostname,
        key_set_port=key_set.port or 443,
        key_set_path=key_set.path,
    )


def listed_gpu_voice(config: CallerConfig, app: str) -> VoiceId:
    """The first GPU voice, by id, that the baked table lets `app` name."""
    listed = sorted(config.apps[app].voices - {RESIDENT_VOICE})
    if not listed:
        message = f"the baked table lets {app} name no GPU voice"
        raise SystemExit(message)
    return listed[0]


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


def _audio_bytes(wav: bytes, data_start: int) -> int:
    """The bytes inside a `data` chunk's payload, by its declared length.

    The streaming length the service writes declares more than any answer
    holds, so the bytes present are all there are.
    """
    declared = int.from_bytes(wav[data_start + 4 : data_start + 8], "little")
    return min(declared, len(wav) - data_start - 8)


def require_marked_wav(wav: bytes) -> None:
    """Fail unless `wav` is a WAV that carries the AI mark and some audio."""
    if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        message = "the answer is not a WAV"
        raise SystemExit(message)
    if ai_mark_of(wav) != AI_MARK:
        message = "the WAV does not carry the AI mark"
        raise SystemExit(message)
    if not any(
        chunk_id == b"data" and _audio_bytes(wav, start) > 0
        for chunk_id, start, _ in _chunks(wav)
    ):
        message = "the WAV holds no audio"
        raise SystemExit(message)


def require_refusal(
    reply: Reply, expected: RefusalCode, param: RequestField | None = None
) -> None:
    """Fail unless `reply` is the wire refusal `expected`, with its status.

    With `param`, the refusal must also name that field.
    """
    try:
        refusal = Refusal.model_validate_json(reply.body)
    except ValueError:
        message = f"status {reply.status} is no wire refusal"
        raise SystemExit(message) from None
    code = refusal.error.code
    if code is not expected or reply.status != REFUSAL_STATUS[code]:
        message = f"status {reply.status} refused as {code}, expected {expected}"
        raise SystemExit(message)
    if param is not None and refusal.error.param != param:
        message = f"{code} named {refusal.error.param}, expected {param}"
        raise SystemExit(message)


def require_fails(action: Callable[[], object], what: str, *, naming: str) -> None:
    """Fail unless `action` ends the proof's way, with a SystemExit naming `naming`."""
    try:
        action()
    except SystemExit as stopped:
        if naming in str(stopped.code):
            return
        message = f"the proof failed on {what}, but not as {naming}: {stopped.code}"
        raise SystemExit(message) from None
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


def _unwanted_files(application_files: Iterable[str]) -> list[str]:
    """What the application tree holds that is a repository, test or key file."""
    return [
        f"holds {path}"
        for path in map(Path, application_files)
        if FORBIDDEN_PATH_PARTS & set(path.parts)
        or path.name.startswith(KEY_FILE_PREFIXES)
        or path.suffix in KEY_FILE_SUFFIXES
    ]


def image_faults(
    report: ImageReport, locked: set[str], voice_hashes: dict[str, str]
) -> list[str]:
    """What the image holds that it must not, or lacks; empty when it is as locked."""
    installed = {distribution_name(name) for name in report.distributions}
    faults = [f"missing {name}" for name in sorted(locked - installed)]
    faults += [f"extra {name}" for name in sorted(installed - locked)]
    faults += [
        f"holds {name}"
        for name in sorted(installed)
        if name.startswith(FORBIDDEN_DISTRIBUTION_PREFIX)
    ]
    faults += [f"holds {name}" for name in report.nvidia_directories]
    if report.user == 0:
        faults.append("runs as root")
    faults += _unwanted_files(report.application_files)
    faults += [
        f"holds a private key in {name}"
        for name in report.application_files_with_private_keys
    ]
    faults += [
        f"holds {name} among the packages" for name in report.package_stray_files
    ]
    faults += [
        f"holds a private key in {name}"
        for name in report.package_files_with_private_keys
    ]
    faults += [f"carries {name}" for name, found in report.executables.items() if found]
    if report.voice_hashes != voice_hashes:
        faults.append("the voice files are not the owner's pinned bytes")
    if report.hearing_model_directory is None:
        faults.append("the hearing model of the pinned revision is not in the image")
    return faults


def log_faults(
    log: str, spoken: Iterable[str], heard: Iterable[str], tokens: Iterable[str]
) -> list[str]:
    """The spoken texts, heard texts and tokens a service log holds."""
    leaked = [f"spoken text {text[:12]!r}" for text in spoken_in_log(log, spoken)]
    heard_leaks = spoken_in_log(log, heard)
    if heard_leaks:
        leaked.append(f"{len(heard_leaks)} heard text(s)")
    return leaked + ["a token" for token in tokens if token in log]


def require_log_clean(
    log: str, spoken: Iterable[str], heard: Iterable[str], tokens: Iterable[str]
) -> None:
    """Fail unless the service log holds no spoken text, heard text or token."""
    faults = log_faults(log, spoken, heard, tokens)
    if faults:
        message = "the service log holds " + ", ".join(faults)
        raise SystemExit(message)


def external_networks(
    compose_config: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> list[str]:
    """The names of the networks a rendered compose file takes from outside."""
    return [
        str(network["name"])
        for network in compose_config["networks"].values()
        if network.get("external")
    ]


def heard_words(text: str, sentence: str) -> list[str]:
    """The words of `sentence` that `text` holds, leaving out the short ones."""
    heard = set(words(text))
    return [
        word
        for word in words(sentence)
        if len(word) >= MIN_HEARD_WORD_LENGTH and word in heard
    ]


def require_heard(text: str, sentence: str) -> None:
    """Fail unless `text` holds a word of `sentence`."""
    if not heard_words(text, sentence):
        message = "the text holds no word of the spoken sentence"
        raise SystemExit(message)


def _transcribed(reply: Reply) -> str:
    """The text of a 200 transcription."""
    if reply.status != HTTPStatus.OK:
        message = f"the recording answered {reply.status}, not a transcription"
        raise SystemExit(message)
    try:
        return Transcription.model_validate_json(reply.body).text
    except ValueError:
        message = "the 200 answer holds no transcription"
        raise SystemExit(message) from None


def require_transcript(reply: Reply, sentence: str) -> str:
    """The text of a 200 transcription that holds a word of `sentence`."""
    text = _transcribed(reply)
    require_heard(text, sentence)
    return text


def require_limit_answer(reply: Reply, *, busy_allowed: bool = False) -> None:
    """Fail unless a recording at the byte limit was read through and answered.

    A body of exactly the limit passes the byte check, so the answer is a
    transcription or `too_large` naming the `file` (its length); `too_large`
    without a field is the byte check refusing it. In a burst `busy` with
    `Retry-After` is also an answer.
    """
    if reply.status == HTTPStatus.OK:
        _transcribed(reply)
    elif busy_allowed and reply.status == REFUSAL_STATUS[RefusalCode.BUSY]:
        require_refusal(reply, RefusalCode.BUSY)
        try:
            busy_wait(reply.status, reply.headers)
        except RetryAfterError as error:
            raise SystemExit(str(error)) from None
    else:
        require_refusal(reply, RefusalCode.TOO_LARGE)
        if Refusal.model_validate_json(reply.body).error.param != "file":
            message = "the byte limit refused a body of exactly the limit"
            raise SystemExit(message)


def require_tmp_clean(before: TmpUsage, after: TmpUsage, recording_bytes: int) -> None:
    """Fail unless /tmp could hold a recording at the limit and is back to `before`."""
    faults = []
    if after.size_bytes < recording_bytes:
        faults.append("the tmpfs is smaller than one recording at the limit")
    if after.used_bytes > before.used_bytes:
        faults.append("a recording's spool is left in /tmp")
    if faults:
        raise SystemExit("; ".join(faults))


def live_texts(outcome: LiveOutcome) -> list[str]:
    """The texts a live turn showed that the log must lack: partials, then the final.

    A partial shorter than a word of the sentence matches any log by accident.
    """
    shown = [
        partial
        for partial in outcome.partials
        if len(partial.strip()) >= MIN_HEARD_WORD_LENGTH
    ]
    return [*shown, *([outcome.final] if outcome.final else [])]


def require_final(outcome: LiveOutcome, sentence: str) -> None:
    """Fail unless a live turn ended in a final that holds a word of `sentence`."""
    if outcome.final is None:
        message = f"the live turn ended without a final: {outcome.error}"
        raise SystemExit(message)
    require_heard(outcome.final, sentence)


def is_speaking_alone(health: Mapping[str, Mapping[str, object]]) -> bool:
    """Whether `/health` shows the voice ready and hearing not."""
    return (
        health["speaking"]["ready"] is True and health["hearing"]["ready"] is not True
    )


def require_hearing_ready(
    health: Mapping[str, Mapping[str, object]], model: str, *, healthcheck_passes: bool
) -> None:
    """Fail unless hearing runs the configured model and the healthcheck says so."""
    hearing = health["hearing"]
    if hearing["ready"] is not True or hearing["model"] != model:
        message = f"hearing is not ready with {model}: {dict(hearing)}"
        raise SystemExit(message)
    if not healthcheck_passes:
        message = "the healthcheck does not pass with hearing ready"
        raise SystemExit(message)


def duration_seconds(text: str) -> int:
    """The seconds of a rendered compose duration such as `30s` or `1m30s`."""
    return sum(
        int(count) * DURATION_SECONDS[unit]
        for count, unit in DURATION_PART.findall(text)
    )


def seconds_to_healthy(
    started_at: str, health_log: Iterable[Mapping[str, Any]]
) -> float:
    """From a start to the first passing check that ended after it.

    `started_at` and the log entries are as `docker inspect` prints them; the log
    also holds the checks of earlier starts.
    """
    started = datetime.fromisoformat(started_at)
    for entry in health_log:
        ended = datetime.fromisoformat(entry["End"])
        if entry["ExitCode"] == 0 and ended > started:
            return (ended - started).total_seconds()
    message = "no healthcheck passed after the start"
    raise SystemExit(message)


def require_ready_within(seconds: float, start_period: str) -> None:
    """Fail unless the container was healthy before its `start_period` ended."""
    if seconds > duration_seconds(start_period):
        message = f"the service took {seconds:.0f} s to be healthy, past {start_period}"
        raise SystemExit(message)


@dataclass(frozen=True)
class GpuVoiceReport:
    """What `gpu_voice.py` found in the service container.

    `workers` are the pids of running GPU voice workers, `temporary` the names
    of the entries in its temporary directory.
    """

    workers: list[int]
    temporary: list[str]


def require_offer(health: Health, voices: Iterable[VoiceId]) -> None:
    """Fail unless `/health` offers exactly `voices`."""
    offered = set(health.speaking.voices)
    if offered != {voice.value for voice in voices}:
        message = f"/health offers {sorted(offered)}"
        raise SystemExit(message)


def require_gpu_voice_down(health: Health, voice: VoiceId) -> None:
    """Fail unless `voice` is not ready while speaking and hearing are."""
    voice_ready = health.speaking.voices[voice.value].ready
    if not (health.speaking.ready and health.hearing.ready) or voice_ready:
        message = (
            f"/health is not {voice} down beside a ready service: "
            f"speaking.ready={health.speaking.ready} "
            f"hearing.ready={health.hearing.ready} {voice}.ready={voice_ready}"
        )
        raise SystemExit(message)


def require_workers_found_no_cuda_device(log: str, count: int) -> None:
    """Fail unless `count` workers ended before ready, each on finding no CUDA device.

    Each record the service logs for such a worker carries the worker's own
    stderr tail, so the cause is the real worker's, not a probe's.
    """
    records = log.split(ENDED_BEFORE_READY)[1:]
    if len(records) != count:
        message = f"{len(records)} workers ended before ready, not {count}"
        raise SystemExit(message)
    for record in records:
        if NO_CUDA_DEVICE not in record:
            message = f"a worker ended before ready on another cause{record}"
            raise SystemExit(message)


def require_no_worker_residue(before: GpuVoiceReport, after: GpuVoiceReport) -> None:
    """Fail unless the temporary directory holds nothing it did not hold `before`."""
    left = sorted(set(after.temporary) - set(before.temporary))
    if left:
        message = f"reaped workers left in the temporary directory: {left}"
        raise SystemExit(message)


def workers_ended_before_ready(log: str) -> int:
    """How many provider workers a service log says ended before they were ready."""
    return log.count(ENDED_BEFORE_READY)


class Build(StrEnum):
    """A timed build: with no cache at all, or from the caches a cold one left."""

    COLD = "cold"
    REBUILD = "rebuild"


BUILD_LIMIT_SECONDS: Final = {
    Build.COLD: HAL_BUILD_CAP_SECONDS,
    Build.REBUILD: REBUILD_LIMIT_SECONDS,
}


def require_build_within(build: Build, seconds: float) -> None:
    """Fail unless the `build` took no longer than its limit."""
    limit = BUILD_LIMIT_SECONDS[build]
    if seconds > limit:
        message = f"the {build} build took {seconds:.0f} s, past its {limit} s"
        raise SystemExit(message)


def card_faults(card_listing: str, runtimes: Iterable[str]) -> list[str]:
    """Why Docker cannot give a container the card: no card, or no nvidia runtime.

    `card_listing` is `nvidia-smi --list-gpus` or its `unavailable:` note, and
    `runtimes` the runtimes `docker info` names.
    """
    faults = []
    if card_listing.startswith("unavailable:") or not card_listing.strip():
        faults.append(f"no card: {card_listing or 'nvidia-smi lists none'}")
    if NVIDIA_RUNTIME not in runtimes:
        faults.append(f"Docker names no {NVIDIA_RUNTIME} runtime")
    return faults


def compute_pids(listing: str) -> set[int]:
    """The pids `nvidia-smi --query-compute-apps=pid` lists; fails on its note."""
    if listing.startswith("unavailable:"):
        message = f"the card's compute processes cannot be read: {listing}"
        raise SystemExit(message)
    return {int(pid) for pid in listing.split()}


def audio_seconds(wav: bytes) -> float:
    """How long the WAV `wav` plays."""
    rate, pcm = pcm_from_wav(wav)
    return len(pcm) / BYTES_PER_SAMPLE / rate


def card_figures(
    first: Reply, again: Reply, baseline_mib: int, samples_mib: list[int]
) -> VoiceFigures:
    """A GPU voice's figures from its first request and the same request again.

    The first request waits for the load and the second does not, so the load is
    what the first one's first byte took beyond the second's. The card's peak
    is sampled during the first request, over its use before it.
    """
    return VoiceFigures(
        load_seconds=first.first_byte_seconds - again.first_byte_seconds,
        first_audio_seconds=again.first_byte_seconds,
        real_time_factor=audio_seconds(again.body) / again.seconds,
        peak_card_mib=peak_over_baseline(baseline_mib, samples_mib),
    )
