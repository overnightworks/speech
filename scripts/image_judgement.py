"""What the image proof judges: pure checks on bytes, reports, logs and the lock."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from packaging.markers import Marker

from prove_speech import spoken_in_log
from speech_runtime_client.wire import REFUSAL_STATUS, Refusal, RefusalCode
from speech_runtime_service.pcm import AI_MARK
from speech_runtime_service.voices import VoiceId

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from image_stack import Reply
    from proof_identity import Minting
    from speech_runtime_service.config import CallerConfig

FORBIDDEN_PATH_PARTS: Final = frozenset({".git", "tests", "providers"})
KEY_FILE_SUFFIXES: Final = (".pem", ".key")
KEY_FILE_PREFIXES: Final = (".env", "id_")
FORBIDDEN_DISTRIBUTION_PREFIX: Final = "nvidia-"
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
    package_stray_files: list[str]
    package_files_with_private_keys: list[str]
    voice_hashes: dict[str, str]
    executables: dict[str, str | None]
    nvidia_directories: list[str]


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
