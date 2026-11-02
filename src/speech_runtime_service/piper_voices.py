"""The Piper voices the repository knows: where they live and which bytes are meant."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import httpx2

if TYPE_CHECKING:
    from collections.abc import Callable

_CHUNK_BYTES = 1 << 20
_REPOSITORY = "rhasspy/piper-voices"
_REVISION = "b4e358ffcb232591bf73a540e70f7de528e4174a"


class PinnedVoice(StrEnum):
    """The closed catalogue of Piper voices."""

    THORSTEN = "de_DE-thorsten-medium"
    LESSAC = "en_US-lessac-medium"


@dataclass(frozen=True, slots=True)
class PinnedFile:
    """A file of a voice: its path in the repository and its SHA-256."""

    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class VoicePin:
    """The repository and revision a voice is fetched at, and the hash of each file."""

    repository: str
    revision: str
    files: tuple[PinnedFile, ...]


PINS: dict[PinnedVoice, VoicePin] = {
    PinnedVoice.THORSTEN: VoicePin(
        _REPOSITORY,
        _REVISION,
        (
            PinnedFile(
                "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
                "7e64762d8e5118bb578f2eea6207e1a35a8e0c30595010b666f983fc87bb7819",
            ),
            PinnedFile(
                "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json",
                "974adee790533adb273a1ac88f49027d2a1b8f0f2cf4905954a4791e79264e85",
            ),
        ),
    ),
    PinnedVoice.LESSAC: VoicePin(
        _REPOSITORY,
        _REVISION,
        (
            PinnedFile(
                "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
                "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
            ),
            PinnedFile(
                "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
                "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
            ),
        ),
    ),
}


class VoiceHashMismatchError(RuntimeError):
    """A downloaded voice file is not the pinned bytes."""


def download_file(url: str, destination: Path) -> None:
    """Stream `url` into `destination`, failing on any HTTP error."""
    with (
        httpx2.stream("GET", url, follow_redirects=True) as response,
        destination.open("wb") as target,
    ):
        response.raise_for_status()
        for chunk in response.iter_bytes(_CHUNK_BYTES):
            target.write(chunk)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _is_present(path: Path, pinned: PinnedFile) -> bool:
    return path.is_file() and _sha256_of(path) == pinned.sha256


def fetch_voice_files(
    voice: PinnedVoice,
    directory: Path,
    download: Callable[[str, Path], None] = download_file,
) -> Path:
    """Place the pinned files of `voice` in `directory`, and return its `.onnx` path.

    Both files are downloaded beside their final paths and moved in only after every
    hash matches, so a failure leaves nothing at a final path. A voice already there
    and matching is not fetched again. `download` writes a url's bytes to a path.

    Raises `VoiceHashMismatchError` naming the first file whose hash differs.
    """
    pin = PINS[voice]
    targets = [(directory / Path(file.path).name, file) for file in pin.files]
    if all(_is_present(target, file) for target, file in targets):
        return targets[0][0]
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory, prefix=".fetch-") as scratch:
        staged = []
        for target, file in targets:
            url = f"https://huggingface.co/{pin.repository}/resolve/{pin.revision}/{file.path}"
            part = Path(scratch) / target.name
            download(url, part)
            if _sha256_of(part) != file.sha256:
                message = (
                    f"{target.name} is not the pinned {file.sha256} "
                    f"(revision {pin.revision})"
                )
                raise VoiceHashMismatchError(message)
            staged.append((part, target))
        for part, target in staged:
            part.replace(target)
    return targets[0][0]
