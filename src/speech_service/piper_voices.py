"""The Piper voices the repository knows: where they live and which bytes are meant."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import hf_hub_download

from speech_client.wire import VoiceId
from speech_service.voices import piper_voice_paths

if TYPE_CHECKING:
    from collections.abc import Callable

_CHUNK_BYTES = 1 << 20
_REPOSITORY = "rhasspy/piper-voices"
_REVISION = "b4e358ffcb232591bf73a540e70f7de528e4174a"


@dataclass(frozen=True, slots=True)
class PinnedFile:
    """A file of a voice: its path in the repository and its SHA-256."""

    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class VoicePin:
    """The repository and revision a voice is fetched at, and its two files."""

    repository: str
    revision: str
    model: PinnedFile
    metadata: PinnedFile


PINS: dict[VoiceId, VoicePin] = {
    VoiceId.THORSTEN: VoicePin(
        _REPOSITORY,
        _REVISION,
        PinnedFile(
            "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
            "7e64762d8e5118bb578f2eea6207e1a35a8e0c30595010b666f983fc87bb7819",
        ),
        PinnedFile(
            "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json",
            "974adee790533adb273a1ac88f49027d2a1b8f0f2cf4905954a4791e79264e85",
        ),
    ),
    # Licence: "public domain" on the model card at the pinned revision
    # (en/en_US/norman/medium/MODEL_CARD); dataset LibriVox.
    VoiceId.NORMAN: VoicePin(
        _REPOSITORY,
        _REVISION,
        PinnedFile(
            "en/en_US/norman/medium/en_US-norman-medium.onnx",
            "b9739443232a80a59c7d18810dd856899bf16a7964725f5ab81ea49b1351cb71",
        ),
        PinnedFile(
            "en/en_US/norman/medium/en_US-norman-medium.onnx.json",
            "6c2db7f558a4a8deb9fe822583c1c5105f6c4e834dd0f9de8ad17a888ee9fe1d",
        ),
    ),
    # Licence: "public domain" on the model card at the pinned revision
    # (en/en_US/ljspeech/medium/MODEL_CARD); dataset LJ Speech.
    VoiceId.LJSPEECH: VoicePin(
        _REPOSITORY,
        _REVISION,
        PinnedFile(
            "en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx",
            "6f52a751e2349abe7a76735eb09dc1875298c77ea2342ffd2fef79ff81b87f22",
        ),
        PinnedFile(
            "en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json",
            "141d612cc0a95ed7efc1ca936b845c2364967f2e9217c5dbfcf69fc4d6c65860",
        ),
    ),
    # Licence: "public domain" on the model card at the pinned revision
    # (en/en_GB/cori/medium/MODEL_CARD); dataset LibriVox.
    VoiceId.CORI: VoicePin(
        _REPOSITORY,
        _REVISION,
        PinnedFile(
            "en/en_GB/cori/medium/en_GB-cori-medium.onnx",
            "1899f98e5fb8310154f3c2973f4b8a929ba7245e722b3d3a85680b833d95f10d",
        ),
        PinnedFile(
            "en/en_GB/cori/medium/en_GB-cori-medium.onnx.json",
            "e262c16d7f192f69d4edd6b4ef8a5915379e67495fcc402f1ab15eeb33da3d36",
        ),
    ),
}


class VoiceHashMismatchError(RuntimeError):
    """A downloaded voice file is not the pinned bytes."""


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _is_present(path: Path, pinned: PinnedFile) -> bool:
    return path.is_file() and _sha256_of(path) == pinned.sha256


def fetch_voice_files(
    voice: VoiceId,
    directory: Path,
    download: Callable[..., str] = hf_hub_download,
) -> Path:
    """Place the pinned files of `voice` in `directory`, and return its `.onnx` path.

    A file there that fails its hash is removed first. Both files are downloaded
    into a staging directory inside `directory`, with no token and no read of a
    user's cache, and moved in only after every hash matches, so a failure leaves
    nothing at a final path. A voice already there and matching is not fetched
    again. `download` is the Hugging Face file downloader.

    Raises `VoiceHashMismatchError` naming the first file whose hash differs.
    """
    pin = PINS[voice]
    targets = list(
        zip(
            piper_voice_paths(directory, voice.value),
            (pin.model, pin.metadata),
            strict=True,
        )
    )
    for target, file in targets:
        if target.is_file() and not _is_present(target, file):
            target.unlink()
    onnx = targets[0][0]
    if all(_is_present(target, file) for target, file in targets):
        return onnx
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory, prefix=".fetch-") as scratch:
        staged = []
        for target, file in targets:
            part = Path(
                download(
                    pin.repository,
                    file.path,
                    revision=pin.revision,
                    local_dir=scratch,
                    token=False,
                    force_download=True,
                )
            )
            if _sha256_of(part) != file.sha256:
                message = (
                    f"{target.name} is not the pinned {file.sha256} "
                    f"(revision {pin.revision})"
                )
                raise VoiceHashMismatchError(message)
            staged.append((part, target))
        for part, target in staged:
            part.replace(target)
    return onnx
