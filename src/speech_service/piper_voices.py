"""Fetching a Piper voice at its pin, and checking the bytes meant."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import hf_hub_download

from speech_service.voices import piper_voice_paths

if TYPE_CHECKING:
    from collections.abc import Callable

    from speech_service.catalogue import PinnedFile, PiperEntry

_CHUNK_BYTES = 1 << 20


class VoiceHashMismatchError(RuntimeError):
    """A downloaded voice file is not the pinned bytes."""


def sha256_of(path: Path) -> str:
    """The hex SHA-256 of the file at `path`."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _is_present(path: Path, pinned: PinnedFile) -> bool:
    return path.is_file() and sha256_of(path) == pinned.sha256


def fetch_voice_files(
    voice: PiperEntry,
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
    pin = voice.pin
    targets = list(
        zip(
            piper_voice_paths(directory, voice.id.value),
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
            if sha256_of(part) != file.sha256:
                message = (
                    f"{target.name} is not the pinned {file.sha256} "
                    f"(revision {pin.revision})"
                )
                raise VoiceHashMismatchError(message)
            staged.append((part, target))
        for part, target in staged:
            part.replace(target)
    return onnx
