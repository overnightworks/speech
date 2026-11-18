"""Render candidate voices from a text description with the pinned VoiceDesign model."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import wave
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

from speech_provider_contract import QWEN_ARTIFACTS

from speech_qwen.server import Server, start_server

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
"""Apache-2.0 per its model card; about 4.5 GB, the speaking model's file set."""
DESIGN_REVISION = "5ecdb67327fd37bb2e042aab12ff7391903235d3"
REFERENCE_SENTENCES = {
    "de": (
        "Guten Tag, ich freue mich, dich kennenzulernen. "
        "Heute erzähle ich dir etwas Schönes."
    ),
    "en": (
        "Good day, I am glad to meet you. "
        "Today I would like to tell you something lovely."
    ),
}
"""The fixed sentence every candidate of a language speaks, so candidates compare."""
_RESPONSE_SECONDS = 600.0
_STOP_GRACE_SECONDS = 30.0


class DesignVoiceError(RuntimeError):
    """The server refused a design request or answered with no audio."""


def _start_until_stopped(command: Sequence[str], socket_path: Path) -> Server:
    """Start the server; its end is `design_voices`' own doing, so nothing exits."""
    return start_server(command, socket_path, when_server_ends=lambda: None)


def _download_checkpoint(repo_id: str, revision: str, cache: Path) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        allow_patterns=QWEN_ARTIFACTS,
        cache_dir=cache,
    )


def design_voices(  # noqa: PLR0913 - one value per input of the script
    description: str,
    language: str,
    count: int,
    destination: Path,
    cache: Path,
    *,
    start: Callable[[Sequence[str], Path], Server] = _start_until_stopped,
    download: Callable[[str, str, Path], str] = _download_checkpoint,
) -> list[Path]:
    """Write `count` candidate WAVs of `description` and their transcripts.

    They go to `destination`.

    Each candidate is `candidate-NN.wav` with its exact spoken text in
    `candidate-NN.txt`. Nothing is written when any candidate fails, and an
    existing candidate is never overwritten.
    """
    sentence = REFERENCE_SENTENCES.get(language)
    if sentence is None:
        message = "unsupported design language"
        raise ValueError(message)
    audio_paths = [
        destination / f"candidate-{number:02d}.wav" for number in range(1, count + 1)
    ]
    destination.mkdir(parents=True, exist_ok=True)
    for path in audio_paths:
        if path.exists() or path.with_suffix(".txt").exists():
            raise FileExistsError(path)

    from speech_qwen.model import LANGUAGE_NAMES, start_on_checkpoint

    checkpoint = download(DESIGN_MODEL_ID, DESIGN_REVISION, cache)
    server = start_on_checkpoint(Path(checkpoint), start)
    with _stopped(server):
        audio_bodies = [
            _render(server, description, LANGUAGE_NAMES[language], sentence)
            for _ in audio_paths
        ]
    for path, body in zip(audio_paths, audio_bodies, strict=True):
        path.write_bytes(body)
        path.with_suffix(".txt").write_text(sentence, encoding="utf-8")
    return audio_paths


def _render(
    server: Server, description: str, language_name: str, sentence: str
) -> bytes:
    request = {
        "task_type": "VoiceDesign",
        "instructions": description,
        "input": sentence,
        "language": language_name,
        "response_format": "wav",
        "stream": False,
    }
    connection = server.connect(_RESPONSE_SECONDS)
    try:
        connection.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(request),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read()
        if response.status != HTTPStatus.OK:
            message = f"the server answered {response.status}"
            raise DesignVoiceError(message)
    finally:
        connection.close()
    _require_audio(body)
    return body


def _require_audio(body: bytes) -> None:
    try:
        with wave.open(io.BytesIO(body)) as reader:
            has_frames = reader.getnframes() > 0
    except (wave.Error, EOFError) as error:
        message = "the server's answer is no WAV"
        raise DesignVoiceError(message) from error
    if not has_frames:
        message = "the server's WAV holds no audio"
        raise DesignVoiceError(message)


@contextlib.contextmanager
def _stopped(server: Server) -> Iterator[None]:
    """Stop the server's process when the block ends."""
    try:
        yield
    finally:
        server.process.terminate()
        try:
            server.process.wait(timeout=_STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.process.wait()


def main() -> int:
    """Download the checkpoint, render the candidates and print their paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description", required=True, help="the voice, in words")
    parser.add_argument(
        "--language", choices=sorted(REFERENCE_SENTENCES), required=True
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="the Hub cache")
    arguments = parser.parse_args()
    for path in design_voices(
        arguments.description,
        arguments.language,
        arguments.count,
        arguments.destination,
        arguments.cache,
    ):
        print(path)  # noqa: T201 - the script's output is the written paths
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
