"""Fetch a voice at its pin: a Piper voice by its name, a GPU voice by its id."""

# ruff: noqa: T201 - this command prints where the voice went

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from huggingface_hub import hf_hub_download, snapshot_download

from speech_service.config import load_settings
from speech_service.piper_voices import (
    PinnedVoice,
    fetch_voice_files,
)
from speech_service.voices import VoiceId, fetch_qwen

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def main(
    argv: Sequence[str] | None = None,
    download: Callable[..., str] = hf_hub_download,
    download_snapshot: Callable[..., str] = snapshot_download,
) -> None:
    """Download the voice named in `argv`, once.

    A Piper voice goes to `SPEECH_VOICE_CACHE`, Qwen to `SPEECH_HUGGINGFACE_CACHE`.
    `argv` defaults to the command line.
    """
    parser = argparse.ArgumentParser(
        prog="speech-fetch-voice",
        allow_abbrev=False,
        description="Fetch a Piper voice by its name, or a GPU voice by its id, at "
        "its pinned revision so the service can load it from local files.",
    )
    parser.add_argument("voice", choices=[*PinnedVoice, VoiceId.QWEN])
    voice = parser.parse_args(argv).voice
    settings = load_settings()
    if voice == VoiceId.QWEN:
        print(fetch_qwen(settings.huggingface_cache, download_snapshot))
    else:
        print(fetch_voice_files(PinnedVoice(voice), settings.voice_cache, download))
