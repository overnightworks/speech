"""Fetch a Piper voice at its pinned revision into the voice cache."""

# ruff: noqa: T201 - this command prints where the voice went

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from huggingface_hub import hf_hub_download

from speech_runtime_service.config import load_settings
from speech_runtime_service.piper_voices import (
    PinnedVoice,
    fetch_voice_files,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def main(
    argv: Sequence[str] | None = None,
    download: Callable[..., str] = hf_hub_download,
) -> None:
    """Download the voice named in `argv` into `SPEECH_VOICE_CACHE`, once.

    `argv` defaults to the command line.
    """
    parser = argparse.ArgumentParser(
        prog="speech-runtime-fetch-voice",
        allow_abbrev=False,
        description="Fetch a Piper voice at its pinned revision so the "
        "service can load it from local files.",
    )
    parser.add_argument("voice", type=PinnedVoice, choices=list(PinnedVoice))
    voice = parser.parse_args(argv).voice
    print(fetch_voice_files(voice, load_settings().voice_cache, download))
