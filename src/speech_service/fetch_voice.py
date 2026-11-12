"""Fetch the voices `SPEECH_VOICES` offers, each at its pin."""

# ruff: noqa: T201 - this command prints where each voice went

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, assert_never

from huggingface_hub import hf_hub_download, snapshot_download

from speech_client.wire import VOICES, SpeakingModel, VoiceId
from speech_service.config import load_settings
from speech_service.piper_voices import fetch_voice_files
from speech_service.voices import fetch_qwen

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def main(
    argv: Sequence[str] | None = None,
    download: Callable[..., str] = hf_hub_download,
    download_snapshot: Callable[..., str] = snapshot_download,
) -> None:
    """Download every offered voice, once, and print where each went.

    A Piper voice goes to `SPEECH_VOICE_CACHE`, Qwen to `SPEECH_HUGGINGFACE_CACHE`.
    `argv` defaults to the command line and takes no argument.
    """
    parser = argparse.ArgumentParser(
        prog="speech-fetch-voice",
        allow_abbrev=False,
        description="Fetch the voices SPEECH_VOICES offers at their pinned "
        "revisions so the service can load them from local files.",
    )
    parser.parse_args(argv)
    settings = load_settings()
    for voice in (voice for voice in VoiceId if voice in settings.voices):
        match VOICES[voice].model:
            case SpeakingModel.PIPER:
                print(fetch_voice_files(voice, settings.voice_cache, download))
            case SpeakingModel.QWEN:
                print(fetch_qwen(settings.huggingface_cache, download_snapshot))
            case unreachable:
                assert_never(unreachable)
