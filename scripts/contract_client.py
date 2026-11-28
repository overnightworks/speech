"""Drive health and the speak wire over TCP.

This process must not import the `speech_service` package. It is the
parallel-lane proof: a caller that only knows the HTTP contract, through the
types of `speech_client`.
"""

# ruff: noqa: T201 - the client prints its measurements as the command line output

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from proof_minting import Minting, load_minting, mint_pair
from speech_client import speak as speaking_call
from speech_client.health import fetch_health
from speech_client.wire import Health, Language, ResponseFormat, VoiceId
from wav_audio import SILENCE_RMS, pcm_from_wav, rms

SENTENCE = (
    "Die Kamera zeigt den Vortragenden, während im Hintergrund "
    "die nächste Folie automatisch vorbereitet wird."
)
HEALTH_DEADLINE_SECONDS = 30
HEALTH_CAP_BYTES = 64 * 1024
SPEAK_DEADLINE_SECONDS = 60
WAV_CAP_BYTES = 64 * 1024 * 1024


def _must_not_have_imported_speech() -> None:
    if "speech_service" in sys.modules:
        message = "contract client imported speech_service"
        raise SystemExit(message)


def speak(
    base_url: str, text: str, voice: VoiceId, language: Language, minting: Minting
) -> bytes:
    """POST text to the speech service with `voice` in `language`.

    Return the whole WAV bytes.

    Each call mints a fresh pair of tokens, so a long run never presents an
    expired one.
    """
    pair = mint_pair(minting)
    outcome = speaking_call.speak(
        base_url,
        text,
        voice,
        language=language,
        response_format=ResponseFormat.WAV,
        workload_token=pair.workload,
        person_token=pair.person,
        deadline_seconds=SPEAK_DEADLINE_SECONDS,
        cap_bytes=WAV_CAP_BYTES,
    )
    if not isinstance(outcome, speaking_call.Spoken):
        message = f"speak ended as {type(outcome).__name__}"
        raise SystemExit(message)
    return outcome.audio


def run(base_url: str, voice: VoiceId, minting: Minting) -> None:
    """Check health, speak the voice once and refuse a silent answer."""
    health = fetch_health(
        base_url,
        deadline_seconds=HEALTH_DEADLINE_SECONDS,
        cap_bytes=HEALTH_CAP_BYTES,
    )
    if not isinstance(health, Health):
        message = f"health ended as {type(health).__name__}"
        raise SystemExit(message)
    print("contract_health", json.dumps(health.model_dump(), sort_keys=True))
    if not health.speaking.ready:
        message = "the speaking model is not ready"
        raise SystemExit(message)

    language = health.speaking.voices[voice].languages[0]
    wav = speak(base_url, SENTENCE, voice, language, minting)
    try:
        rate, pcm = pcm_from_wav(wav)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    level = rms(pcm)
    print(
        "contract_speak",
        f"bytes={len(wav)} pcm={len(pcm)} rate={rate} rms={level:.4f}",
    )
    if level < SILENCE_RMS:
        message = "speak returned silence"
        raise SystemExit(message)


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    """Read the command line: the service URL, the voice and the minting file's path."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--minting-file", type=Path, required=True)
    parser.add_argument("base_url")
    parser.add_argument("voice", choices=[voice.value for voice in VoiceId])
    options = parser.parse_args(arguments)
    options.voice = VoiceId(options.voice)
    return options


def main() -> None:
    """Drive health and speak against the speech service URL."""
    _must_not_have_imported_speech()
    options = parse_arguments(sys.argv[1:])
    run(options.base_url, options.voice, load_minting(options.minting_file))


if __name__ == "__main__":
    main()
