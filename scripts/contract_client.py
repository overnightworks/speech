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
from array import array
from pathlib import Path

from proof_identity import Minting, load_minting, mint_pair
from speech_client import speak as speaking_call
from speech_client.health import fetch_health
from speech_client.wire import Health, ResponseFormat, VoiceId

SENTENCE = (
    "Die Kamera zeigt den Vortragenden, während im Hintergrund "
    "die nächste Folie automatisch vorbereitet wird."
)
HEALTH_DEADLINE_SECONDS = 30
HEALTH_CAP_BYTES = 64 * 1024
SPEAK_DEADLINE_SECONDS = 60
WAV_CAP_BYTES = 64 * 1024 * 1024
BYTES_PER_SAMPLE = 2
SILENCE_RMS = 0.01


def _must_not_have_imported_speech() -> None:
    if "speech_service" in sys.modules:
        message = "contract client imported speech_service"
        raise SystemExit(message)


def _pcm_from_wav(data: bytes) -> tuple[int, bytes]:
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        message = "speak did not return a WAV"
        raise SystemExit(message)
    rate = int.from_bytes(data[24:28], "little")
    position = 12
    while position + 8 <= len(data):
        chunk_id = data[position : position + 4]
        size = int.from_bytes(data[position + 4 : position + 8], "little")
        payload = position + 8
        if chunk_id == b"data":
            return rate, data[payload:]
        position = payload + size
    message = "WAV has no data chunk"
    raise SystemExit(message)


def speak(base_url: str, text: str, voice: VoiceId, minting: Minting) -> bytes:
    """POST German text to the speech service and return the whole WAV bytes.

    Each call mints a fresh pair of tokens, so a long run never presents an
    expired one.
    """
    pair = mint_pair(minting)
    outcome = speaking_call.speak(
        base_url,
        text,
        voice,
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


def _rms(pcm: bytes) -> float:
    if len(pcm) < BYTES_PER_SAMPLE:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)])
    total = sum(sample * sample for sample in samples)
    return (total / len(samples)) ** 0.5 / 32768.0


def _run(base_url: str, voice: VoiceId, minting: Minting) -> None:
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

    wav = speak(base_url, SENTENCE, voice, minting)
    rate, pcm = _pcm_from_wav(wav)
    print(
        "contract_speak",
        f"bytes={len(wav)} pcm={len(pcm)} rate={rate} rms={_rms(pcm):.4f}",
    )
    if _rms(pcm) < SILENCE_RMS:
        message = "speak returned silence"
        raise SystemExit(message)


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    """Read the command line: the service URL, the voice and the minting file's path."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--minting-file", type=Path, required=True)
    parser.add_argument("base_url")
    parser.add_argument("voice", type=VoiceId)
    return parser.parse_args(arguments)


def main() -> None:
    """Drive health and speak against the speech service URL."""
    _must_not_have_imported_speech()
    options = parse_arguments(sys.argv[1:])
    _run(options.base_url, options.voice, load_minting(options.minting_file))


if __name__ == "__main__":
    main()
