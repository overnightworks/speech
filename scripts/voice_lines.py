"""The voice's absolute lines, measured through the service's real entry point.

One service starts with a voice and a hearing model both ready. After one
discarded warm-up, the command speaks and hears one request at a time and prints
one JSON document on stdout; every other line goes to stderr. With
`--two-halves` it measures an idle half, writes `HALF_SIGNAL` on stderr, waits
for one line on stdin (the caller's own load has started) and measures a loaded
half. Given lines are judged on the last half by the slowest warm run.
Exit 0: every given line met. Exit 3: a line missed. Exit 1: a run failed.
"""

# ruff: noqa: T201 - the command's stderr carries its one machine-readable signal

from __future__ import annotations

import argparse
import contextlib
import json
import os
import select
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TextIO

import numpy as np

import speaking_figures as figures
from hearing_client import hear_file
from hearing_figures import LIVE_INPUT_RATE
from prove_hearing import GERMAN_LONG_TURN, german_clip, hearing_service
from prove_speech import (
    SENTENCES,
    UNDER_PROBE_RUN_ENV,
    prove_under_probe_run,
    run_under_probe_run,
    speak_measured,
)
from shipped_catalogue import ON_DEMAND_VOICES, first_language
from speech_client.wire import Language, VoiceId
from speech_service.hearing_models import HearingModel
from speech_service.pcm import BYTES_PER_SAMPLE, pcm_from_wav

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractContextManager

    from hearing_client import Service

HALF_SIGNAL: Final = "VOICE_LINES_IDLE_HALF_MEASURED"
"""The line on stderr that tells the caller to start its load."""
CALLER_WAIT_SECONDS: Final = 900.0
"""How long the command waits for the caller's line before it ends its service."""
MISSED_EXIT: Final = 3
AUDIBLE_SAMPLE: Final = 1000
LOAD_AT_START_SETTING: Final = "SPEECH_LOAD_GPU_VOICES_AT_START"
FIGURES: Final = (
    "first_audio_seconds",
    "first_audible_seconds",
    "real_time_factor",
    "total_seconds",
    "audio_seconds",
    "final_seconds",
)

type Speak = Callable[[str, VoiceId], tuple[bytes, float, float]]
type Hear = Callable[[], float]


@dataclass(frozen=True)
class Run:
    """The four figures of one warm run."""

    first_audio_seconds: float
    first_audible_seconds: float
    real_time_factor: float
    total_seconds: float
    audio_seconds: float
    final_seconds: float


@dataclass(frozen=True)
class Figure:
    """One figure over a half's runs."""

    median: float
    max: float
    runs: list[float]


def first_audible_seconds(spoken: figures.Spoken) -> float:
    """The first byte's time plus the audio position of the first audible sample."""
    _rate, pcm = pcm_from_wav(spoken.wav)
    audible = np.flatnonzero(np.abs(np.frombuffer(pcm, "<i2")) > AUDIBLE_SAMPLE)
    if not len(audible):
        message = "the speak held no audible sample"
        raise SystemExit(message)
    return spoken.first_audio_seconds + audible[0] / spoken.rate


def measure_run(speak: Speak, hear: Hear, voice: VoiceId) -> Run:
    """Speak, then hear, one request at a time."""
    spoken = figures.spoken_of(speak(SENTENCES[first_language(voice)], voice))
    return Run(
        spoken.first_audio_seconds,
        first_audible_seconds(spoken),
        spoken.real_time_factor,
        spoken.total_seconds,
        spoken.audio_seconds,
        hear(),
    )


def summarised(runs: Sequence[Run]) -> dict[str, Figure]:
    """Median, maximum and every value of each figure."""
    result: dict[str, Figure] = {}
    for name in FIGURES:
        values = [getattr(run, name) for run in runs]
        result[name] = Figure(statistics.median(values), max(values), values)
    return result


def judged(
    half: Mapping[str, Figure], limits: Mapping[str, float | None]
) -> dict[str, dict[str, object]]:
    """Each given limit with whether the half's slowest run kept it."""
    return {
        name: {"limit": limit, "met": half[name].max <= limit}
        for name, limit in limits.items()
        if limit is not None
    }


def wait_for_caller(stream: TextIO, seconds: float) -> None:
    """Wait for one line on `stream`; a silent or gone caller ends the command."""
    ready, _, _ = select.select([stream], [], [], seconds)
    if not ready or not stream.readline():
        message = "the caller sent no line to start the loaded half"
        raise SystemExit(message)


def start_settings(voice: VoiceId) -> dict[str, str]:
    """What the service needs to have `voice` loaded when it starts."""
    return {LOAD_AT_START_SETTING: "1"} if voice in ON_DEMAND_VOICES else {}


def clip_of(service: Service, seconds: float) -> bytes:
    """German speech of exactly `seconds`, made through the service."""
    needed = round(seconds * LIVE_INPUT_RATE) * BYTES_PER_SAMPLE
    pcm = b""
    while len(pcm) < needed:
        pcm += german_clip(service, GERMAN_LONG_TURN)
    return pcm[:needed]


def measure_half(
    speak: Speak, hear: Hear, voice: VoiceId, runs: int
) -> dict[str, Figure]:
    """The summarised figures of `runs` warm runs."""
    return summarised([measure_run(speak, hear, voice) for _ in range(runs)])


def measure(
    arguments: argparse.Namespace,
    speak: Speak,
    hear: Hear,
    wait: Callable[[], None] = lambda: wait_for_caller(sys.stdin, CALLER_WAIT_SECONDS),
) -> dict[str, object]:
    """The command's JSON document for a started service's `speak` and `hear`."""
    voice = arguments.voice
    measure_run(speak, hear, voice)
    halves: dict[str, dict[str, Figure]] = {}
    if arguments.two_halves:
        halves["idle"] = measure_half(speak, hear, voice, arguments.runs)
        print(HALF_SIGNAL, file=sys.stderr, flush=True)
        wait()
        halves["loaded"] = measure_half(speak, hear, voice, arguments.runs)
    else:
        halves["single"] = measure_half(speak, hear, voice, arguments.runs)
    lines = judged(
        list(halves.values())[-1],
        {
            "first_audio_seconds": arguments.first_audio_max,
            "real_time_factor": arguments.rtf_max,
            "final_seconds": arguments.final_max,
        },
    )
    return {
        "voice": arguments.voice,
        "hearing_model": arguments.hearing_model,
        "device": arguments.device,
        "runs": arguments.runs,
        "clip_seconds": arguments.clip_seconds,
        "halves": {
            half: {name: asdict(figure) for name, figure in summary.items()}
            for half, summary in halves.items()
        },
        "lines": lines,
        "met": all(line["met"] for line in lines.values()),
    }


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    """Read the command line before anything starts."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--voice", type=VoiceId, choices=list(VoiceId), required=True)
    parser.add_argument(
        "--hearing-model", type=HearingModel, choices=list(HearingModel), required=True
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--clip-seconds", type=float, default=12.5)
    for name in ("first-audio-max", "rtf-max", "final-max"):
        parser.add_argument(f"--{name}", type=float, help="the line, in its unit")
    parser.add_argument("--two-halves", action="store_true")
    parsed = parser.parse_args(arguments)
    if parsed.runs < 1 or parsed.clip_seconds <= 0:
        parser.error("--runs and --clip-seconds must be positive")
    return parsed


def prove(
    arguments: argparse.Namespace,
    *,
    service: Callable[..., AbstractContextManager[Service]] = hearing_service,
    speak: Callable[..., tuple[bytes, float, float]] = speak_measured,
    hear: Callable[[Service, Language, bytes], tuple[float, str]] = hear_file,
    clip: Callable[[Service, float], bytes] = clip_of,
) -> None:
    """Start the service, measure, print the document and exit with its verdict."""
    output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        os.environ.update(start_settings(arguments.voice))
        with service(
            arguments.device, arguments.hearing_model, arguments.voice
        ) as started:
            spoken_clip = clip(started, arguments.clip_seconds)
            result = measure(
                arguments,
                lambda text, voice: speak(
                    started.base_url, text, voice, started.minting
                ),
                lambda: hear(started, Language.DE, spoken_clip)[0],
            )
    json.dump(result, output)
    output.write("\n")
    raise SystemExit(0 if result["met"] else MISSED_EXIT)


def main() -> None:
    """Run the command under `probe-run` unless the caller is already inside it."""
    arguments = parse_arguments(sys.argv[1:])
    if UNDER_PROBE_RUN_ENV not in os.environ:
        raise SystemExit(run_under_probe_run(sys.argv[1:], script=Path(__file__)))
    prove_under_probe_run(lambda: prove(arguments))


if __name__ == "__main__":
    main()
