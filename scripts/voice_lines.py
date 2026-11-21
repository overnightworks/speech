"""The voice's absolute lines, measured through the service's real entry point.

One service starts with a voice and a hearing model both ready. After
`--warmups` discarded runs (reported, never judged), the command speaks and
hears one request at a time and prints one JSON document on stdout; every other
line goes to stderr. `--route` names the hearing route of the final: the file
route, the live-hearing socket (its final runs from `speech_stopped` to
`transcript.completed`) or both. Each run samples the host's load and CPU busy
share at its start and end. With `--two-halves` it measures an idle half,
writes `HALF_SIGNAL` on stderr, waits for one line on stdin (the caller's own
load has started) and measures a loaded half. Given lines are judged on the
last half by the slowest warm run, each final on the routes named.
Exit 0: every given line met. Exit 3: a line missed. Exit 1: a run failed.
"""

# ruff: noqa: T201 - the command's stderr carries its one machine-readable signal

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import select
import statistics
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, TextIO

import numpy as np

import speaking_figures as figures
from hearing_client import hear_file, hear_live
from hearing_figures import LIVE_INPUT_RATE, LiveTurn, final_after_speech_stopped
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
    from collections.abc import Awaitable, Callable, Mapping, Sequence
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
    "live_final_seconds",
)
HOST_FIGURES: Final = ("load_start", "load_end", "cpu_busy_share")
PROC: Final = Path("/proc")
FINAL_ROUTES: Final = {"final_seconds": "file", "live_final_seconds": "live"}

type Speak = Callable[[str, VoiceId], tuple[bytes, float, float]]
type Hear = Callable[[], float]


class Route(StrEnum):
    """The hearing routes whose final a run measures."""

    FILE = "file"
    LIVE = "live"
    BOTH = "both"


@dataclass(frozen=True)
class HostReading:
    """The host at one instant: its 1-minute load and its CPU ticks."""

    load: float
    busy_ticks: int
    total_ticks: int


@dataclass(frozen=True)
class HostSample:
    """The host over one run: its load at both ends and the CPU share it was busy."""

    load_start: float
    load_end: float
    cpu_busy_share: float


@dataclass(frozen=True)
class Run:
    """The figures of one run; a final is `None` on a route the run did not take."""

    first_audio_seconds: float
    first_audible_seconds: float
    real_time_factor: float
    total_seconds: float
    audio_seconds: float
    final_seconds: float | None
    live_final_seconds: float | None
    host: HostSample


@dataclass(frozen=True)
class Figure:
    """One figure over a half's runs."""

    median: float
    max: float
    runs: list[float]


@dataclass(frozen=True)
class Summary:
    """A half's figures and its host figures."""

    figures: dict[str, Figure]
    host: dict[str, Figure]


def host_reading(proc: Path = PROC) -> HostReading:
    """The load and CPU ticks from `proc`'s `loadavg` and `stat`."""
    load = float((proc / "loadavg").read_text().split()[0])
    cpu = [
        int((proc / "stat").read_text().splitlines()[0].split()[i]) for i in range(1, 9)
    ]
    idle = cpu[3] + cpu[4]
    return HostReading(load, sum(cpu) - idle, sum(cpu))


def host_sample(start: HostReading, end: HostReading) -> HostSample:
    """The host's load at both readings and the busy share of the ticks between."""
    ticks = end.total_ticks - start.total_ticks
    busy = (end.busy_ticks - start.busy_ticks) / ticks if ticks else 0.0
    return HostSample(start.load, end.load, busy)


def first_audible_seconds(spoken: figures.Spoken) -> float:
    """The first byte's time plus the audio position of the first audible sample."""
    _rate, pcm = pcm_from_wav(spoken.wav)
    audible = np.flatnonzero(np.abs(np.frombuffer(pcm, "<i2")) > AUDIBLE_SAMPLE)
    if not len(audible):
        message = "the speak held no audible sample"
        raise SystemExit(message)
    return spoken.first_audio_seconds + audible[0] / spoken.rate


def measure_run(
    speak: Speak,
    voice: VoiceId,
    host: Callable[[], HostReading],
    file: Hear | None,
    live: Hear | None,
) -> Run:
    """Speak, then hear on each route given, one request at a time."""
    started = host()
    spoken = figures.spoken_of(speak(SENTENCES[first_language(voice)], voice))
    file_final = file() if file else None
    live_final = live() if live else None
    return Run(
        spoken.first_audio_seconds,
        first_audible_seconds(spoken),
        spoken.real_time_factor,
        spoken.total_seconds,
        spoken.audio_seconds,
        file_final,
        live_final,
        host_sample(started, host()),
    )


def figure_of(values: Sequence[float]) -> Figure:
    """Median, maximum and every value."""
    return Figure(statistics.median(values), max(values), list(values))


def summarised(runs: Sequence[Run]) -> Summary:
    """Median, maximum and every value of each figure the runs took."""
    result: dict[str, Figure] = {}
    for name in FIGURES:
        values = [getattr(run, name) for run in runs]
        if None not in values:
            result[name] = figure_of(values)
    host = {
        name: figure_of([getattr(run.host, name) for run in runs])
        for name in HOST_FIGURES
    }
    return Summary(result, host)


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


def live_final(live_turn: Callable[[], LiveTurn]) -> Hear:
    """The live final of each call: `speech_stopped` to `transcript.completed`."""
    return lambda: final_after_speech_stopped(live_turn())


def measure(  # noqa: PLR0913 - each port is one boundary the tests replace
    arguments: argparse.Namespace,
    speak: Speak,
    hear: Hear,
    wait: Callable[[], None] = lambda: wait_for_caller(sys.stdin, CALLER_WAIT_SECONDS),
    *,
    live: Hear | None = None,
    host: Callable[[], HostReading] = host_reading,
) -> dict[str, object]:
    """The command's JSON document for a started service's `speak` and `hear`.

    `hear` is the file route's final, `live` the live route's.
    """
    voice = arguments.voice
    file = hear if arguments.route in {Route.FILE, Route.BOTH} else None
    live_route = live if arguments.route in {Route.LIVE, Route.BOTH} else None

    def take(count: int) -> list[Run]:
        return [measure_run(speak, voice, host, file, live_route) for _ in range(count)]

    warmups = take(arguments.warmups)
    halves: dict[str, Summary] = {}
    if arguments.two_halves:
        halves["idle"] = summarised(take(arguments.runs))
        print(HALF_SIGNAL, file=sys.stderr, flush=True)
        wait()
        halves["loaded"] = summarised(take(arguments.runs))
    else:
        halves["single"] = summarised(take(arguments.runs))
    last = list(halves.values())[-1].figures
    lines = judged(
        last,
        {
            "first_audio_seconds": arguments.first_audio_max,
            "real_time_factor": arguments.rtf_max,
            **dict.fromkeys(set(FINAL_ROUTES) & set(last), arguments.final_max),
        },
    )
    return {
        "voice": arguments.voice,
        "hearing_model": arguments.hearing_model,
        "device": arguments.device,
        "runs": arguments.runs,
        "warmups": [asdict(run) for run in warmups],
        "route": arguments.route,
        "final_routes": {
            name: FINAL_ROUTES[name] for name in FINAL_ROUTES if name in last
        },
        "clip_seconds": arguments.clip_seconds,
        "halves": {
            half: {
                **{name: asdict(figure) for name, figure in summary.figures.items()},
                "host": {name: asdict(figure) for name, figure in summary.host.items()},
            }
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
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--route", type=Route, choices=list(Route), default=Route.FILE)
    for name in ("first-audio-max", "rtf-max", "final-max"):
        parser.add_argument(f"--{name}", type=float, help="the line, in its unit")
    parser.add_argument("--two-halves", action="store_true")
    parsed = parser.parse_args(arguments)
    if parsed.runs < 1 or parsed.clip_seconds <= 0 or parsed.warmups < 0:
        parser.error(
            "--runs and --clip-seconds must be positive, --warmups not negative"
        )
    return parsed


def prove(  # noqa: PLR0913 - each port is one boundary the tests replace
    arguments: argparse.Namespace,
    *,
    service: Callable[..., AbstractContextManager[Service]] = hearing_service,
    speak: Callable[..., tuple[bytes, float, float]] = speak_measured,
    hear: Callable[[Service, Language, bytes], tuple[float, str]] = hear_file,
    clip: Callable[[Service, float], bytes] = clip_of,
    live: Callable[
        [Service, tuple[Language, ...], bytes], Awaitable[LiveTurn]
    ] = hear_live,
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
                live=live_final(
                    lambda: asyncio.run(live(started, (Language.DE,), spoken_clip))
                ),
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
