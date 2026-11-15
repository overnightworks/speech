"""Measure, on CPU, which AudioSet sound tagger fits beside the service's voices.

Three candidates run on one thread as ONNX: YAMNet, the sherpa-onnx zipformer
small audio tagger (its 32-bit and 8-bit files) and PretrainedSED's frame_mn10.
A candidate whose ONNX output differs from its reference is reported as excluded.
Per candidate the report holds:

- CPU per hop at p95, alone and while two of the service's Piper voices speak and
  the service's own hearing hears a recording, and how much the hearing time rises;
- the hit rate per sound on public clips (FSD50K, VocalSound);
- the false alarms per hour on public speech with silence after each sentence;
- the RSS the candidate adds.

The operator's "määp" and his own echo need recordings only he can make: pass
their directory with `--operator-recordings`. Without it those rows are reported
as not measured; with a directory that lacks them the run refuses.

Every model, checkpoint and clip is fetched by pinned revision into the cache
directory or the Hugging Face cache, never into the repository. Load one
candidate per process, as the added RSS is the process's growth: a run of several
candidates starts one child process per candidate. The report is Markdown on
standard output.

Run it as `uv run scripts/probe_sound_detector.py --candidates yamnet`.
"""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "ai-edge-litert==2.3.0",
#     "faster-whisper==1.2.1",
#     "fsspec==2026.9.0",
#     "huggingface-hub==2.4.0",
#     "numpy==2.5.4",
#     "onnx==1.23.2",
#     "onnxruntime==1.29.0",
#     "onnxscript==0.7.2",
#     "piper-tts==1.8.0",
#     "pydantic==2.14.3",
#     "sherpa-onnx==2.0.1",
#     "soundfile==0.13.1",
#     "soxr==1.1.0",
#     "torch==2.11.0",
#     "torchaudio==2.11.0",
#     "torchvision==0.26.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchaudio = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
# ///

# ruff: noqa: T201 - the measurement prints its report as the command line output

from __future__ import annotations

import argparse
import itertools
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from huggingface_hub import constants

# The service's hearing and Piper voices are read from this checkout: installing
# the service would pull its private dependencies into the measurement.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/client/src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "providers/provider-contract/src")
)

from sound_corpus import (
    OPERATOR_DIRECTORIES,
    SPEECH_REPOSITORY,
    Corpus,
    Sentence,
    Sound,
    load_sound_clips,
    operator_recordings,
    speech_sentences,
    speech_with_silence,
)
from sound_taggers import (
    CANDIDATES,
    SAMPLE_RATE,
    Check,
    Samples,
    Tagger,
)
from speech_client.wire import Language, VoiceId
from speech_service.hearing import ScoreIdentifier, WhisperHearing
from speech_service.hearing_models import HearingModel, fetch_model, voiced_floor
from speech_service.piper_voices import fetch_voice_files
from speech_service.speaking import PiperSpeaking

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

HOP_SAMPLES: Final = SAMPLE_RATE // 2
HOP_SECONDS: Final = HOP_SAMPLES / SAMPLE_RATE
REFRACTORY_HOPS: Final = 2
THRESHOLDS: Final = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# The proposed pass thresholds of the item.
MAX_CPU_P95_MILLISECONDS: Final = 50.0
MAX_HEARING_RISE: Final = 0.10
MIN_HIT_RATE: Final = 0.80
MAX_FALSE_ALARMS_PER_HOUR: Final = 6.0
MIN_ECHO_MINUTES: Final = 30.0
MIN_SPEECH_MINUTES: Final = 120.0
MAX_ADDED_RSS_MEGABYTES: Final = 200.0

RECORDING_SECONDS: Final = (8.0, 14.0)
HEARING_ROUNDS: Final = 3
PHASE_SECONDS: Final = 40.0
CHECKED_WINDOWS: Final = 8
WARM_UP_PASSES: Final = 3
CONTENTION_CLIPS: Final = 40
SPOKEN_SENTENCES: Final = 40
HEARING_MODEL: Final = HearingModel.SMALL
PIPER_VOICES: Final = ((VoiceId.THORSTEN, Language.DE), (VoiceId.NORMAN, Language.EN))
_MEGABYTE: Final = 1_000_000
_INT16_PEAK: Final = 32767


# Labels that are speech, silence or the room are not a sound for "määp".
QUIET: Final = re.compile(
    "speech|silence|inside,|outside,|conversation|narration|babbling|whisper|human",
    re.IGNORECASE,
)
ANY_SOUND: Final = len(Sound)


def _rss_megabytes() -> float:
    pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / _MEGABYTE


@dataclass(frozen=True, slots=True)
class HopCost:
    """What one hop cost the thread that ran it, in seconds."""

    cpu: float
    wall: float


def windows_of(samples: Samples, size: int) -> list[Samples]:
    """The windows a recording is scored in: one per hop, a short one padded."""
    padded = np.pad(samples, (0, max(0, size - len(samples))))
    last = len(padded) - size + 1
    return [padded[start : start + size] for start in range(0, last, HOP_SAMPLES)]


def timed(tagger: Tagger, window: Samples) -> tuple[npt.NDArray[np.float32], HopCost]:
    """The scores of `window` and what they cost this thread."""
    cpu, wall = time.thread_time(), time.perf_counter()
    scores = tagger.scores(window)
    return scores, HopCost(time.thread_time() - cpu, time.perf_counter() - wall)


@dataclass(frozen=True, slots=True)
class Heard:
    """What a tagger made of some segments.

    Per segment the best score of every sound and of any sound, for each hop; the
    seconds the segments span; the cost of each hop.
    """

    hops: list[npt.NDArray[np.float32]]
    seconds: float
    costs: list[HopCost]

    @classmethod
    def of(cls, tagger: Tagger, segments: Sequence[Samples]) -> Heard:
        """Hear every segment hop by hop."""
        columns = [[tagger.labels.index(n) for n in s.audioset_labels] for s in Sound]
        columns.append([i for i, n in enumerate(tagger.labels) if not QUIET.search(n)])
        hops, costs = [], []
        for segment in segments:
            timings = [
                timed(tagger, w) for w in windows_of(segment, tagger.window_samples)
            ]
            scores = np.stack([scores for scores, _ in timings])
            hops.append(np.stack([scores[:, c].max(axis=1) for c in columns], axis=1))
            costs += [cost for _, cost in timings]
        return cls(hops, sum(len(s) for s in segments) / SAMPLE_RATE, costs)

    @property
    def peaks(self) -> npt.NDArray[np.float32]:
        """Per segment (row) the best score of every column over its hops."""
        return np.stack([hop.max(axis=0) for hop in self.hops])

    def false_alarms_per_hour(self, threshold: float) -> float:
        """The events of the six sounds per hour of audio."""
        return self.per_hour(threshold, range(ANY_SOUND))

    def per_hour(self, threshold: float, columns: Sequence[int]) -> float:
        """The events of the columns per hour of audio.

        An event is a run of hops over the threshold; a gap of `REFRACTORY_HOPS`
        hops ends it.
        """
        events = 0
        for column in columns:
            for hop in self.hops:
                last = None
                for index in np.flatnonzero(hop[:, column] >= threshold):
                    events += last is None or index - last > REFRACTORY_HOPS
                    last = index
        return events * 3600 / self.seconds


def hit_rate(peaks: npt.NDArray[np.float32], column: int, threshold: float) -> float:
    """The share of recordings (rows) whose best hop reaches the threshold."""
    return float((peaks[:, column] >= threshold).mean())


@dataclass(frozen=True, slots=True)
class Contention:
    """The detector's hops and the hearing's times while two voices speak."""

    hops: tuple[HopCost, ...]
    hearing_alone: tuple[float, ...]
    hearing_with_detector: tuple[float, ...]

    @property
    def rise(self) -> float:
        """How much the median hearing time rises with the detector running."""
        with_detector = statistics.median(self.hearing_with_detector)
        return with_detector / statistics.median(self.hearing_alone) - 1


def _speak(voice: PiperSpeaking, texts: list[str], stop: threading.Event) -> None:
    """Synthesise `texts` in a loop at the pace a client plays them."""
    started, played = time.perf_counter(), 0.0
    for text in itertools.cycle(texts):
        for chunk in voice.pcm_chunks(text, ""):
            played += len(chunk) / 2 / voice.sample_rate
            if stop.wait(max(0.0, started + played - time.perf_counter())):
                return


def _detect_paced(
    tagger: Tagger,
    windows: Sequence[Samples],
    stop: threading.Event,
    costs: list[HopCost],
) -> None:
    due = time.perf_counter()
    for window in itertools.cycle(windows):
        if stop.is_set():
            return
        costs.append(timed(tagger, window)[1])
        due += HOP_SECONDS
        stop.wait(max(0.0, due - time.perf_counter()))


def _hear_times(
    hearing: WhisperHearing, recording: Sentence, runs: int, seconds: float
) -> list[float]:
    """The times of at least `runs` hearings that together last `seconds`."""
    pcm = (recording.samples * _INT16_PEAK).round().astype("<i2").tobytes()
    times, begun = [], time.perf_counter()
    while len(times) < runs or time.perf_counter() - begun < seconds:
        started = time.perf_counter()
        hearing.transcribe(pcm, (recording.language,))
        times.append(time.perf_counter() - started)
    return times


def measure_contention(
    tagger: Tagger, corpus: Corpus, cache: Path, runs: int
) -> Contention:
    """The hearing's time with and without the detector, two voices speaking.

    The two conditions alternate `HEARING_ROUNDS` times, each for `PHASE_SECONDS`
    and at least `runs` hearings, so that a drifting machine load falls on both.
    The detector hops at the service's pace over windows of the public clips; the
    voices speak at the pace of a playing client.
    """
    hearing = WhisperHearing(
        HEARING_MODEL,
        fetch_model(HEARING_MODEL, Path(constants.HF_HUB_CACHE)),
        "cpu",
        ScoreIdentifier(voiced_floor(HEARING_MODEL)),
    )
    hearing.load()
    stop, speakers = threading.Event(), []
    spoken = corpus.sentences[: 2 * SPOKEN_SENTENCES]
    for voice_id, language in PIPER_VOICES:
        fetch_voice_files(voice_id, cache / "voices")
        voice = PiperSpeaking(voice_id.value, cache / "voices")
        voice.load()
        texts = [s.transcript for s in spoken if s.language is language]
        speakers.append(
            threading.Thread(target=_speak, args=(voice, texts, stop), daemon=True)
        )
    low, high = RECORDING_SECONDS
    recording = next(
        s for s in corpus.sentences if low <= len(s.samples) / SAMPLE_RATE <= high
    )
    spread = corpus.clips[:: max(1, len(corpus.clips) // CONTENTION_CLIPS)]
    windows = [
        w for clip in spread for w in windows_of(clip.samples, tagger.window_samples)
    ]
    alone, with_detector, hops = [], [], []
    for speaker in speakers:
        speaker.start()
    _hear_times(hearing, recording, 1, 0.0)
    for _ in range(HEARING_ROUNDS):
        alone += _hear_times(hearing, recording, runs, PHASE_SECONDS)
        paused = threading.Event()
        detector = threading.Thread(
            target=_detect_paced, args=(tagger, windows, paused, hops), daemon=True
        )
        detector.start()
        with_detector += _hear_times(hearing, recording, runs, PHASE_SECONDS)
        paused.set()
        detector.join()
    stop.set()
    for speaker in speakers:
        speaker.join()
    return Contention(tuple(hops), tuple(alone), tuple(with_detector))


@dataclass(frozen=True, slots=True)
class Findings:
    """Everything measured of one candidate."""

    tagger: Tagger
    check: Check
    size_megabytes: float
    added_rss_megabytes: float = 0.0
    clips: Mapping[Sound, Heard] | None = None
    speech: Heard | None = None
    contention: Contention | None = None
    maep: Heard | None = None
    echo: Heard | None = None


def measure(tagger: Tagger, corpus: Corpus, cache: Path, runs: int) -> Findings:
    """Measure one candidate; one that fails its reference check stops there."""
    size = sum(f.stat().st_size for f in tagger.files if f.suffix != ".csv") / _MEGABYTE
    before = _rss_megabytes()
    tagger.load()
    warm = windows_of(corpus.clips[0].samples, tagger.window_samples)
    for window in warm * WARM_UP_PASSES:
        tagger.scores(window)
    # Read before the check: it loads the reference runtimes a service never holds.
    added = _rss_megabytes() - before
    checked = tagger.check()
    if not checked.passed:
        return Findings(tagger, checked, size)
    clips = {
        sound: Heard.of(tagger, [c.samples for c in corpus.clips if c.sound is sound])
        for sound in Sound
    }
    return Findings(
        tagger,
        checked,
        size,
        added,
        clips,
        Heard.of(tagger, corpus.segments),
        measure_contention(tagger, corpus, cache, runs),
        Heard.of(tagger, corpus.maep) if corpus.maep else None,
        Heard.of(tagger, corpus.echo) if corpus.echo else None,
    )


def _row(*cells: object) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _rule(text: str, figure: str, *, ok: bool | None) -> str:
    """One pass threshold, its figure and the verdict; `None` is not judged."""
    return _row(text, figure, "-" if ok is None else "pass" if ok else "FAIL")


def _share(rate: float) -> str:
    return f"{rate:.0%}"


def _milliseconds(costs: Sequence[HopCost]) -> str:
    """Median / p95 of the CPU and of the wall time per hop."""
    cpu, wall = [c.cpu * 1000 for c in costs], [c.wall * 1000 for c in costs]
    return (
        f"CPU {np.median(cpu):.1f} / {np.percentile(cpu, 95):.1f},"
        f" wall {np.median(wall):.1f} / {np.percentile(wall, 95):.1f}"
    )


def report_data(corpus: Corpus, cache: Path) -> list[str]:
    """The Markdown lines on what was heard: counts and licences per source."""
    counts: dict[tuple[str, str], int] = {}
    for clip in corpus.clips:
        key = (clip.source, clip.licence)
        counts[key] = counts.get(key, 0) + 1
    minutes = sum(len(s) for s in corpus.segments) / SAMPLE_RATE / 60
    per_sound = (f"{s.value} {sum(c.sound is s for c in corpus.clips)}" for s in Sound)
    return [
        "## Data",
        "",
        _row("Source", "Licence", "Clips"),
        "|---|---|---|",
        *(_row(source, licence, n) for (source, licence), n in sorted(counts.items())),
        _row(
            f"{SPEECH_REPOSITORY} dev", "CC-BY-4.0", f"{len(corpus.segments)} sentences"
        ),
        "",
        (
            f"Clips per sound: {', '.join(per_sound)}. Speech with 2 s of quiet noise"
            f" after each sentence: {minutes:.1f} min. Cache `{cache}`."
        ),
        "",
    ]


def _operator_rules(findings: Findings, threshold: float) -> list[str]:
    maep, echo = findings.maep, findings.echo
    echo_text = (
        f"false alarms <= {MAX_FALSE_ALARMS_PER_HOUR:g}/h over >="
        f" {MIN_ECHO_MINUTES:g} min of his echo"
    )
    maep_text = f"hit rate >= {MIN_HIT_RATE:.0%} for his määp"
    if maep is None or echo is None:
        needs = "not measured: needs the operator's recordings"
        return [_rule(maep_text, needs, ok=None), _rule(echo_text, needs, ok=None)]
    rate = hit_rate(maep.peaks, ANY_SOUND, threshold)
    minutes = echo.seconds / 60
    per_hour = echo.false_alarms_per_hour(threshold)
    long_enough = minutes >= MIN_ECHO_MINUTES
    return [
        _rule(
            maep_text, f"{_share(rate)} of {len(maep.hops)}", ok=rate >= MIN_HIT_RATE
        ),
        _rule(
            echo_text,
            f"{per_hour:.1f}/h over {minutes:.1f} min"
            + ("" if long_enough else ", too short to judge"),
            ok=per_hour <= MAX_FALSE_ALARMS_PER_HOUR if long_enough else None,
        ),
    ]


def _report_header(findings: Findings) -> list[str]:
    tagger, check = findings.tagger, findings.check
    return [
        f"## {tagger.title}",
        "",
        (
            f"- Licence {tagger.licence}; model {findings.size_megabytes:.1f} MB;"
            f" window {tagger.window_samples / SAMPLE_RATE:.3g} s,"
            f" hop {HOP_SECONDS:g} s."
        ),
        f"- ONNX against its reference: {check.detail}: "
        + ("held." if check.passed else "EXCLUDED."),
    ]


def _operating_threshold(speech: Heard) -> float | None:
    """The lowest threshold whose false alarms on the speech stay within the limit."""
    return next(
        (
            t
            for t in THRESHOLDS
            if speech.false_alarms_per_hour(t) <= MAX_FALSE_ALARMS_PER_HOUR
        ),
        None,
    )


def _measurements(
    findings: Findings, contention: Contention, speech: Heard
) -> list[str]:
    loaded = list(contention.hops)
    return [
        f"- Added RSS {findings.added_rss_megabytes:.0f} MB.",
        (
            f"- ms per hop on one thread, median / p95: {_milliseconds(speech.costs)}"
            f" alone; {_milliseconds(loaded)} while two Piper voices speak and a"
            f" recording is heard ({len(loaded)} hops)."
        ),
        (
            f"- Hearing time with the detector {contention.rise:+.1%}: median"
            f" {statistics.median(contention.hearing_alone):.2f} s alone,"
            f" {statistics.median(contention.hearing_with_detector):.2f} s with it."
        ),
        "",
    ]


def _threshold_table(clips: Mapping[Sound, Heard], speech: Heard) -> list[str]:
    lines = [
        _row(
            "Threshold", *Sound, "other sounds' clips", "false alarms/h", "non-speech/h"
        ),
        "|---|" + "---|" * (len(Sound) + 3),
    ]
    for step in THRESHOLDS:
        own = [hit_rate(clips[s].peaks, i, step) for i, s in enumerate(Sound)]
        others = [
            hit_rate(
                np.concatenate([clips[o].peaks for o in Sound if o is not s]), i, step
            )
            for i, s in enumerate(Sound)
        ]
        lines.append(
            _row(
                f"{step:g}",
                *map(_share, own),
                _share(float(np.mean(others))),
                f"{speech.false_alarms_per_hour(step):.1f}",
                f"{speech.per_hour(step, [ANY_SOUND]):.1f}",
            )
        )
    return [
        *lines,
        "",
        (
            "A hit is a hop at or over the threshold in a clip of the sound; 'other"
            " sounds' clips' is how often a sound's detector fires on the clips of the"
            " other sounds; a false alarm is an event over"
            f" {speech.seconds / 3600:.2f} h of speech and silence, and 'non-speech'"
            " counts events of any label that is not speech, silence or the room."
        ),
        "",
    ]


def _speech_rule(speech: Heard, threshold: float, *, operating: bool) -> str:
    minutes = speech.seconds / 60
    long_enough = minutes >= MIN_SPEECH_MINUTES
    return _rule(
        f"false alarms <= {MAX_FALSE_ALARMS_PER_HOUR:g}/h over >="
        f" {MIN_SPEECH_MINUTES:g} min of speech",
        f"{speech.false_alarms_per_hour(threshold):.1f}/h at threshold {threshold:g}"
        f" over {minutes:.1f} min" + ("" if long_enough else ", too short to judge"),
        ok=operating if long_enough else None,
    )


def _pass_rules(findings: Findings, threshold: float, *, operating: bool) -> list[str]:
    clips, speech, contention = findings.clips, findings.speech, findings.contention
    cpu_p95 = float(np.percentile([c.cpu * 1000 for c in contention.hops], 95))
    rss = findings.added_rss_megabytes
    lines = [
        _row("Pass threshold", "Figure", "Verdict"),
        "|---|---|---|",
        _rule(
            f"CPU per hop <= {MAX_CPU_P95_MILLISECONDS:g} ms at p95, under load",
            f"{cpu_p95:.1f} ms",
            ok=cpu_p95 <= MAX_CPU_P95_MILLISECONDS,
        ),
        _rule(
            f"hearing time rise <= {MAX_HEARING_RISE:.0%}",
            f"{contention.rise:+.1%}",
            ok=contention.rise <= MAX_HEARING_RISE,
        ),
        _rule(
            f"added RSS <= {MAX_ADDED_RSS_MEGABYTES:g} MB",
            f"{rss:.0f} MB",
            ok=rss <= MAX_ADDED_RSS_MEGABYTES,
        ),
        _speech_rule(speech, threshold, operating=operating),
    ]
    for i, sound in enumerate(Sound):
        rate = hit_rate(clips[sound].peaks, i, threshold)
        text = (
            f"hit rate >= {MIN_HIT_RATE:.0%}, {sound.value}, at threshold {threshold:g}"
        )
        lines.append(_rule(text, _share(rate), ok=rate >= MIN_HIT_RATE))
    return lines


def report(findings: Findings) -> list[str]:
    """The Markdown lines of one candidate against every pass threshold."""
    lines = _report_header(findings)
    if not findings.check.passed:
        return [*lines, ""]
    speech = findings.speech
    operating = _operating_threshold(speech)
    threshold = THRESHOLDS[-1] if operating is None else operating
    return [
        *lines,
        *_measurements(findings, findings.contention, speech),
        *_threshold_table(findings.clips, speech),
        *_pass_rules(findings, threshold, operating=operating is not None),
        *_operator_rules(findings, threshold),
        "",
    ]


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    keys = sorted(CANDIDATES)
    parser.add_argument("--candidates", nargs="+", choices=keys, default=keys)
    parser.add_argument("--clips-per-sound", type=int, default=30)
    parser.add_argument("--speech-minutes", type=float, default=120.0)
    parser.add_argument("--speech-start-minutes", type=float, default=0.0)
    parser.add_argument("--hearings", type=int, default=3)
    parser.add_argument("--operator-recordings", type=Path)
    parser.add_argument("--section", action="store_true", help=argparse.SUPPRESS)
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    parser.add_argument("--cache", type=Path, default=cache / "probe_sound_detector")
    return parser.parse_args(argv)


def _run_in_own_process(arguments: argparse.Namespace, key: str) -> None:
    """Measure one candidate in a fresh process, so its RSS growth is its own."""
    command = [
        sys.executable,
        __file__,
        "--section",
        "--candidates",
        key,
        "--clips-per-sound",
        str(arguments.clips_per_sound),
        "--speech-minutes",
        str(arguments.speech_minutes),
        "--speech-start-minutes",
        str(arguments.speech_start_minutes),
        "--hearings",
        str(arguments.hearings),
        "--cache",
        str(arguments.cache),
    ]
    if arguments.operator_recordings:
        command += ["--operator-recordings", str(arguments.operator_recordings)]
    subprocess.run(command, check=True)  # noqa: S603 - argv is this interpreter and this script with its own parsed options and no shell


def _corpus(arguments: argparse.Namespace, *, measured: bool) -> Corpus:
    """What a run listens to; the parent of several candidates only reports the data."""
    maep, echo = (
        operator_recordings(arguments.operator_recordings)
        if measured and arguments.operator_recordings
        else ([], [])
    )
    return Corpus(
        load_sound_clips(arguments.clips_per_sound, arguments.cache),
        speech_with_silence(arguments.speech_start_minutes, arguments.speech_minutes),
        list(itertools.islice(speech_sentences(), 2 * SPOKEN_SENTENCES))
        if measured
        else [],
        maep,
        echo,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Fetch the data, measure each candidate and print the report."""
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    cache = arguments.cache
    own_processes = len(arguments.candidates) > 1
    corpus = _corpus(arguments, measured=not own_processes)
    if not arguments.section:
        print("# Sound tagger probe\n")
        print("\n".join(report_data(corpus, cache)), flush=True)
    if own_processes:
        del corpus
        for key in arguments.candidates:
            _run_in_own_process(arguments, key)
    else:
        spread = corpus.clips[:: max(1, len(corpus.clips) // CHECKED_WINDOWS)]
        checked_on = [windows_of(c.samples, 2 * SAMPLE_RATE)[0] for c in spread]
        tagger = CANDIDATES[arguments.candidates[0]](cache, checked_on)
        print(
            "\n".join(report(measure(tagger, corpus, cache, arguments.hearings))),
            flush=True,
        )
    if not arguments.operator_recordings and not arguments.section:
        print(
            "The määp and own-echo rows need the operator's recordings: `systemd-run"
            " --user --scope -q -p MemoryMax=4G -- probe-run uv run"
            " scripts/probe_sound_detector.py --operator-recordings <directory with"
            f" {' and '.join(OPERATOR_DIRECTORIES)}> --candidates"
            f" {' '.join(arguments.candidates)}`"
        )


if __name__ == "__main__":
    main()
