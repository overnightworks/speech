"""The pure figures of the hearing stage proof: timings, word errors, clips, record.

Nothing here starts a process or reads a clock; `prove_hearing` drives the
service and hands the observations to these functions.
"""

from __future__ import annotations

import io
import math
import re
import statistics
import wave
from dataclasses import asdict, dataclass
from enum import StrEnum
from http import HTTPStatus
from typing import TYPE_CHECKING, Final

import numpy as np

from speech_runtime_client.wire import RefusalCode
from speech_runtime_service.hearing_session import (
    MAX_TURN_SECONDS,
    SILENCE_SECONDS,
    SPEECH_RMS,
)
from speech_runtime_service.pcm import BYTES_PER_SAMPLE, LIVE_INPUT_RATE

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

MIN_RUNS: Final = 5
MIN_GROWING_DELTAS: Final = 2
FRAME_SECONDS: Final = 0.1
FRAME_BYTES: Final = int(LIVE_INPUT_RATE * FRAME_SECONDS) * BYTES_PER_SAMPLE
TRAILING_SILENCE_SECONDS: Final = 2.0
INT16_SCALE: Final = 32768.0


@dataclass(frozen=True)
class LiveTurn:
    """What a client saw of one turn; every time counts from its first audio."""

    question_seconds: float
    delta_seconds: tuple[float, ...]
    deltas: tuple[str, ...]
    completed_seconds: float
    transcript: str


@dataclass(frozen=True)
class Timing:
    """The first partial and the final of one turn, from its start and its end.

    A time after the question's end is negative when the partial came during
    the question. The first partial is `None` for a turn that had no delta.
    """

    first_partial_from_start: float | None
    first_partial_after_end: float | None
    final_from_start: float
    final_after_end: float


def timing_of(turn: LiveTurn) -> Timing:
    """The turn's first partial and final against its start and its end."""
    first = turn.delta_seconds[0] if turn.delta_seconds else None
    return Timing(
        first_partial_from_start=first,
        first_partial_after_end=None
        if first is None
        else first - turn.question_seconds,
        final_from_start=turn.completed_seconds,
        final_after_end=turn.completed_seconds - turn.question_seconds,
    )


def count_growing_deltas(deltas: Sequence[str]) -> int:
    """The deltas that add words; each one grows the provisional text."""
    return sum(1 for delta in deltas if delta.strip())


def partial_runs(timings: Sequence[Timing]) -> int:
    """How many of the runs had a first partial."""
    return sum(1 for timing in timings if timing.first_partial_from_start is not None)


def _median_of_present(values: Sequence[float | None]) -> float | None:
    """The median over the runs that have it; `None` below `MIN_RUNS` of them."""
    present = [value for value in values if value is not None]
    return statistics.median(present) if len(present) >= MIN_RUNS else None


def median_timing(timings: Sequence[Timing]) -> Timing:
    """The median of each figure over `MIN_RUNS` runs or more.

    Raises `ValueError` for fewer runs: a median of fewer is no measurement. A
    first partial that fewer than `MIN_RUNS` runs had is `None` for that reason.
    """
    if len(timings) < MIN_RUNS:
        message = f"a median needs at least {MIN_RUNS} runs, got {len(timings)}"
        raise ValueError(message)
    return Timing(
        first_partial_from_start=_median_of_present(
            [timing.first_partial_from_start for timing in timings]
        ),
        first_partial_after_end=_median_of_present(
            [timing.first_partial_after_end for timing in timings]
        ),
        final_from_start=statistics.median(t.final_from_start for t in timings),
        final_after_end=statistics.median(t.final_after_end for t in timings),
    )


def words(text: str) -> list[str]:
    """The comparable words of a text: case and punctuation do not count."""
    return re.findall(r"\w+", text.casefold())


@dataclass(frozen=True)
class WordErrors:
    """Edits (substitutions, insertions, deletions) against a reference text.

    `reference_words_total` counts the reference's words once per comparison, so
    the errors of several runs added together have the words of all of them.
    """

    errors: int
    reference_words_total: int

    @property
    def rate(self) -> float:
        """Errors per reference word."""
        return self.errors / self.reference_words_total

    def plus(self, other: WordErrors) -> WordErrors:
        """The errors of both comparisons taken together."""
        return WordErrors(
            self.errors + other.errors,
            self.reference_words_total + other.reference_words_total,
        )


def word_errors(reference: str, heard: str) -> WordErrors:
    """The word edit distance of `heard` from `reference`."""
    known = words(reference)
    got = words(heard)
    previous = list(range(len(got) + 1))
    for row, known_word in enumerate(known, start=1):
        current = [row]
        for column, got_word in enumerate(got, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (known_word != got_word),
                )
            )
        previous = current
    return WordErrors(previous[-1], len(known))


def seconds_of(pcm: bytes) -> float:
    """The length of 24 kHz PCM16 mono in seconds."""
    return len(pcm) / (LIVE_INPUT_RATE * BYTES_PER_SAMPLE)


class ClipError(ValueError):
    """A clip that cannot serve as one continuous turn."""


def _frame_rms(pcm: bytes) -> list[float]:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / INT16_SCALE
    count = len(samples) // (FRAME_BYTES // BYTES_PER_SAMPLE)
    frames = samples[: count * (FRAME_BYTES // BYTES_PER_SAMPLE)].reshape(count, -1)
    return np.sqrt((frames**2).mean(axis=1)).tolist()


def check_continuous_turn(pcm: bytes) -> float:
    """Fail loud unless `pcm` (24 kHz) is one turn the service hears whole.

    No stretch of `SILENCE_SECONDS` inside the speech may fall under the
    service's speech threshold, since that would end the turn early, and the
    clip with the silence that ends it must stay below the turn cap. Returns
    the clip's seconds.
    """
    seconds = seconds_of(pcm)
    if seconds + SILENCE_SECONDS >= MAX_TURN_SECONDS:
        message = (
            f"the clip is {seconds:.1f} s; with the closing silence it must stay"
            f" below the {MAX_TURN_SECONDS:.0f} s turn cap"
        )
        raise ClipError(message)
    all_levels = _frame_rms(pcm)
    loud = [index for index, level in enumerate(all_levels) if level >= SPEECH_RMS]
    if not loud:
        message = "the clip holds no speech"
        raise ClipError(message)
    levels = all_levels[loud[0] : loud[-1] + 1]
    window = math.ceil(SILENCE_SECONDS / FRAME_SECONDS)
    for start in range(len(levels) - window + 1):
        energy = sum(level**2 for level in levels[start : start + window]) / window
        if math.sqrt(energy) < SPEECH_RMS:
            at = (loud[0] + start) * FRAME_SECONDS
            message = (
                f"a pause of {SILENCE_SECONDS} s at {at:.1f} s would end the turn early"
            )
            raise ClipError(message)
    return seconds


def to_wire_rate(pcm: bytes, rate: int) -> bytes:
    """PCM16 mono at `rate` as PCM16 mono at the wire's 24 kHz, by interpolation."""
    samples = np.frombuffer(pcm, dtype="<i2")
    count = round(len(samples) * LIVE_INPUT_RATE / rate)
    positions = np.arange(count) * rate / LIVE_INPUT_RATE
    resampled = np.interp(positions, np.arange(len(samples)), samples)
    return resampled.round().astype("<i2").tobytes()


def wav_of(pcm: bytes) -> bytes:
    """A WAV file around 24 kHz PCM16 mono."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(BYTES_PER_SAMPLE)
        writer.setframerate(LIVE_INPUT_RATE)
        writer.writeframes(pcm)
    return buffer.getvalue()


def frames_with_closing_silence(pcm: bytes) -> list[bytes]:
    """The clip in wire frames, then the silence that lets the service end the turn."""
    closed = pcm + bytes(
        int(LIVE_INPUT_RATE * TRAILING_SILENCE_SECONDS) * BYTES_PER_SAMPLE
    )
    return [
        closed[start : start + FRAME_BYTES]
        for start in range(0, len(closed), FRAME_BYTES)
    ]


@dataclass(frozen=True)
class MemoryReading:
    """The card's used and total memory in mebibytes."""

    used_mib: int
    total_mib: int


def parse_memory(text: str) -> MemoryReading:
    """Read `nvidia-smi --query-gpu=memory.used,memory.total` without units."""
    used, total = (int(part) for part in text.splitlines()[0].split(","))
    return MemoryReading(used, total)


def peak_over_baseline(baseline_mib: int, samples_mib: Sequence[int]) -> int:
    """How far the highest sample rose above the baseline, never below zero."""
    if not samples_mib:
        message = "no memory sample was taken"
        raise ValueError(message)
    return max(max(samples_mib) - baseline_mib, 0)


@dataclass(frozen=True)
class TakenCardObservation:
    """What a service started on a taken card answered."""

    speaking_ready: bool
    hearing_ready: bool
    file_status: int
    file_code: str | None
    live_status: int | None
    live_code: str | None
    spoke_bytes: int


def taken_card_failures(seen: TakenCardObservation) -> list[str]:
    """What differs from cell 30: speaking up, hearing refused as unavailable."""
    expectations = {
        "speaking is not ready": seen.speaking_ready,
        "hearing is ready on a taken card": not seen.hearing_ready,
        "the file route did not answer 503 unavailable": (
            seen.file_status == HTTPStatus.SERVICE_UNAVAILABLE
            and seen.file_code == RefusalCode.UNAVAILABLE
        ),
        "the live upgrade did not answer 503 unavailable": (
            seen.live_status == HTTPStatus.SERVICE_UNAVAILABLE
            and seen.live_code == RefusalCode.UNAVAILABLE
        ),
        "speaking returned no audio": seen.spoke_bytes > 0,
    }
    return [problem for problem, held in expectations.items() if not held]


class StageStatus(StrEnum):
    """Where a stage or model of the record stands; only the last two are finished."""

    UNFINISHED = "unfinished"
    MEASURED = "measured"
    NOT_APPLICABLE = "not applicable"


def open_stage(device: str, applies_on: str) -> dict[str, object]:
    """A stage's record before it runs: unfinished on `applies_on`, else why not."""
    if device != applies_on:
        return {
            "status": StageStatus.NOT_APPLICABLE,
            "reason": f"the device is {device}",
        }
    return {"status": StageStatus.UNFINISHED}


def _is_finished(entry: object) -> bool:
    return isinstance(entry, dict) and entry.get("status") in (
        StageStatus.MEASURED,
        StageStatus.NOT_APPLICABLE,
    )


def _heard_nothing(heard: object) -> bool:
    """True for a heard text, or a list of them, that holds a blank text."""
    texts = [heard] if isinstance(heard, str) else heard
    return isinstance(texts, list) and any(not text.strip() for text in texts)


def _unmeasured_figures(entry: object, path: tuple[str, ...] = ()) -> Iterator[str]:
    """The paths of the figures of the record that measure nothing.

    A null `median` (fewer than `MIN_RUNS` runs had it) or long-turn `timing`
    figure, a long turn with fewer than `MIN_GROWING_DELTAS` growing deltas, and
    a `heard` text that is empty, which would score a word error rate of 1.0 yet
    prove no hearing.
    """
    if not isinstance(entry, dict):
        return
    for key, value in entry.items():
        if key in ("median", "timing") and isinstance(value, dict):
            yield from (
                " ".join((*path, figure))
                for figure, number in value.items()
                if number is None
            )
        elif key == "growing_deltas":
            if value < MIN_GROWING_DELTAS:
                yield " ".join((*path, key))
        elif key == "heard":
            if _heard_nothing(value):
                yield " ".join((*path, key))
        else:
            yield from _unmeasured_figures(value, (*path, str(key)))


def not_measured(
    required_models: Iterable[str],
    models: Mapping[str, object],
    stages: Mapping[str, object],
) -> list[str]:
    """What the record lacks: unfinished models and stages, and unmeasured figures.

    The figures are those `_unmeasured_figures` names.
    """
    return [
        *(
            f"hearing model {model}"
            for model in required_models
            if not _is_finished(models.get(model))
        ),
        *(f"stage {name}" for name, stage in stages.items() if not _is_finished(stage)),
        *_unmeasured_figures(models),
    ]


def timing_record(timing: Timing) -> dict[str, float | None]:
    """A timing as record fields."""
    return asdict(timing)


def error_record(errors: WordErrors) -> dict[str, float | int]:
    """Word errors as record fields."""
    return {
        "errors": errors.errors,
        "reference_words_total": errors.reference_words_total,
        "rate": errors.rate,
    }
