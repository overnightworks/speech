"""The turn end measurement's figures: what was heard, what a detector found.

A detector is asked wherever the service has heard a decision silence: once a
read sentence is over, and at each pause the reader makes inside it. Its answers
read under two rules. Ending the turn: asked at `DECISION_SILENCE_SECONDS`, a
sentence end called finished ends the turn, sooner than the fixed silence by
what the verdict's own time leaves of the silence saved, and a pause called
finished cuts it. Starting early: the fixed silence still ends the turn, and a
verdict of finished at one of `EARLY_START_SILENCES_SECONDS` starts the agent,
sooner at the sentence end by the same count and for nothing at a pause.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

FIXED_SILENCE_SECONDS: Final = 0.5
FIXED_SILENCES_SECONDS: Final = (FIXED_SILENCE_SECONDS, 1.0, 2.0, 3.0)
DECISION_SILENCE_SECONDS: Final = 0.2
EARLY_START_SILENCES_SECONDS: Final = (DECISION_SILENCE_SECONDS, 0.1)
FINISHED_ABOVE: Final = 0.5

_FEWEST_FOR_QUANTILES: Final = 2


@dataclass(frozen=True, slots=True)
class Stretches[Heard]:
    """What is heard of one sentence wherever one length of silence ends a turn.

    `pauses` holds what is heard up to each pause inside the sentence, `end` what
    is heard once the sentence is over.
    """

    pauses: tuple[Heard, ...]
    end: Heard


@dataclass(frozen=True, slots=True)
class Sentence[Heard]:
    """One read sentence as the service hears it: its audio, or the words in it.

    `heard` holds its stretches by the decision silence they stop at, as 16 kHz
    PCM16 mono audio or as `Words`. `cut_by` names the fixed silences that end
    the turn inside the sentence.
    """

    transcript: str
    heard: Mapping[float, Stretches[Heard]]
    cut_by: frozenset[float]


@dataclass(frozen=True, slots=True)
class Words:
    """What the service's hearing makes of a stretch, and what hearing it cost."""

    text: str
    cpu_seconds: float
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class Judgement:
    """How finished the detector scored the turn, and what deciding cost.

    `words_seconds` is the wall time of the hearing pass that made the words a
    judge of words was given; a detector of audio waits for none.
    """

    score: float
    cpu_seconds: float
    wall_seconds: float
    words_seconds: float = 0.0

    @property
    def finished(self) -> bool:
        """Whether the score calls the turn finished."""
        return self.score > FINISHED_ABOVE

    @property
    def waited_seconds(self) -> float:
        """The wall time from the silence to the verdict, its words included."""
        return self.words_seconds + self.wall_seconds


@dataclass(frozen=True, slots=True)
class SentenceResult:
    """A detector's judgements on one sentence, beside today's fixed silence."""

    cut_today: bool
    at_end: Judgement
    at_pauses: tuple[Judgement, ...]

    @property
    def finished_inside(self) -> bool:
        """Whether the detector calls a pause inside the sentence finished."""
        return any(judgement.finished for judgement in self.at_pauses)


@dataclass(frozen=True, slots=True)
class Findings:
    """One detector on one language: its sentences, and its validity control.

    `results` holds its sentences by the decision silence it was asked at.
    `control` names each case whose truth is known and holds, per sentence,
    whether the detector called it finished.
    """

    label: str
    results: Mapping[float, tuple[SentenceResult, ...]]
    control: Mapping[str, tuple[bool, ...]]


def share(part: int, whole: int) -> str:
    """A count of a whole, with its percentage."""
    return f"{part} of {whole} ({part / whole:.0%})" if whole else "0 of 0"


def milliseconds(seconds: Sequence[float]) -> str:
    """The median and the 95th percentile of durations, in milliseconds."""
    if len(seconds) < _FEWEST_FOR_QUANTILES:
        return "none"
    median = statistics.median(seconds) * 1000
    high = statistics.quantiles(seconds, n=20)[-1] * 1000
    return f"{median:.1f} / {high:.1f}"


def reading_lines() -> list[str]:
    """The Markdown lines saying how a detector's answers read under the two rules."""
    early = " or ".join(f"{silence:g} s" for silence in EARLY_START_SILENCES_SECONDS)
    return [
        (
            f"- Ending the turn: a detector asked at {DECISION_SILENCE_SECONDS:g} s of"
            " silence ends the turn where it says finished. At the sentence end that"
            f" is sooner than the fixed {FIXED_SILENCE_SECONDS:g} s silence only by"
            " what its decision, and the hearing pass a judge of words waits for,"
            " leave of the silence saved; at a pause inside the sentence it cuts the"
            " turn."
        ),
        (
            f"- Starting early: the fixed {FIXED_SILENCE_SECONDS:g} s silence still"
            f" ends the turn, and a detector asked at {early} of silence starts the"
            " agent where it says finished. At the sentence end the agent starts"
            " sooner by the same count; at a pause it starts for nothing, and no"
            " turn is cut."
        ),
    ]


def results_of[Heard](
    sentences: Sequence[Sentence[Heard]], judged: Callable[[Heard], Judgement]
) -> dict[float, tuple[SentenceResult, ...]]:
    """Every sentence's judgements, by the decision silence they are asked at."""
    return {
        silence: tuple(
            SentenceResult(
                FIXED_SILENCE_SECONDS in sentence.cut_by,
                judged(sentence.heard[silence].end),
                tuple(judged(pause) for pause in sentence.heard[silence].pauses),
            )
            for sentence in sentences
        )
        for silence in EARLY_START_SILENCES_SECONDS
    }


def heard_lines[Heard](
    recordings: int, sentences: Sequence[Sentence[Heard]]
) -> list[str]:
    """The Markdown lines counting the sentences, their pauses and today's cuts."""
    cut = ", ".join(
        f"{silence:g} s: {sum(silence in s.cut_by for s in sentences)}"
        for silence in FIXED_SILENCES_SECONDS
    )
    pauses = (
        (
            f"- Pauses of {silence:g} s or more inside a sentence:"
            f" {sum(len(s.heard[silence].pauses) for s in sentences)},"
            f" in {sum(bool(s.heard[silence].pauses) for s in sentences)} sentences."
        )
        for silence in EARLY_START_SILENCES_SECONDS
    )
    return [
        (
            f"- Sentences: {len(sentences)} of {recordings} recordings; the others"
            f" stop before {DECISION_SILENCE_SECONDS:g} s of closing silence."
        ),
        *pauses,
        f"- Sentences a fixed silence cuts, of all {len(sentences)}: {cut}.",
    ]


def _closing_mark(words: str) -> str:
    """How the words end: in an ellipsis, in another mark or in none."""
    ending = words.rstrip()
    if ending.endswith(("...", "…")):
        return "an ellipsis"
    if ending and not ending[-1].isalnum():
        return f"`{ending[-1]}`"
    return "no mark"


def _closing_marks(heard: Iterable[Words]) -> str:
    marks = Counter(_closing_mark(words.text) for words in heard)
    counted = ", ".join(f"{mark} ({count})" for mark, count in marks.most_common())
    return counted or "none heard"


def words_lines(
    sentences: Sequence[Sentence[Words]], hearing_threads: int
) -> list[str]:
    """The Markdown lines on the words the judges of words are given.

    They say what a pass of the hearing on `hearing_threads` threads cost, and
    how the words heard close: the hearing's own guess at whether they end.
    """
    stretches = [
        (silence, heard)
        for sentence in sentences
        for silence, heard in sentence.heard.items()
    ]
    passes = [words for _, heard in stretches for words in (heard.end, *heard.pauses)]
    lines = [
        (
            f"- The judges of words are given the first {len(sentences)} sentences."
            f" Each decision follows a pass of the hearing model on {hearing_threads}"
            f" threads: {milliseconds([words.cpu_seconds for words in passes])} CPU ms"
            f" and {milliseconds([words.wall_seconds for words in passes])} wall ms,"
            " median / p95."
        )
    ]
    for asked in EARLY_START_SILENCES_SECONDS:
        heard = [heard for silence, heard in stretches if silence == asked]
        at_pauses = (words for stretch in heard for words in stretch.pauses)
        lines.append(
            f"- How the words heard at {asked:g} s of silence close: at the sentence"
            f" end {_closing_marks(stretch.end for stretch in heard)}; at a pause"
            f" {_closing_marks(at_pauses)}."
        )
    return lines


def _quartiles(scores: Sequence[float]) -> str:
    if len(scores) < _FEWEST_FOR_QUANTILES:
        return "none"
    return " / ".join(f"{quartile:.2f}" for quartile in statistics.quantiles(scores))


def _pauses(results: Sequence[SentenceResult]) -> list[Judgement]:
    return [judgement for result in results for judgement in result.at_pauses]


def _end_row(findings: Findings, silence: float) -> str:
    """How a detector's verdicts at the sentence end compare with the fixed silence.

    A verdict of finished comes sooner only by what is left of the silence it
    saves once the wall time of the decision, its words included, has passed.
    """
    results = findings.results[silence]
    sooner = [
        FIXED_SILENCE_SECONDS - silence - result.at_end.waited_seconds
        for result in results
        if result.at_end.finished
    ]
    median = f"{statistics.median(sooner):+.2f}" if sooner else "none"
    return (
        f"| {findings.label} | {silence:g} s | {len(results)}"
        f" | {share(len(sooner), len(results))}"
        f" | {share(sum(seconds > 0 for seconds in sooner), len(sooner))}"
        f" | {median} |"
    )


def _pause_row(findings: Findings, silence: float) -> str:
    """How a detector's verdicts at the pauses read, on its own sentences."""
    results = findings.results[silence]
    pauses = _pauses(results)
    cut_today = [result for result in results if result.cut_today]
    whole = sum(not result.finished_inside for result in cut_today)
    asked = f"{(len(results) + len(pauses)) / len(results):.2f}" if results else "none"
    return (
        f"| {findings.label} | {silence:g} s | {len(results)} | {asked}"
        f" | {share(sum(judgement.finished for judgement in pauses), len(pauses))}"
        f" | {share(sum(r.finished_inside for r in results), len(results))}"
        f" | {share(len(cut_today), len(results))}"
        f" | {share(whole, len(cut_today))} |"
    )


def _cost_row(findings: Findings) -> str:
    judgements = [
        judgement
        for results in findings.results.values()
        for result in results
        for judgement in (result.at_end, *result.at_pauses)
    ]
    return (
        f"| {findings.label}"
        f" | {milliseconds([judgement.cpu_seconds for judgement in judgements])}"
        f" | {milliseconds([judgement.wall_seconds for judgement in judgements])}"
        f" | {milliseconds([judgement.waited_seconds for judgement in judgements])} |"
    )


def _score_row(findings: Findings, silence: float) -> str:
    results = findings.results[silence]
    at_end = [result.at_end.score for result in results]
    at_pauses = [judgement.score for judgement in _pauses(results)]
    return (
        f"| {findings.label} | {silence:g} s | {_quartiles(at_end)}"
        f" | {_quartiles(at_pauses)} |"
    )


def _control_rows(findings: Findings) -> list[str]:
    return [
        f"| {findings.label} | {case} | {share(sum(finished), len(finished))} |"
        for case, finished in findings.control.items()
    ]


def _by_silence(measured: Sequence[Findings]) -> list[tuple[Findings, float]]:
    return [
        (findings, silence) for findings in measured for silence in findings.results
    ]


def report_language(
    title: str, heard: Sequence[str], measured: Sequence[Findings]
) -> list[str]:
    """The Markdown lines of one language's figures, under what was `heard` of it."""
    asked = _by_silence(measured)
    return [
        f"### {title}",
        "",
        *heard,
        "",
        (
            "| Detector | Asked at | Sentences | Sentence ends called finished"
            f" | Of them sooner than the fixed {FIXED_SILENCE_SECONDS:g} s silence,"
            " the decision's time counted"
            " | Seconds sooner, median (negative: later) |"
        ),
        "|---|---|---|---|---|---|",
        *(_end_row(findings, silence) for findings, silence in asked),
        "",
        (
            "| Detector | Asked at | Sentences | Decisions per sentence"
            " | Pauses called finished | Sentences with a pause called finished"
            f" | Sentences the fixed {FIXED_SILENCE_SECONDS:g} s silence cuts"
            " | Of those, without a pause called finished |"
        ),
        "|---|---|---|---|---|---|---|---|",
        *(_pause_row(findings, silence) for findings, silence in asked),
        "",
        (
            "| Detector | CPU ms per decision, median / p95 | Wall ms, median / p95"
            " | Wall ms with the words it needs, median / p95 |"
        ),
        "|---|---|---|---|",
        *(_cost_row(findings) for findings in measured),
        "",
        (
            "| Detector | Asked at | Score at the sentence end, quartiles"
            " | Score at a pause, quartiles |"
        ),
        "|---|---|---|---|",
        *(_score_row(findings, silence) for findings, silence in asked),
        "",
        "| Detector | Validity control | Called finished |",
        "|---|---|---|",
        *(row for findings in measured for row in _control_rows(findings)),
        "",
    ]
