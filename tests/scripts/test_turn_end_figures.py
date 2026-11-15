"""The turn end measurement's figures hold for sentences worked out by hand."""

from __future__ import annotations

import pytest

from turn_end_figures import (
    EARLY_START_SILENCES_SECONDS,
    FIXED_SILENCE_SECONDS,
    Findings,
    Judgement,
    Sentence,
    SentenceResult,
    Stretches,
    report_language,
    results_of,
)

DETECTOR = "a detector"
DECISION_SECONDS = 0.05


def judgement(*, finished: bool, words_seconds: float = 0.0) -> Judgement:
    score = 0.9 if finished else 0.1
    return Judgement(score, DECISION_SECONDS, DECISION_SECONDS, words_seconds)


def three_sentences(silence: float, words_seconds: float) -> Findings:
    """Two sentences today's silence cuts, one of them at a pause called finished.

    The third is whole today, and its end is not called finished.
    """

    def heard(*, finished: bool) -> Judgement:
        return judgement(finished=finished, words_seconds=words_seconds)

    cut_at_a_pause = SentenceResult(
        cut_today=True,
        at_end=heard(finished=True),
        at_pauses=(heard(finished=False), heard(finished=True)),
    )
    kept_whole = SentenceResult(
        cut_today=True,
        at_end=heard(finished=True),
        at_pauses=(heard(finished=False),),
    )
    never_ended = SentenceResult(
        cut_today=False, at_end=heard(finished=False), at_pauses=()
    )
    results = (cut_at_a_pause, kept_whole, never_ended)
    return Findings(DETECTOR, {silence: results}, {})


def rows_of(findings: Findings) -> list[list[str]]:
    """The detector's rows in the order of the report's tables, as their cells."""
    lines = report_language("a language", [], [findings])
    return [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in lines
        if line.startswith(f"| {DETECTOR} |")
    ]


@pytest.mark.parametrize(
    ("silence", "words_seconds", "sooner", "median_seconds_sooner"),
    [
        pytest.param(0.2, 0.0, "2 of 2 (100%)", "+0.25", id="a detector of audio"),
        pytest.param(0.2, 0.4, "0 of 2 (0%)", "-0.15", id="words that take too long"),
        pytest.param(0.1, 0.3, "2 of 2 (100%)", "+0.05", id="an earlier silence"),
    ],
)
def test_a_verdict_is_sooner_only_by_what_its_decision_and_words_leave(
    silence: float, words_seconds: float, sooner: str, median_seconds_sooner: str
) -> None:
    at_end = rows_of(three_sentences(silence, words_seconds))[0]

    assert at_end == [
        DETECTOR,
        f"{silence:g} s",
        "3",
        "2 of 3 (67%)",
        sooner,
        median_seconds_sooner,
    ]


def test_pauses_and_the_fixed_silence_count_on_the_same_sentences() -> None:
    at_pauses = rows_of(three_sentences(0.2, 0.0))[1]

    assert at_pauses == [
        DETECTOR,
        "0.2 s",
        "3",
        "2.00",
        "1 of 3 (33%)",
        "1 of 3 (33%)",
        "2 of 3 (67%)",
        "1 of 2 (50%)",
    ]


def test_a_decision_costs_its_own_time_and_the_time_of_its_words() -> None:
    cost = rows_of(three_sentences(0.2, 0.4))[2]

    assert cost == [DETECTOR, "50.0 / 50.0", "50.0 / 50.0", "450.0 / 450.0"]


def test_a_sentence_is_asked_at_every_pause_each_silence_hears() -> None:
    longer, shorter = EARLY_START_SILENCES_SECONDS
    sentence = Sentence(
        "a sentence",
        {
            longer: Stretches(("to the pause",), "to the end"),
            shorter: Stretches(("to a breath", "to the pause"), "to the end"),
        },
        cut_by=frozenset({FIXED_SILENCE_SECONDS}),
    )

    results = results_of(
        [sentence], lambda heard: judgement(finished=heard != "to a breath")
    )

    at_pauses = rows_of(Findings(DETECTOR, results, {}))[2:4]
    every, none = "1 of 1 (100%)", "0 of 1 (0%)"
    assert at_pauses == [
        [DETECTOR, f"{longer:g} s", "1", "2.00", every, every, every, none],
        [DETECTOR, f"{shorter:g} s", "1", "3.00", "1 of 2 (50%)", every, every, none],
    ]
