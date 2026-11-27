"""The sustained-speech probe's figures hold for scenarios worked out by hand."""

from __future__ import annotations

import math

import pytest

import sustained_figures as figures
from speech_service.voice_activity import WINDOW_SAMPLES
from sustained_figures import (
    Backchannel,
    Cell,
    CellFigures,
    DelayCorpus,
    DelayRow,
    DelayTrial,
    Fate,
    HeardClip,
    HoldTrial,
    Onset,
    ReplayDetector,
    Segment,
    Sound,
    Tally,
    Timed,
)
from tests.support.detector_support import FOUR_TENTHS_MS

_PROBABILITY = {"v": 0.9, ".": 0.0}


def windows(pattern: str) -> tuple[float, ...]:
    """One window per character: `v` is voiced speech, `.` is not voiced."""
    return tuple(_PROBABILITY[character] for character in pattern)


def word(start: float, end: float) -> Timed:
    return Timed(start, end, spoken=True)


def vocal_sound(start: float, end: float) -> Timed:
    return Timed(start, end, spoken=False)


@pytest.mark.parametrize(
    ("chunks", "answers"),
    [
        pytest.param([3 * WINDOW_SAMPLES], 3, id="whole-windows"),
        pytest.param([WINDOW_SAMPLES + 5, WINDOW_SAMPLES - 5], 2, id="a-split-window"),
        pytest.param([WINDOW_SAMPLES - 1], 0, id="short-of-a-window"),
    ],
)
def test_the_replay_detector_answers_the_windows_a_stream_completes(
    chunks: list[int], answers: int
) -> None:
    detector = ReplayDetector(windows("v.v..v"))

    heard = [p for size in chunks for p in detector.probabilities(bytes(size * 2))]

    assert heard == list(windows("v.v..v"))[:answers]


@pytest.mark.parametrize("duration", figures.DURATION_WINDOWS)
def test_a_cell_is_asked_for_the_windows_of_its_duration(duration: int) -> None:
    cell = Cell(duration, 5)
    lead = "." * 20

    reached = figures.first_sustain_ms(windows(lead + "v" * duration), cell)
    short = figures.first_sustain_ms(windows(lead + "v" * (duration - 1)), cell)

    assert reached == (20 + duration) * figures.WINDOW_MS
    assert short is None


@pytest.mark.parametrize("run", figures.RESTART_RUNS)
def test_a_cell_restarts_the_count_after_its_run_of_unvoiced_windows(run: int) -> None:
    cell = Cell(13, run)
    voiced = "v" * 8

    within = figures.first_sustain_ms(windows(voiced + "." * (run - 1) + "v" * 6), cell)
    restarting = figures.first_sustain_ms(windows(voiced + "." * run + "v" * 6), cell)

    assert within is not None
    assert restarting is None


def test_the_ruled_cell_is_the_ruled_pair() -> None:
    assert figures.RULED_CELL.duration_ms == FOUR_TENTHS_MS
    assert figures.RULED_CELL.pause_seconds == pytest.approx(0.144)


@pytest.mark.parametrize(
    ("pattern", "options", "fate"),
    [
        pytest.param("." * 20, {}, Fate.NOT_HEARD, id="no-voiced-window"),
        pytest.param("." * 16 + "vv", {}, Fate.HEARD, id="lead-in-of-half-a-second"),
        pytest.param("." * 15 + "vv", {}, Fate.NO_LEAD_IN, id="lead-in-just-short"),
        pytest.param(
            "." * 20 + "vv" + "." * 20,
            {"inside": (0.64, 0.68)},
            Fate.NOT_ISOLATED,
            id="voice-past-the-end-of-the-allowed-span",
        ),
        pytest.param(
            "." * 20 + "vv" + "." * 20,
            {"inside": (0.6, 0.72)},
            Fate.HEARD,
            id="voice-exactly-inside-the-allowed-span",
        ),
        pytest.param(
            "." * 20 + "v" * 31, {"max_windows": 31}, Fate.HEARD, id="31-windows"
        ),
        pytest.param(
            "." * 20 + "v" * 32, {"max_windows": 31}, Fate.TOO_LONG, id="32-windows"
        ),
        pytest.param(
            "." * 5 + "v" * 40,
            {"inside": (0.0, 0.5), "max_windows": 31},
            Fate.NO_LEAD_IN,
            id="the-lead-in-is-judged-first",
        ),
    ],
)
def test_an_excerpt_is_heard_not_heard_or_dropped_for_its_first_reason(
    pattern: str, options: dict[str, object], fate: Fate
) -> None:
    assert figures.fate_of(windows(pattern), **options) is fate


def test_only_heard_and_not_heard_excerpts_are_trials() -> None:
    assert {fate for fate in Fate if fate.kept} == {Fate.HEARD, Fate.NOT_HEARD}


@pytest.mark.parametrize(
    ("timed", "acts", "found"),
    [
        pytest.param(
            [word(10.0, 10.5)], [range(1)], [(10.0, 10.5)], id="one-word-alone"
        ),
        pytest.param(
            [word(10.0, 10.25), word(10.25, 10.5), word(10.5, 11.0)],
            [range(3)],
            [(10.0, 11.0)],
            id="three-words-in-one-second",
        ),
        pytest.param(
            [word(10.0, 10.25), word(10.25, 10.5), word(10.5, 10.75), word(10.75, 11)],
            [range(4)],
            [],
            id="four-words-are-no-backchannel",
        ),
        pytest.param([word(10.0, 11.25)], [range(1)], [], id="a-span-over-a-second"),
        pytest.param([word(10.0, 10.0)], [range(1)], [], id="a-span-of-no-length"),
        pytest.param(
            [word(8.0, 9.0), word(10.0, 10.5), word(11.5, 12.0)],
            [range(1, 2)],
            [(10.0, 10.5)],
            id="others-exactly-a-second-away",
        ),
        pytest.param(
            [word(8.0, 9.25), word(10.0, 10.5)],
            [range(1, 2)],
            [],
            id="a-word-just-under-a-second-before",
        ),
        pytest.param(
            [word(10.0, 10.5), word(11.25, 12.0)],
            [range(1)],
            [],
            id="a-word-just-under-a-second-after",
        ),
        pytest.param(
            [word(10.0, 10.25), vocal_sound(10.25, 10.5)],
            [range(2)],
            [],
            id="a-vocal-sound-inside-the-act",
        ),
        pytest.param(
            [vocal_sound(8.5, 9.5), word(10.0, 10.5)],
            [range(1, 2)],
            [],
            id="a-vocal-sound-before",
        ),
        pytest.param(
            [word(5.0, 5.5), word(10.0, 10.5), word(20.0, 20.5), word(30.0, 30.5)],
            [range(1, 2), range(3, 4)],
            [(10.0, 10.5), (30.0, 30.5)],
            id="each-act-on-its-own",
        ),
    ],
)
def test_a_backchannel_is_a_short_act_that_lies_alone(
    timed: list[Timed], acts: list[range], found: list[tuple[float, float]]
) -> None:
    spans = figures.backchannel_spans(timed, acts, others=[])

    assert [(span.start, span.end) for span in spans] == found


@pytest.mark.parametrize(
    ("segment", "overlapped"),
    [
        pytest.param(Segment(9.0, 10.25), True, id="another-speaks-at-its-start"),
        pytest.param(Segment(10.25, 12.0), True, id="another-speaks-inside-it"),
        pytest.param(Segment(10.75, 12.0), False, id="another-starts-after-it"),
        pytest.param(Segment(8.0, 9.75), False, id="another-ends-before-it"),
    ],
)
def test_a_backchannel_is_overlapped_when_another_speaker_talks_during_its_span(
    segment: Segment, *, overlapped: bool
) -> None:
    spans = figures.backchannel_spans([word(10.0, 10.5)], [range(1)], [segment])

    assert spans == [Backchannel(10.0, 10.5, overlapped=overlapped)]


@pytest.mark.parametrize(
    ("timed", "onsets"),
    [
        pytest.param([word(5.0, 5.5), word(5.5, 6.5)], [5.0], id="two-words-over-1-5s"),
        pytest.param([word(0.25, 0.75), word(0.75, 1.75)], [0.25], id="the-first-word"),
        pytest.param([word(5.0, 5.5), word(5.5, 6.25)], [], id="a-run-under-1-5s"),
        pytest.param(
            [word(5.0, 5.5), word(6.0, 6.5)], [], id="a-gap-of-0-5s-ends-the-run"
        ),
        pytest.param([word(5.0, 5.5), word(5.9375, 6.5)], [5.0], id="a-gap-under-0-5s"),
        pytest.param(
            [word(1.0, 1.5), word(1.5, 3.0), word(4.0, 4.5), word(4.5, 6.0)],
            [1.0, 4.0],
            id="a-second-of-quiet-opens-the-next-onset",
        ),
        pytest.param(
            [word(1.0, 3.0), word(3.75, 5.5)],
            [1.0],
            id="a-pause-under-a-second-is-no-new-onset",
        ),
        pytest.param(
            [vocal_sound(4.5, 5.0), word(5.5, 5.75), word(5.75, 7.0)],
            [],
            id="a-vocal-sound-takes-the-quiet-before-the-word",
        ),
        pytest.param(
            [vocal_sound(5.0, 5.5), word(5.5, 7.0)],
            [],
            id="a-vocal-sound-is-no-onset",
        ),
        pytest.param(
            [word(5.0, 5.25), vocal_sound(5.25, 6.0), word(6.0, 6.5)],
            [5.0],
            id="a-vocal-sound-continues-the-run",
        ),
    ],
)
def test_an_onset_is_a_word_after_a_second_of_quiet_that_goes_on(
    timed: list[Timed], onsets: list[float]
) -> None:
    assert [onset.at for onset in figures.speech_onsets(timed, others=[])] == onsets


def test_an_onset_is_overlapped_when_another_speakers_segment_spans_it() -> None:
    timed = [word(5.0, 5.5), word(5.5, 6.5)]

    spans = figures.speech_onsets(timed, [Segment(4.0, 5.0)])
    clear = figures.speech_onsets(timed, [Segment(4.0, 4.75)])

    assert spans == [Onset(5.0, overlapped=True)]
    assert clear == [Onset(5.0, overlapped=False)]


def clip(fills: str) -> HeardClip:
    """A clip of one wire window per character, filled with that character's byte."""
    pcm = b"".join(
        character.encode() * figures.WIRE_WINDOW_BYTES for character in fills
    )
    return HeardClip(pcm, windows("".join("v" if c.isupper() else "." for c in fills)))


def test_a_pair_joins_the_speech_of_both_clips_around_a_gap() -> None:
    first = clip("aaBBc")
    second = clip("ddEEf")

    joined = figures.join_pair(first, second, gap_windows=3)

    window = figures.WIRE_WINDOW_BYTES
    expected = (
        first.pcm[: 4 * window]
        + bytes(3 * window)
        + second.pcm[2 * window : 4 * window]
    )
    assert joined == expected


def test_the_pair_gaps_take_every_length_in_turn_and_start_over() -> None:
    gaps = [figures.pair_gap_windows(index) for index in range(30)]

    assert gaps[:14] == list(range(2, 16))
    assert gaps[14:28] == gaps[:14]
    assert gaps[28] == gaps[0]


def test_clips_are_paired_two_by_two_and_an_odd_one_is_left_out() -> None:
    assert figures.two_by_two([1, 2, 3, 4, 5]) == [(1, 2), (3, 4)]


def hold_trial(
    pattern: str,
    sound: Sound = Sound.COUGH,
    *,
    paired: bool = False,
    overlapped: bool = False,
) -> HoldTrial:
    return HoldTrial(sound, paired, overlapped, figures.closed(windows(pattern)))


def test_a_trial_holds_when_the_replay_raises_the_event_in_its_cell() -> None:
    lead = "." * 20
    trials = [
        hold_trial(lead + "v" * 12),
        hold_trial(lead + "v" * 13),
        hold_trial(lead + "v" * 6 + "." * 5 + "v" * 6),
    ]

    measured = figures.measure(Cell(13, 5), trials, [])

    assert measured.holds == (False, True, False)
    assert figures.measure(Cell(9, 5), trials, []).holds == (True, True, False)


def delay_trial(
    pattern: str, onset_seconds: float, *, overlapped: bool = False
) -> DelayTrial:
    return DelayTrial(DelayCorpus.AMI, overlapped, windows(pattern), onset_seconds)


def test_a_delay_is_the_first_event_counted_from_the_onset() -> None:
    onset = 20 * figures.WINDOW_SECONDS
    trial = delay_trial("." * 20 + "v" * 20, onset)

    delay = figures.delay_seconds(trial, Cell(13, 5))

    assert delay == pytest.approx(13 * figures.WINDOW_SECONDS)


def test_a_delay_is_infinite_when_the_excerpt_raises_no_event() -> None:
    assert (
        figures.delay_seconds(delay_trial("." * 20 + "v" * 5, 0.64), Cell(13, 5))
        == math.inf
    )


def test_an_overlapped_ami_onset_counts_in_the_ami_row_and_its_own() -> None:
    plain = delay_trial("v", 0.0)
    overlapped = delay_trial("v", 0.0, overlapped=True)
    german = DelayTrial(
        DelayCorpus.FLEURS,
        overlapped=False,
        probabilities=windows("v"),
        onset_seconds=0.0,
    )

    assert plain.rows == (DelayRow.AMI,)
    assert overlapped.rows == (DelayRow.AMI, DelayRow.AMI_OVERLAPPED)
    assert german.rows == (DelayRow.FLEURS,)


def cell_figures(
    cell: Cell,
    holds: str,
    ami: tuple[float, ...] = (0.4,) * 5,
    fleurs: tuple[float, ...] = (0.4,) * 5,
) -> CellFigures:
    """A cell's figures: `x` in `holds` is a false hold, and its delay rows."""
    return CellFigures(
        cell,
        tuple(character == "x" for character in holds),
        {DelayRow.AMI: ami, DelayRow.AMI_OVERLAPPED: (), DelayRow.FLEURS: fleurs},
    )


@pytest.mark.parametrize(
    ("ami", "fleurs", "meets"),
    [
        pytest.param((0.5,) * 5, (0.5,) * 5, True, id="median-and-p95-at-the-caps"),
        pytest.param((0.51,) * 5, (0.4,) * 5, False, id="ami-median-over"),
        pytest.param(
            (0.4,) * 5, (0.3, 0.3, 0.3, 0.3, 0.8), False, id="fleurs-p95-over"
        ),
        pytest.param((0.4,) * 5, (0.3, 0.3, 0.3, 0.3, 0.7), True, id="p95-at-the-cap"),
        pytest.param(
            (math.inf, math.inf, 0.3, 0.3, 0.3), (0.4,) * 5, False, id="infinite-p95"
        ),
        pytest.param(
            (math.inf, math.inf, math.inf, 0.3, 0.3),
            (0.4,) * 5,
            False,
            id="infinite-median",
        ),
    ],
)
def test_a_cell_meets_the_delay_cap_on_both_rows_or_none(
    ami: tuple[float, ...], fleurs: tuple[float, ...], *, meets: bool
) -> None:
    assert cell_figures(Cell(12, 5), "x", ami, fleurs).meets_delay_cap() is meets


def test_the_95th_percentile_is_the_nearest_rank() -> None:
    assert figures.percentile_95(list(map(float, range(1, 21)))) == pytest.approx(19.0)
    assert figures.percentile_95([5.0]) == pytest.approx(5.0)


SLOW = (0.6,) * 5


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        pytest.param(
            [
                cell_figures(Cell(11, 5), "xx.."),
                cell_figures(Cell(12, 5), "x..."),
                cell_figures(Cell(13, 5), "....", SLOW),
            ],
            Cell(12, 5),
            id="the-lowest-share-among-the-cells-under-the-cap",
        ),
        pytest.param(
            [
                cell_figures(Cell(12, 6), "x..."),
                cell_figures(Cell(12, 4), "x..."),
                cell_figures(Cell(11, 9), "x..."),
                cell_figures(Cell(11, 7), "x..."),
            ],
            Cell(11, 7),
            id="a-tie-goes-to-the-shorter-duration-then-the-smaller-run",
        ),
    ],
)
def test_the_pick_is_the_lowest_pooled_share_among_cells_under_the_delay_cap(
    table: list[CellFigures], expected: Cell
) -> None:
    taken = figures.pick(table)

    assert taken.cell == expected
    assert taken.replaces_ruled


@pytest.mark.parametrize(
    ("ruled_holds", "best_holds", "kept"),
    [
        pytest.param("xx......", "x.......", True, id="one-extra-of-one-disagreeing"),
        pytest.param("xxxx....", "........", True, id="four-extra-of-four-is-not-over"),
        pytest.param("xxxxx...", "........", False, id="five-extra-of-five-is-over"),
        pytest.param("xxxxxxxx", "xxxxx...", True, id="shared-holds-do-not-count"),
        pytest.param("xxxxxxxxxx", "xx........", False, id="eight-extra-of-eight"),
    ],
)
def test_a_better_cell_replaces_the_ruled_pair_only_beyond_two_sqrt_of_the_disagreement(
    ruled_holds: str, best_holds: str, *, kept: bool
) -> None:
    padding = "." * (10 - len(ruled_holds))
    table = [
        cell_figures(figures.RULED_CELL, ruled_holds + padding),
        cell_figures(Cell(12, 5), best_holds + padding),
    ]

    taken = figures.pick(table)

    assert (taken.cell == figures.RULED_CELL) is kept
    assert taken.replaces_ruled is not kept


def test_a_ruled_pair_that_misses_the_delay_cap_is_replaced_without_the_test() -> None:
    table = [
        cell_figures(figures.RULED_CELL, "....", SLOW),
        cell_figures(Cell(12, 5), "x..."),
    ]

    assert figures.pick(table).cell == Cell(12, 5)


def test_no_cell_under_the_delay_cap_picks_nothing() -> None:
    taken = figures.pick([cell_figures(Cell(12, 5), "....", SLOW)])

    assert taken.cell is None
    assert not taken.replaces_ruled
    assert not taken.within_bound


@pytest.mark.parametrize(
    ("cell", "within"),
    [
        pytest.param(Cell(10, 4), True, id="lower-corner"),
        pytest.param(Cell(15, 6), True, id="upper-corner"),
        pytest.param(Cell(9, 5), False, id="duration-below"),
        pytest.param(Cell(16, 5), False, id="duration-above"),
        pytest.param(Cell(12, 3), False, id="run-below"),
        pytest.param(Cell(12, 7), False, id="run-above"),
    ],
)
def test_a_pick_outside_the_bound_is_flagged_for_the_operator(
    cell: Cell, *, within: bool
) -> None:
    taken = figures.pick([cell_figures(cell, "....")])

    assert taken.cell == cell
    assert taken.within_bound is within


def test_a_cell_record_shows_false_holds_by_group_and_label_pooled_and_delays() -> None:
    trials = [
        hold_trial("v", Sound.BACKCHANNEL),
        hold_trial("v", Sound.BACKCHANNEL, overlapped=True),
        hold_trial("v", Sound.BACKCHANNEL, paired=True),
        hold_trial("v", Sound.COUGH),
    ]
    delays = {
        DelayRow.AMI: (0.4, math.inf),
        DelayRow.AMI_OVERLAPPED: (),
        DelayRow.FLEURS: (0.3,),
    }
    measured = CellFigures(Cell(13, 5), (True, False, True, False), delays)

    record = figures.cell_record(measured, trials)

    assert record["false_holds"] == {
        "backchannel": {"holds": 1, "trials": 2, "share_percent": 50.0},
        "backchannel pair": {"holds": 1, "trials": 1, "share_percent": 100.0},
        "cough": {"holds": 0, "trials": 1, "share_percent": 0.0},
    }
    assert record["backchannel_by_label"] == {
        "alone": {"holds": 1, "trials": 1, "share_percent": 100.0},
        "overlapped": {"holds": 0, "trials": 1, "share_percent": 0.0},
    }
    assert record["pooled"] == {"holds": 2, "trials": 4, "share_percent": 50.0}
    assert record["delay"] == {
        "ami english": {"count": 2, "median": None, "p95": None, "infinite_share": 0.5},
        "fleurs german": {"count": 1, "median": 0.3, "p95": 0.3, "infinite_share": 0.0},
    }


def test_a_group_below_its_minimum_is_missed_and_drops_are_counted_by_reason() -> None:
    tally = Tally()
    for _ in range(60):
        tally.add(Sound.COUGH, Fate.HEARD)
    tally.add(Sound.COUGH, Fate.NOT_HEARD)
    tally.add(Sound.COUGH, Fate.TOO_LONG)
    tally.add(Sound.LAUGHTER, Fate.HEARD)
    tally.add(Sound.LAUGHTER, Fate.NO_LEAD_IN)

    assert tally.missed([Sound.COUGH, Sound.LAUGHTER, Sound.THROAT_CLEARING]) == [
        Sound.LAUGHTER,
        Sound.THROAT_CLEARING,
    ]
    assert tally.record()[Sound.COUGH] == {
        "kept": 61,
        "not_heard": 1,
        "dropped": {"too long": 1},
    }
