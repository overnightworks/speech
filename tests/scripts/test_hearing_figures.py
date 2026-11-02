"""The hearing stage proof's figures hold for scenarios worked out by hand."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

import hearing_figures as figures
from speech_runtime_client.wire import (
    DurationUsage,
    RefusalCode,
    TranscriptCompleted,
    TranscriptDelta,
)

RATE = 24000
LOUD = 8000
BARELY_AUDIBLE = 1


def tone(seconds: float, amplitude: int = LOUD, rate: int = RATE) -> bytes:
    count = round(seconds * rate)
    wave_form = amplitude * np.sin(2 * np.pi * 440 * np.arange(count) / rate)
    return wave_form.astype("<i2").tobytes()


def turn(question: float, deltas: tuple[float, ...], final: float) -> figures.LiveTurn:
    return figures.LiveTurn(
        question_seconds=question,
        delta_seconds=deltas,
        deltas=tuple("wort" for _ in deltas),
        completed_seconds=final,
        transcript="wort",
    )


def test_a_turn_is_timed_from_its_start_and_from_its_end() -> None:
    timing = figures.timing_of(turn(question=3.0, deltas=(2.0, 4.5), final=9.0))

    assert timing == figures.Timing(
        first_partial_from_start=2.0,
        first_partial_after_end=-1.0,
        final_from_start=9.0,
        final_after_end=6.0,
    )


def test_a_turn_without_a_delta_has_no_first_partial() -> None:
    timing = figures.timing_of(turn(question=3.0, deltas=(), final=5.0))

    assert timing == figures.Timing(
        first_partial_from_start=None,
        first_partial_after_end=None,
        final_from_start=5.0,
        final_after_end=2.0,
    )


def test_the_median_takes_the_middle_run_of_each_figure() -> None:
    finals = [9.0, 5.0, 7.0, 6.0, 8.0]
    timings = [
        figures.timing_of(turn(question=3.0, deltas=(first,), final=final))
        for first, final in zip([1.0, 2.0, 4.0, 3.0, 5.0], finals, strict=True)
    ]

    median = figures.median_timing(timings)

    assert median == figures.Timing(
        first_partial_from_start=3.0,
        first_partial_after_end=0.0,
        final_from_start=7.0,
        final_after_end=4.0,
    )


def test_the_median_of_partials_skips_runs_that_had_none() -> None:
    timings = [
        figures.timing_of(turn(question=3.0, deltas=delta, final=5.0))
        for delta in [(1.0,), (), (3.0,), (), (2.0,)]
    ]

    median = figures.median_timing(timings)

    assert median.first_partial_from_start == pytest.approx(2.0)


def test_fewer_than_five_runs_give_no_median() -> None:
    timings = [figures.timing_of(turn(3.0, (1.0,), 5.0))] * 4

    with pytest.raises(ValueError, match="at least 5 runs"):
        figures.median_timing(timings)


@pytest.mark.parametrize(
    ("deltas", "growing"),
    [
        ([], 0),
        (["Wie"], 1),
        (["Wie", " spät", " ist", " es"], 4),
        (["Wie", " ", ""], 1),
    ],
)
def test_only_deltas_that_add_words_are_counted(
    deltas: list[str], growing: int
) -> None:
    assert figures.count_growing_deltas(deltas) == growing


@pytest.mark.parametrize(
    ("reference", "heard", "errors", "reference_words"),
    [
        ("Wie spät ist es?", "wie spät ist es", 0, 4),
        ("Wie spät ist es", "Wie spät ist das", 1, 4),
        ("Wie spät ist es", "Wie ist es", 1, 4),
        ("Wie spät ist es", "Wie früh spät ist es", 1, 4),
        ("Wie spät ist es", "", 4, 4),
        ("Wie spät ist es", "Was kostet das Brot hier", 5, 4),
    ],
)
def test_word_errors_count_the_edits_against_the_known_sentence(
    reference: str, heard: str, errors: int, reference_words: int
) -> None:
    found = figures.word_errors(reference, heard)

    assert (found.errors, found.reference_words) == (errors, reference_words)


def test_word_errors_of_several_runs_add_up_to_one_rate() -> None:
    first = figures.word_errors("a b c d", "a b c x")
    second = figures.word_errors("a b c d", "a b c d")

    assert first.plus(second).rate == 1 / 8


def test_a_clip_of_continuous_speech_is_one_turn() -> None:
    assert figures.check_continuous_turn(tone(5.0)) == pytest.approx(5.0)


def test_a_short_pause_does_not_end_the_turn() -> None:
    clip = tone(2.0) + tone(0.5, amplitude=0) + tone(2.0)

    assert figures.check_continuous_turn(clip) == pytest.approx(4.5)


def test_a_pause_of_the_silence_length_ends_the_turn_early() -> None:
    clip = tone(2.0) + tone(0.8, amplitude=0) + tone(2.0)

    with pytest.raises(figures.ClipError, match="would end the turn early"):
        figures.check_continuous_turn(clip)


def test_silence_before_and_after_the_speech_is_not_a_pause() -> None:
    clip = tone(1.0, amplitude=0) + tone(3.0) + tone(1.0, amplitude=0)

    assert figures.check_continuous_turn(clip) == pytest.approx(5.0)


def test_a_clip_that_reaches_the_turn_cap_is_refused() -> None:
    clip = tone(29.5)

    with pytest.raises(figures.ClipError, match="turn cap"):
        figures.check_continuous_turn(clip)


def test_a_clip_without_speech_is_refused() -> None:
    clip = tone(3.0, amplitude=BARELY_AUDIBLE)

    with pytest.raises(figures.ClipError, match="no speech"):
        figures.check_continuous_turn(clip)


def test_resampling_gives_the_samples_of_the_wire_rate() -> None:
    resampled = figures.to_wire_rate(tone(1.0, rate=22050), 22050)

    assert len(resampled) == RATE * 2
    assert figures.check_continuous_turn(resampled) == 1.0


def test_a_wav_carries_the_clip_at_the_wire_rate() -> None:
    clip = tone(0.5)

    with wave.open(io.BytesIO(figures.wav_of(clip))) as reader:
        assert reader.getframerate() == RATE
        assert reader.getnchannels() == 1
        assert reader.readframes(reader.getnframes()) == clip


def test_the_frames_end_in_silence_for_two_seconds() -> None:
    clip = tone(1.0)

    frames = figures.frames_with_closing_silence(clip)

    assert b"".join(frames[:10]) == clip
    assert all(frame == bytes(figures.FRAME_BYTES) for frame in frames[10:])
    assert len(frames) * figures.FRAME_SECONDS == pytest.approx(3.0)


def test_a_completed_and_a_delta_event_are_read_and_others_are_not() -> None:
    delta = TranscriptDelta(item_id="item", delta="Wie")
    completed = TranscriptCompleted(
        item_id="item", transcript="Wie spät", usage=DurationUsage(seconds=1.0)
    )

    assert figures.parse_live_event(delta.model_dump_json()) == delta
    assert figures.parse_live_event(completed.model_dump_json()) == completed
    assert figures.parse_live_event('{"type": "session.created"}') is None
    assert figures.parse_live_event("not json") is None


def test_the_card_reading_is_used_and_total_memory() -> None:
    assert figures.parse_memory("671, 24576\n") == figures.MemoryReading(671, 24576)


def test_the_peak_is_the_highest_sample_above_the_baseline() -> None:
    assert figures.peak_over_baseline(700, [710, 3900, 3100]) == 3900 - 700


def test_a_peak_below_the_baseline_is_zero() -> None:
    assert figures.peak_over_baseline(700, [600]) == 0


def test_no_sample_gives_no_peak() -> None:
    with pytest.raises(ValueError, match="no memory sample"):
        figures.peak_over_baseline(700, [])


def taken(**changes: object) -> list[str]:
    held = {
        "speaking_ready": True,
        "hearing_ready": False,
        "file_status": 503,
        "file_code": RefusalCode.UNAVAILABLE.value,
        "live_status": 503,
        "spoke_bytes": 4096,
    }
    return figures.taken_card_failures(
        figures.TakenCardObservation(**{**held, **changes})  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"speaking_ready": False}, "speaking is not ready"),
        ({"hearing_ready": True}, "hearing is ready on a taken card"),
        ({"file_status": 200}, "the file route did not answer 503 unavailable"),
        ({"file_code": "busy"}, "the file route did not answer 503 unavailable"),
        ({"live_status": 101}, "the live upgrade did not answer 503"),
        ({"spoke_bytes": 0}, "speaking returned no audio"),
    ],
)
def test_a_taken_card_that_differs_from_cell_30_names_what_differs(
    change: dict[str, object], problem: str
) -> None:
    assert taken(**change) == [problem]


def test_a_taken_card_as_cell_30_says_has_no_failures() -> None:
    assert taken() == []


def test_a_stage_that_needs_the_card_is_not_applicable_on_the_cpu() -> None:
    calls: list[str] = []

    def measure() -> dict[str, object]:
        calls.append("measured")
        return {"peak_mib": 1}

    record = figures.stage_record("cpu", "cuda", measure)

    assert record == {"status": "not applicable", "reason": "the device is cpu"}
    assert calls == []


def test_a_stage_that_needs_the_card_is_measured_on_cuda() -> None:
    def measure() -> dict[str, object]:
        return {"peak_mib": 3200}

    assert figures.stage_record("cuda", "cuda", measure) == {
        "status": "measured",
        "peak_mib": 3200,
    }


def test_timing_and_error_records_carry_plain_numbers() -> None:
    timing = figures.Timing(1.0, 0.5, 4.0, 3.5)

    assert figures.timing_record(timing) == {
        "first_partial_from_start": 1.0,
        "first_partial_after_end": 0.5,
        "final_from_start": 4.0,
        "final_after_end": 3.5,
    }
    assert figures.error_record(figures.WordErrors(1, 4)) == {
        "errors": 1,
        "reference_words": 4,
        "rate": 0.25,
    }
