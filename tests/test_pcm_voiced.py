"""The voiced seconds of a turn count frames of speech, never the turn's length."""

import struct

import pytest

from speech_service.pcm import (
    SPEECH_RMS,
    VOICED_FRAME_SECONDS,
    is_silence,
    rms,
    voiced_seconds,
)

RATE = 16_000
FRAME_SAMPLES = round(RATE * VOICED_FRAME_SECONDS)
LOUD = 5000
JUST_SPEECH = 656
JUST_SILENCE = 655


def _frames(count: int, level: int) -> bytes:
    return struct.pack(f"<{count * FRAME_SAMPLES}h", *[level] * (count * FRAME_SAMPLES))


def test_leading_and_trailing_silence_is_not_counted() -> None:
    pcm = _frames(40, 0) + _frames(10, LOUD) + _frames(60, 0)

    assert voiced_seconds(pcm, RATE) == pytest.approx(10 * VOICED_FRAME_SECONDS)


def test_a_pause_inside_the_speech_is_not_counted() -> None:
    pcm = _frames(5, LOUD) + _frames(7, 0) + _frames(5, LOUD)

    assert voiced_seconds(pcm, RATE) == pytest.approx(10 * VOICED_FRAME_SECONDS)


@pytest.mark.parametrize(
    ("level", "frames"),
    [(JUST_SPEECH, 4), (JUST_SILENCE, 0)],
    ids=["just-over-counts", "just-under-does-not"],
)
def test_a_frame_counts_by_the_comparison_the_session_makes_at_its_speech_threshold(
    level: int, frames: int
) -> None:
    pcm = _frames(4, level)

    assert voiced_seconds(pcm, RATE) == pytest.approx(frames * VOICED_FRAME_SECONDS)
    assert is_silence(pcm, SPEECH_RMS) is (frames == 0)


def test_silence_and_nothing_hold_no_voiced_seconds() -> None:
    assert voiced_seconds(_frames(30, 0), RATE) == 0.0
    assert voiced_seconds(b"", RATE) == 0.0


def test_the_voiced_seconds_of_a_longer_turn_with_a_short_word_stay_short() -> None:
    pcm = _frames(166, 0) + _frames(20, LOUD) + _frames(10, 0)

    assert voiced_seconds(pcm, RATE) == pytest.approx(20 * VOICED_FRAME_SECONDS)


def test_the_root_mean_square_of_loud_samples_does_not_overflow_sixteen_bits() -> None:
    pcm = struct.pack("<4h", 32767, -32768, 32767, -32768)

    assert rms(pcm) == pytest.approx(1.0, abs=1e-3)


def test_less_than_one_sample_has_no_level() -> None:
    assert rms(b"") == rms(b"\x7f") == 0.0


def test_an_odd_trailing_byte_is_ignored() -> None:
    pcm = struct.pack("<2h", 3000, 3000)

    assert rms(pcm + b"\x7f") == rms(pcm)
