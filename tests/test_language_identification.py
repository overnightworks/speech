"""The shipped identifier names a language of the set above its voiced floor."""

import struct

import pytest

from speech_client.wire import HearingLanguage
from speech_service.hearing import (
    HEARING_SAMPLE_RATE,
    LanguageScores,
    ScoreIdentifier,
    Utterance,
)
from speech_service.pcm import VOICED_FRAME_SECONDS

DE = HearingLanguage.DE
EN = HearingLanguage.EN
FRAME_SAMPLES = round(HEARING_SAMPLE_RATE * VOICED_FRAME_SECONDS)
FLOOR_FRAMES = 50
FLOOR = FLOOR_FRAMES * VOICED_FRAME_SECONDS
LOUD = 5000


def _frames(count: int, level: int) -> bytes:
    return struct.pack(f"<{count * FRAME_SAMPLES}h", *[level] * (count * FRAME_SAMPLES))


def _speech(frames: int = FLOOR_FRAMES + 20) -> bytes:
    return _frames(frames, LOUD)


def _utterance(pcm: bytes, *, de: float, en: float) -> Utterance:
    return Utterance(pcm, LanguageScores({DE: de, EN: en}))


def test_a_language_outside_the_set_is_never_chosen() -> None:
    """WIRE-135: no language outside the set is ever chosen."""
    scores = LanguageScores({DE: 0.04, EN: 0.03})
    identifier = ScoreIdentifier(FLOOR)

    assert identifier.identify(Utterance(_speech(), scores), (DE, EN)) is DE
    assert identifier.identify(Utterance(_speech(), scores), (EN,)) is EN


@pytest.mark.parametrize(
    ("de", "en", "named"),
    [(0.51, 0.49, DE), (0.49, 0.51, EN)],
    ids=["german-leads", "english-leads"],
)
def test_above_the_floor_the_higher_score_is_named_either_way(
    de: float, en: float, named: HearingLanguage
) -> None:
    identifier = ScoreIdentifier(FLOOR)

    assert identifier.identify(_utterance(_speech(), de=de, en=en), (DE, EN)) is named
    assert identifier.identify(_utterance(_speech(), de=de, en=en), (EN, DE)) is named


@pytest.mark.parametrize(
    ("de", "en"), [(0.4, 0.4), (0.0, 0.0)], ids=["tie", "all-zero"]
)
def test_a_tie_or_scores_of_zero_are_not_known(de: float, en: float) -> None:
    identifier = ScoreIdentifier(FLOOR)

    assert identifier.identify(_utterance(_speech(), de=de, en=en), (DE, EN)) is None


@pytest.mark.parametrize(
    ("frames", "named"),
    [(FLOOR_FRAMES, EN), (FLOOR_FRAMES - 1, None)],
    ids=["at-the-floor", "one-frame-below"],
)
def test_the_floor_is_met_at_exactly_its_voiced_seconds_and_not_one_frame_below(
    frames: int, named: HearingLanguage | None
) -> None:
    identifier = ScoreIdentifier(FLOOR)
    utterance = _utterance(_speech(frames), de=0.1, en=0.9)

    assert identifier.identify(utterance, (DE, EN)) is named


def test_silence_holds_no_voiced_speech_so_it_is_not_known() -> None:
    identifier = ScoreIdentifier(FLOOR)
    utterance = _utterance(_frames(500, 0), de=0.1, en=0.9)

    assert identifier.identify(utterance, (EN, DE)) is None
    assert identifier.identify(_utterance(b"", de=0.1, en=0.9), (EN, DE)) is None


def test_a_short_word_after_a_long_silence_is_judged_by_its_speech_not_its_length() -> (
    None
):
    word = _speech(FLOOR_FRAMES - 1)
    pcm = _frames(200, 0) + word
    identifier = ScoreIdentifier(FLOOR)

    assert identifier.identify(_utterance(pcm, de=0.2, en=0.8), (DE, EN)) is None
