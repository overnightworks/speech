"""The adapter names a language in one encoder pass and decodes as faster-whisper."""

import math
import struct
from pathlib import Path

import numpy as np
import pytest
from faster_whisper import WhisperModel

from speech_client.wire import HearingLanguage
from speech_service.hearing import (
    HEARING_SAMPLE_RATE,
    Heard,
    LanguageIdentifier,
    LanguageScores,
    ScoreIdentifier,
    Utterance,
    WhisperHearing,
)
from speech_service.hearing_models import HearingModel, voiced_floor
from speech_service.pcm import VOICED_FRAME_SECONDS
from tests.support.whisper_support import StandInWhisper

DE = HearingLanguage.DE
EN = HearingLanguage.EN
FLOOR = voiced_floor(HearingModel.SMALL)
FRAME_SAMPLES = round(HEARING_SAMPLE_RATE * VOICED_FRAME_SECONDS)
# The tolerance keeps a floor that is an exact frame multiple from rounding up a frame.
FLOOR_FRAMES = math.ceil(FLOOR / VOICED_FRAME_SECONDS - 1e-9)
LOUD = 5000


class _ReadsOnlyThePcm:
    """A standalone identifier: it answers from the audio and ignores the scores."""

    def __init__(self, *answers: HearingLanguage | None) -> None:
        self._answers = iter(answers)
        self.asked: list[tuple[Utterance, tuple[HearingLanguage, ...]]] = []

    def identify(
        self, utterance: Utterance, allowed: tuple[HearingLanguage, ...]
    ) -> HearingLanguage | None:
        self.asked.append((utterance, allowed))
        return next(self._answers)


def _frames(count: int, level: int) -> bytes:
    return struct.pack(f"<{count * FRAME_SAMPLES}h", *[level] * (count * FRAME_SAMPLES))


def _speech(frames: int) -> bytes:
    """Frames of a noise-like signal, so the features differ per frame."""
    generator = np.random.default_rng(frames)
    samples = generator.integers(-LOUD, LOUD, frames * FRAME_SAMPLES, dtype=np.int16)
    return samples.tobytes()


def _engine(
    stand_in: StandInWhisper, identifier: LanguageIdentifier | None = None
) -> WhisperHearing:
    engine = WhisperHearing(
        HearingModel.SMALL,
        Path("model"),
        "cpu",
        identifier or ScoreIdentifier(FLOOR),
        stand_in.libraries,
    )
    engine.load()
    return engine


def _library_transcribe(stand_in: StandInWhisper, pcm: bytes, language: str) -> None:
    """Today's call: the library's own `transcribe` with the service's options."""
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _info = WhisperModel.transcribe(
        stand_in,
        audio,
        language=language,
        beam_size=1,
        temperature=0.0,
        vad_filter=False,
        without_timestamps=True,
        condition_on_previous_text=False,
    )
    assert "".join(segment.text for segment in segments).strip() == "Guten Tag"


@pytest.fixture
def stand_in() -> StandInWhisper:
    return StandInWhisper()


@pytest.mark.parametrize("seconds", [1.0, 40.0], ids=["one-window", "over-30-seconds"])
@pytest.mark.parametrize("language", list(HearingLanguage))
def test_a_one_element_set_decodes_as_faster_whispers_own_transcribe_does(
    stand_in: StandInWhisper, language: HearingLanguage, seconds: float
) -> None:
    """WIRE-38: greedy at temperature 0, with the options of faster-whisper's call."""
    pcm = _speech(round(seconds / VOICED_FRAME_SECONDS))
    _library_transcribe(stand_in, pcm, language.value)
    expected = stand_in.decodes.pop()

    heard = _engine(stand_in).transcribe(pcm, (language,))

    (decode,) = stand_in.decodes
    assert heard == Heard("Guten Tag", language)
    assert decode.language_code == expected.language_code == language.value
    assert np.array_equal(decode.features, expected.features)
    assert decode.options == expected.options
    assert decode.log_progress == expected.log_progress
    assert decode.encoder_output is None
    assert stand_in.encodes == []
    assert stand_in.detections == []


@pytest.mark.parametrize("seconds", [1.0, 40.0], ids=["one-window", "over-30-seconds"])
def test_a_set_of_two_decodes_with_the_same_options_from_its_one_encoder_pass(
    stand_in: StandInWhisper, seconds: float
) -> None:
    pcm = _speech(round(seconds / VOICED_FRAME_SECONDS))
    _library_transcribe(stand_in, pcm, "de")
    expected = stand_in.decodes.pop()

    heard = _engine(stand_in).transcribe(pcm, (DE, EN))

    (decode,) = stand_in.decodes
    assert heard.language is DE
    assert decode.language_code == "de"
    assert np.array_equal(decode.features, expected.features)
    assert decode.options == expected.options
    assert len(stand_in.encodes) == 1
    assert decode.encoder_output is not None
    assert decode.encoder_output.number == 1
    assert stand_in.detections == [decode.encoder_output]


@pytest.mark.parametrize(
    "frames", [FLOOR_FRAMES + 20, FLOOR_FRAMES - 1], ids=["above-the-floor", "below"]
)
def test_a_decode_of_a_set_of_two_encodes_once_also_when_the_identifier_gives_up(
    stand_in: StandInWhisper, frames: int
) -> None:
    pcm = _speech(frames)

    _engine(stand_in).transcribe(pcm, (DE, EN))

    (decode,) = stand_in.decodes
    assert len(stand_in.encodes) == 1
    assert len(stand_in.detections) == 1
    assert decode.encoder_output == stand_in.detections[0]


def test_the_encoder_is_fed_one_window_of_the_features_the_decode_gets(
    stand_in: StandInWhisper,
) -> None:
    pcm = _speech(round(40.0 / VOICED_FRAME_SECONDS))

    _engine(stand_in).transcribe(pcm, (DE, EN))

    (decode,) = stand_in.decodes
    (encoded,) = stand_in.encodes
    window = stand_in.feature_extractor.nb_max_frames
    assert encoded.shape[-1] == window
    assert np.array_equal(encoded, decode.features[:, :window])


def test_the_options_are_built_anew_for_every_decode(stand_in: StandInWhisper) -> None:
    engine = _engine(stand_in)

    engine.transcribe(_speech(40), (DE,))
    engine.transcribe(_speech(40), (DE,))

    first, second = (decode.options for decode in stand_in.decodes)
    assert first is not second
    assert first.suppress_tokens is not second.suppress_tokens


def test_a_set_of_one_never_asks_the_identifier(stand_in: StandInWhisper) -> None:
    identifier = _ReadsOnlyThePcm()

    _engine(stand_in, identifier).transcribe(_speech(100), (EN,))

    assert identifier.asked == []
    assert stand_in.detections == []


def test_the_identifier_is_given_the_pcm_and_the_probabilities_of_the_set_only(
    stand_in: StandInWhisper,
) -> None:
    """WIRE-135: no language outside the set is chosen; no Whisper token crosses."""
    stand_in.probabilities = {"fr": 0.5, "de": 0.1, "en": 0.4}
    identifier = _ReadsOnlyThePcm(None)
    pcm = _speech(100)

    _engine(stand_in, identifier).transcribe(pcm, (EN, DE))

    ((utterance, allowed),) = identifier.asked
    assert allowed == (EN, DE)
    assert utterance == Utterance(pcm, LanguageScores({DE: 0.1, EN: 0.4}))


@pytest.mark.parametrize(
    ("allowed", "language"),
    [((EN, DE), EN), ((DE, EN), DE)],
    ids=["english-main", "german-main"],
)
def test_when_the_identifier_does_not_know_the_first_language_of_the_set_holds(
    stand_in: StandInWhisper,
    allowed: tuple[HearingLanguage, ...],
    language: HearingLanguage,
) -> None:
    """WIRE-135: not knowing means the main language."""
    heard = _engine(stand_in, _ReadsOnlyThePcm(None)).transcribe(_speech(100), allowed)

    assert heard.language is language
    assert stand_in.decodes[0].language_code == language.value


def test_a_word_after_a_long_pause_below_the_floor_is_heard_in_the_main_language(
    stand_in: StandInWhisper,
) -> None:
    """A turn counts its voiced seconds, not its length."""
    stand_in.probabilities = {"de": 0.1, "en": 0.9}
    five_seconds = round(5.0 / VOICED_FRAME_SECONDS)
    pcm = _frames(five_seconds, 0) + _speech(FLOOR_FRAMES - 1)

    heard = _engine(stand_in).transcribe(pcm, (DE, EN))

    assert heard.language is DE


def test_above_the_floor_the_language_the_scores_favour_is_heard(
    stand_in: StandInWhisper,
) -> None:
    stand_in.probabilities = {"de": 0.1, "en": 0.9}

    heard = _engine(stand_in).transcribe(_speech(FLOOR_FRAMES + 20), (DE, EN))

    assert heard == Heard("Guten Tag", EN)
    assert stand_in.decodes[0].language_code == "en"


def test_an_identifier_that_reads_only_the_pcm_can_replace_the_shipped_one(
    stand_in: StandInWhisper,
) -> None:
    """The identifier is exchangeable behind its port; the decode is unchanged."""
    stand_in.probabilities = {"de": 0.9, "en": 0.1}
    engine = _engine(stand_in, _ReadsOnlyThePcm(EN, None))

    first = engine.transcribe(_speech(100), (DE, EN))
    second = engine.transcribe(_speech(100), (DE, EN))

    assert (first.language, second.language) == (EN, DE)


def _utterance(pcm: bytes, *, de: float, en: float) -> Utterance:
    return Utterance(pcm, LanguageScores({DE: de, EN: en}))


def test_a_language_outside_the_set_is_never_chosen() -> None:
    """WIRE-135: no language outside the set is ever chosen."""
    scores = LanguageScores({DE: 0.04, EN: 0.03})

    named = ScoreIdentifier(FLOOR).identify(
        Utterance(_speech(FLOOR_FRAMES + 20), scores), (EN,)
    )

    assert named is EN


@pytest.mark.parametrize(
    ("de", "en", "named"),
    [(0.51, 0.49, DE), (0.49, 0.51, EN)],
    ids=["german-leads", "english-leads"],
)
def test_above_the_floor_the_higher_score_is_named_either_way(
    de: float, en: float, named: HearingLanguage
) -> None:
    identifier = ScoreIdentifier(FLOOR)
    utterance = _utterance(_speech(FLOOR_FRAMES + 20), de=de, en=en)

    assert identifier.identify(utterance, (DE, EN)) is named
    assert identifier.identify(utterance, (EN, DE)) is named


@pytest.mark.parametrize(
    ("de", "en"), [(0.4, 0.4), (0.0, 0.0)], ids=["tie", "all-zero"]
)
def test_a_tie_or_scores_of_zero_are_not_known(de: float, en: float) -> None:
    utterance = _utterance(_speech(FLOOR_FRAMES + 20), de=de, en=en)

    assert ScoreIdentifier(FLOOR).identify(utterance, (DE, EN)) is None


@pytest.mark.parametrize(
    ("frames", "named"),
    [(FLOOR_FRAMES, EN), (FLOOR_FRAMES - 1, None)],
    ids=["at-the-floor", "one-frame-below"],
)
def test_the_floor_is_met_at_exactly_its_voiced_seconds_and_not_one_frame_below(
    frames: int, named: HearingLanguage | None
) -> None:
    utterance = _utterance(_speech(frames), de=0.1, en=0.9)

    assert ScoreIdentifier(FLOOR).identify(utterance, (DE, EN)) is named


def test_silence_holds_no_voiced_speech_so_it_is_not_known() -> None:
    identifier = ScoreIdentifier(FLOOR)

    assert (
        identifier.identify(_utterance(_frames(500, 0), de=0.1, en=0.9), (EN, DE))
        is None
    )
    assert identifier.identify(_utterance(b"", de=0.1, en=0.9), (EN, DE)) is None
