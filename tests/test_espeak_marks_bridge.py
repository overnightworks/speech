"""espeak-ng's word events through the locked piper-tts bridge (WIRE-152 to 156)."""

from __future__ import annotations

import logging
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from speech_service import espeak_marks
from speech_service.espeak_marks import (
    EspeakEvents,
    Sentence,
    WordMarker,
    WordsOffReason,
    load_espeak_events,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from piper import PiperVoice

SAMPLE_RATE = 22_050
SAMPLES_PER_TOKEN = 100
OUTPUTS_WITH_ALIGNMENT = 2


def a_voice(espeak_voice: str, *, outputs: int = OUTPUTS_WITH_ALIGNMENT) -> PiperVoice:
    """A Piper voice that phonemizes with espeak and has no model to run."""
    from piper import PiperVoice
    from piper.config import PhonemeType, PiperConfig

    config = PiperConfig(
        num_symbols=0,
        num_speakers=1,
        sample_rate=SAMPLE_RATE,
        espeak_voice=espeak_voice,
        phoneme_id_map={},
        phoneme_type=PhonemeType.ESPEAK,
    )
    session = SimpleNamespace(get_outputs=lambda: [object()] * outputs)
    return PiperVoice(session=session, config=config)  # type: ignore[arg-type]


def words_of(
    voice: PiperVoice, events: EspeakEvents, text: str
) -> list[list[str] | None]:
    """The word texts of each sentence Piper phonemizes `text` into."""
    marker = WordMarker(
        text,
        partial(events.read, text, voice.config.espeak_voice),
        SAMPLE_RATE,
    )
    texts: list[list[str] | None] = []
    for phonemes in voice.phonemize(text):
        samples = (SAMPLES_PER_TOKEN,) * (len(phonemes) + 2)
        marks = marker.after(Sentence(tuple(phonemes), samples, sum(samples)))
        assert all(
            text[word.start_char : word.end_char] == word.text
            for word in marks.words or ()
        )
        texts.append(None if marks.words is None else [w.text for w in marks.words])
    return texts


def loaded(voice: PiperVoice) -> EspeakEvents:
    events = load_espeak_events(voice)
    assert events is not None
    return events


@pytest.mark.parametrize("espeak_voice", ["de", "en-us", "en-gb-x-rp"])
def test_the_load_check_passes_on_the_locked_bridge(espeak_voice: str) -> None:
    """WIRE-156: the self-test text gives its known positions on the pinned bridge."""
    assert load_espeak_events(a_voice(espeak_voice)) is not None


def test_phonemizing_gives_the_same_phonemes_after_the_load_check_as_before() -> None:
    voice = a_voice("de")
    text = "Ärger für Öl 😀 schön. Viele Tiere, z.B. Hunde, mögen Wasser."
    before = voice.phonemize(text)

    loaded(voice)

    assert voice.phonemize(text) == before


@pytest.mark.parametrize(
    ("espeak_voice", "text", "expected"),
    [
        ("de", "Ja. Nein. Doch.", [["Ja"], ["Nein"], ["Doch"]]),
        ("de", "Ja. Ja. Ja.", [["Ja"], ["Ja"], ["Ja"]]),
        ("de", "Ja.\r\nNein.", [["Ja"], ["Nein"]]),
        (
            "de",
            "Das  ist   gut.\nUnd   weiter.",
            [["Das", "ist", "gut"], ["Und", "weiter"]],
        ),
        ("de", "Ärger für Öl 😀 schön.", [["Ärger", "für", "Öl", "😀", "schön"]]),
        ("de", "Schön.", [["Schön"]]),
        (
            "de",
            "Im Jahr 1990 fiel die Mauer.",
            [["Im", "Jahr", "1990", "fiel", "die", "Mauer"]],
        ),
        (
            "de",
            "Viele Tiere, z.B. Hunde, mögen Wasser.",
            [["Viele", "Tiere", "z.B"], ["Hunde", "mögen", "Wasser"]],
        ),
        (
            "de",
            "Das Wetter ist heute schön",
            [["Das", "Wetter", "ist", "heute", "schön"]],
        ),
        ("en-us", "One, two; three: four.", [["One", "two", "three", "four"]]),
        ("en-us", "In 1990 the the in the", [["In", "1990", "the", "the", "in the"]]),
        ("en-us", "He said [[ hello ]] there.", [None]),
    ],
)
def test_the_words_of_real_sentences_are_the_spans_of_the_input(
    espeak_voice: str, text: str, expected: list[list[str] | None]
) -> None:
    """WIRE-152, WIRE-154: each word's text is `input[start_char:end_char]`."""
    voice = a_voice(espeak_voice)

    assert words_of(voice, loaded(voice), text) == expected


@pytest.mark.parametrize(
    ("reason", "arrange"),
    [
        (
            WordsOffReason.VERSION,
            lambda mp: mp.setattr(espeak_marks, "ESPEAK_VERSION", "0"),
        ),
        (
            WordsOffReason.SELF_TEST,
            lambda mp: mp.setattr(espeak_marks, "_SELF_TEST_PASSES", (((2,), None),)),
        ),
        (
            WordsOffReason.SELF_TEST,
            lambda mp: mp.setattr(
                espeak_marks, "_SELF_TEST_PASSES", (((1, 5), 10), ((2,), None))
            ),
        ),
        (
            WordsOffReason.SELF_TEST,
            lambda mp: mp.setattr(
                espeak_marks,
                "_SELF_TEST_PASSES",
                (((1, 5), 10), ((10,), None), ((10,), None)),
            ),
        ),
    ],
)
def test_an_unexpected_bridge_switches_words_off_once_and_loudly(
    reason: WordsOffReason,
    arrange: Callable[[pytest.MonkeyPatch], None],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """WIRE-156: a failed load check is one error line; the audio plays on."""
    arrange(monkeypatch)

    with caplog.at_level(logging.ERROR):
        events = load_espeak_events(a_voice("en-us"))

    assert events is None
    assert [record.reason for record in caplog.records] == [reason.value]  # type: ignore[attr-defined]


def test_a_voice_without_the_alignment_output_switches_words_off() -> None:
    """WIRE-156: a model that gives no alignment output gives no words."""
    assert load_espeak_events(a_voice("en-us", outputs=1)) is None


@pytest.mark.parametrize(
    ("reason", "replacement"),
    [(WordsOffReason.LIBRARY, None), (WordsOffReason.LOCK, object())],
    ids=["missing", "not a lock"],
)
def test_a_lock_that_piper_does_not_have_switches_words_off(
    reason: WordsOffReason,
    replacement: object,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """WIRE-156: the load check needs the lock Piper phonemizes under."""
    import piper.voice

    if replacement is None:
        monkeypatch.delattr(piper.voice, "_ESPEAK_PHONEMIZER_LOCK")
    else:
        monkeypatch.setattr(piper.voice, "_ESPEAK_PHONEMIZER_LOCK", replacement)

    with caplog.at_level(logging.ERROR):
        events = load_espeak_events(a_voice("en-us"))

    assert events is None
    assert [record.reason for record in caplog.records] == [reason.value]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "refusing", ["espeak_ng_InitializeOutput", "espeak_ng_SetPhonemeEvents"]
)
def test_a_bridge_that_refuses_its_setup_switches_words_off(
    refusing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = MagicMock()
    library.espeak_Info.return_value = espeak_marks.ESPEAK_VERSION.encode("ascii")
    library.espeak_ng_InitializeOutput.return_value = 0
    library.espeak_ng_SetPhonemeEvents.return_value = 0
    getattr(library, refusing).return_value = 1
    monkeypatch.setattr(espeak_marks.ctypes, "CDLL", lambda _path: library)

    assert load_espeak_events(a_voice("en-us")) is None


def test_a_callback_that_fails_nulls_its_sentence_and_every_later_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WIRE-155: a failure of the callback is caught, and the audio would go on."""
    voice = a_voice("de")
    events = loaded(voice)

    def failing(*_arguments: object) -> int:
        raise RuntimeError

    monkeypatch.setattr(espeak_marks._Collector, "_collect", failing)  # noqa: SLF001

    assert words_of(voice, events, "Ja. Nein.") == [None, None]


def test_a_voice_espeak_does_not_know_gives_no_pass(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events = loaded(a_voice("de"))

    with caplog.at_level(logging.WARNING):
        found = events.read("Ja.", "no-such-voice", 1)

    assert found is None
    assert [record.getMessage() for record in caplog.records] == ["marks.pass_failed"]


def test_a_sentence_spoken_after_much_phonemizing_still_has_all_its_words() -> None:
    """WIRE-155: espeak's queue of stale events never costs a pass its first events."""
    voice = a_voice("de")
    events = loaded(voice)
    text = "Die Kamera zeigt den Vortragenden."
    for _ in range(300):
        voice.phonemize(text)

    assert words_of(voice, events, text) == [
        ["Die", "Kamera", "zeigt", "den", "Vortragenden"]
    ]
