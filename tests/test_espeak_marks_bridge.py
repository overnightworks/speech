"""espeak-ng's word events through the locked piper-tts bridge (WIRE-152 to 156)."""

from __future__ import annotations

import ctypes
import json
import logging
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from speech_service import espeak_marks
from speech_service.espeak_marks import (
    EspeakEvents,
    EventKind,
    NativeEvent,
    Sentence,
    WordMarker,
    WordsOffReason,
    load_espeak_events,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from piper import PiperVoice

SAMPLE_RATE = 22_050
SAMPLES_PER_TOKEN = 100
OUTPUTS_WITH_ALIGNMENT = 2
PHONEMES_BEFORE_AND_AFTER = """
import json, sys
sys.path.insert(0, sys.argv[1])
from test_espeak_marks_bridge import a_voice, loaded

voice = a_voice("de")
before = voice.phonemize(sys.argv[2])
loaded(voice)
print(json.dumps([before, voice.phonemize(sys.argv[2])]))
"""

# What a bridge reports per pass: (kind, 1-based position, name) of each event.
type ScriptedEvents = Sequence[tuple[EventKind, int, bytes]]
SELF_TEST_PASSES: list[ScriptedEvents] = [
    [(EventKind.WORD, 1, b""), (EventKind.WORD, 5, b""), (EventKind.SENTENCE, 10, b"")],
    [(EventKind.WORD, 10, b"")],
]


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


def a_bridge(
    passes: Sequence[ScriptedEvents],
    *,
    version: str = espeak_marks.ESPEAK_VERSION,
    refusing: str | None = None,
) -> Callable[[str], ctypes.CDLL]:
    """An opener of a library whose passes report `passes`, one per `espeak_Synth`."""
    library = MagicMock()
    library.espeak_Info.return_value = version.encode("ascii")
    library.espeak_ng_InitializeOutput.return_value = 0
    library.espeak_ng_SetPhonemeEvents.return_value = 0
    if refusing is not None:
        getattr(library, refusing).return_value = 1
    callbacks: list[Callable[..., int]] = []
    remaining = iter(passes)

    def synth(text: bytes, *_arguments: object) -> int:
        if text not in {b"", espeak_marks.SETTLING_TEXT}:
            reported = next(remaining)
            native = (NativeEvent * (len(reported) + 1))()
            for slot, (kind, position, name) in zip(native, reported, strict=False):
                slot.type = kind
                slot.text_position = position
                slot.id.name = name
            callbacks[-1](None, 0, native)
        return 0

    library.espeak_SetSynthCallback.side_effect = callbacks.append
    library.espeak_Synth.side_effect = synth
    return lambda _path: library


def loaded(
    voice: PiperVoice, open_library: Callable[[str], ctypes.CDLL] = ctypes.CDLL
) -> EspeakEvents:
    events = load_espeak_events(voice, open_library)
    assert events is not None
    return events


@pytest.mark.parametrize("espeak_voice", ["de", "en-us", "en-gb-x-rp"])
def test_the_load_check_passes_on_the_locked_bridge(espeak_voice: str) -> None:
    """WIRE-156: the self-test text gives its known positions on the pinned bridge."""
    assert load_espeak_events(a_voice(espeak_voice)) is not None


def test_phonemizing_gives_the_same_phonemes_after_the_load_check_as_before() -> None:
    """Piper starts espeak once per process, so only a fresh process has a before."""
    text = "Ärger für Öl 😀 schön. Viele Tiere, z.B. Hunde, mögen Wasser."

    ran = subprocess.run(
        [
            sys.executable,
            "-c",
            PHONEMES_BEFORE_AND_AFTER,
            str(Path(__file__).parent),
            text,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )

    before, after = json.loads(ran.stdout)
    assert before
    assert after == before


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
    ("reason", "bridge"),
    [
        (WordsOffReason.VERSION, a_bridge(SELF_TEST_PASSES, version="0")),
        (
            WordsOffReason.SELF_TEST,
            a_bridge(
                [
                    [(EventKind.WORD, 2, b""), *SELF_TEST_PASSES[0][1:]],
                    SELF_TEST_PASSES[1],
                ]
            ),
        ),
        (
            WordsOffReason.SELF_TEST,
            a_bridge([SELF_TEST_PASSES[0], [(EventKind.WORD, 11, b"")]]),
        ),
        (
            WordsOffReason.SELF_TEST,
            a_bridge([SELF_TEST_PASSES[0][:2], SELF_TEST_PASSES[1]]),
        ),
        (
            WordsOffReason.SELF_TEST,
            a_bridge(
                [
                    SELF_TEST_PASSES[0],
                    [*SELF_TEST_PASSES[1], (EventKind.SENTENCE, 15, b"")],
                ]
            ),
        ),
    ],
    ids=[
        "version",
        "word at another position",
        "second sentence at another position",
        "no second sentence announced",
        "a sentence announced after the last",
    ],
)
def test_an_unexpected_bridge_switches_words_off_once_and_loudly(
    reason: WordsOffReason,
    bridge: Callable[[str], ctypes.CDLL],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """WIRE-156: a failed load check is one error line; the audio plays on."""
    with caplog.at_level(logging.ERROR):
        events = load_espeak_events(a_voice("en-us"), bridge)

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
def test_a_bridge_that_refuses_its_setup_switches_words_off(refusing: str) -> None:
    bridge = a_bridge(SELF_TEST_PASSES, refusing=refusing)

    assert load_espeak_events(a_voice("en-us"), bridge) is None


def test_a_pass_whose_callback_fails_gives_no_pass() -> None:
    """WIRE-155: a failure of the callback is caught, and the audio would go on."""
    voice = a_voice("de")
    not_utf8 = [(EventKind.WORD, 1, b""), (EventKind.PHONEME, 1, b"\xff")]
    events = loaded(voice, a_bridge([*SELF_TEST_PASSES, not_utf8]))

    assert events.read("Ja.", "de", 1) is None


def test_a_voice_espeak_does_not_know_gives_no_pass(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events = loaded(a_voice("de"))

    with caplog.at_level(logging.WARNING):
        found = events.read("Ja.", "no-such-voice", 1)

    assert found is None
    assert [record.getMessage() for record in caplog.records] == ["marks.pass_failed"]


def test_phonemizing_after_a_pass_that_stopped_at_the_next_sentence_is_unchanged() -> (
    None
):
    """WIRE-156: a pass is no state the next answer's phonemes can depend on."""
    voice = a_voice("de")
    events = loaded(voice)
    answer = "Ja. Nein. Doch."
    before = voice.phonemize(answer)

    two_long_sentences = " ".join(
        [
            "Die Kamera zeigt den Vortragenden, während im Hintergrund die nächste Folie vorbereitet wird."
        ]
        * 2
    )

    found = events.read(two_long_sentences, "de", 1)

    assert found is not None
    assert found.next_start is not None
    assert voice.phonemize(answer) == before


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
