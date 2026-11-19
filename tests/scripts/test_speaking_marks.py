"""The marks proof judges a `sse` answer by the wire's ruled sentences."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

import pytest

import speaking_marks as marks
from speech_client.wire import Health, Language, ResponseFormat
from speech_service.ogg_opus import ogg_opus_chunks
from speech_service.pcm import wav_header
from tests.support.fakes_support import SPEAK_SAMPLE_RATE, sine_pcm

if TYPE_CHECKING:
    from collections.abc import Callable

    from speech_client.wire import VoiceId

GERMAN_VOICE = "de_DE-thorsten-medium"
ENGLISH_VOICE = "en_US-norman-medium"
TEXT = "Ja gern. Nein danke."
SENTENCE_SECONDS = 0.5
WORDS = [
    [("Ja", 0, 2, 0.05, 0.2), ("gern", 3, 7, 0.25, 0.45)],
    [("Nein", 9, 13, 0.55, 0.75), ("danke", 14, 19, 0.8, 0.95)],
]


def word(entry: tuple[str, int, int, float, float]) -> dict[str, object]:
    text, start_char, end_char, start, end = entry
    return {
        "text": text,
        "start_char": start_char,
        "end_char": end_char,
        "start": start,
        "end": end,
    }


def delta(audio: bytes) -> dict[str, object]:
    return {"type": "speech.audio.delta", "audio": base64.b64encode(audio).decode()}


SOUNDS = [
    [
        ("j", 0.03, 0.12),
        ("a", 0.12, 0.2),
        ("g", 0.25, 0.3),
        ("\u025b", 0.3, 0.35),
        ("\u0250", 0.35, 0.4),
        ("n", 0.4, 0.45),
    ],
    [("n", 0.55, 0.6), ("a\u026a", 0.6, 0.75), ("n", 0.75, 0.75), ("d", 0.8, 0.95)],
]


def sound(entry: tuple[str, float, float]) -> dict[str, object]:
    ipa, start, end = entry
    return {"ipa": ipa, "start": start, "end": end}


def sentence_marks(index: int) -> dict[str, object]:
    return {
        "type": "speech.audio.marks",
        "words": [word(entry) for entry in WORDS[index]],
        "sounds": [sound(entry) for entry in SOUNDS[index]],
    }


DONE = {"type": "speech.audio.done"}


def framed(events: list[dict[str, object]]) -> bytes:
    return b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)


def wav_events() -> list[dict[str, object]]:
    pcm = [sine_pcm(SENTENCE_SECONDS, SPEAK_SAMPLE_RATE) for _ in WORDS]
    return [
        delta(wav_header(SPEAK_SAMPLE_RATE) + pcm[0]),
        sentence_marks(0),
        delta(pcm[1]),
        sentence_marks(1),
        DONE,
    ]


def opus_events() -> list[dict[str, object]]:
    pcm = [sine_pcm(SENTENCE_SECONDS, SPEAK_SAMPLE_RATE) for _ in WORDS]
    pages = list(ogg_opus_chunks(pcm, SPEAK_SAMPLE_RATE, marked=True))
    return [
        delta(pages[0]),
        sentence_marks(0),
        *(delta(page) for page in pages[1:]),
        sentence_marks(1),
        DONE,
    ]


def judged(
    events: list[dict[str, object]], response_format: ResponseFormat
) -> marks.MarksFigures:
    return marks.judge_sse_answer(framed(events), TEXT, response_format, sentences=2)


def test_a_wav_answer_with_marks_in_order_is_judged_by_its_figures() -> None:
    figures = judged(wav_events(), ResponseFormat.WAV)

    assert (figures.marks, figures.words, figures.sounds) == (2, 4, 10)
    assert figures.audio_seconds == pytest.approx(2 * SENTENCE_SECONDS, abs=0.01)


def test_an_opus_answer_with_marks_in_order_is_judged_by_its_figures() -> None:
    figures = judged(opus_events(), ResponseFormat.OPUS)

    assert (figures.marks, figures.words, figures.sounds) == (2, 4, 10)


def replaced(index: int, event: dict[str, object]) -> Callable[[list], list]:
    def change(events: list[dict[str, object]]) -> list[dict[str, object]]:
        return [*events[:index], event, *events[index + 1 :]]

    return change


def with_sound(
    index: int, position: int, entry: tuple[str, float, float]
) -> Callable[[list], list]:
    def change(events: list[dict[str, object]]) -> list[dict[str, object]]:
        marked = dict(events[index])
        listed = list(marked["sounds"])  # type: ignore[call-overload]
        listed[position] = sound(entry)
        marked["sounds"] = listed
        return replaced(index, marked)(events)

    return change


def with_word(
    index: int, position: int, entry: tuple[str, int, int, float, float]
) -> Callable[[list], list]:
    def change(events: list[dict[str, object]]) -> list[dict[str, object]]:
        marked = dict(events[index])
        listed = list(marked["words"])  # type: ignore[call-overload]
        listed[position] = word(entry)
        marked["words"] = listed
        return replaced(index, marked)(events)

    return change


BAD_ANSWERS: dict[str, Callable[[list], list]] = {
    "no done": lambda events: events[:-1],
    "done twice": lambda events: [*events, DONE],
    "done carries usage": replaced(4, {**DONE, "usage": {}}),
    "audio after done": lambda events: [*events, events[0]],
    "marks before any audio": lambda events: [events[1], events[0], *events[2:]],
    "an unknown event": replaced(2, {"type": "speech.audio.other"}),
    "words null": replaced(1, {**sentence_marks(0), "words": None}),
    "sounds null": replaced(1, {**sentence_marks(0), "sounds": None}),
    "no sounds for a sentence that holds words": replaced(
        1, {**sentence_marks(0), "sounds": []}
    ),
    "a sound with its stress mark": with_sound(1, 0, ("\u02c8j", 0.03, 0.12)),
    "a sound that is not NFC": with_sound(1, 0, ("a\u0303", 0.03, 0.12)),
    "a sound without text": with_sound(1, 0, ("", 0.03, 0.12)),
    "a sound start that decreases": with_sound(1, 1, ("a", 0.02, 0.2)),
    "a sound that ends before its start": with_sound(1, 1, ("a", 0.2, 0.12)),
    "a sound after the audio": with_sound(3, 3, ("d", 0.8, 1.5)),
    "a sound outside every word": with_sound(1, 1, ("a", 0.21, 0.24)),
    "a sound that says it is estimated": replaced(
        1,
        {
            **sentence_marks(0),
            "sounds": [
                {**sound(SOUNDS[0][0]), "estimated": False},
                *map(sound, SOUNDS[0][1:]),
            ],
        },
    ),
    "one marks event too few": lambda events: [*events[:3], *events[4:]],
    "a span that is not its text": with_word(1, 1, ("gern", 3, 8, 0.25, 0.45)),
    "a start that does not increase": with_word(3, 0, ("Nein", 9, 13, 0.1, 0.75)),
    "an end after the audio": with_word(3, 1, ("danke", 14, 19, 0.8, 1.5)),
    "an end before its start": with_word(1, 0, ("Ja", 0, 2, 0.2, 0.05)),
    "a word of the text left out": with_word(1, 1, ("Ja", 0, 2, 0.25, 0.45)),
}


@pytest.mark.parametrize("change", list(BAD_ANSWERS.values()), ids=list(BAD_ANSWERS))
def test_an_answer_that_breaks_a_ruled_sentence_fails_the_proof(
    change: Callable[[list], list],
) -> None:
    events = change(wav_events())

    with pytest.raises(SystemExit):
        judged(events, ResponseFormat.WAV)


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"data: {}\n",
        b"event: x\n\n",
        b"data: {}\n\ndata: {}",
        b"data: not json\n\n",
    ],
    ids=["empty", "no blank line", "other field", "unfinished event", "not json"],
)
def test_a_body_that_is_not_framed_as_events_fails_the_proof(body: bytes) -> None:
    with pytest.raises(SystemExit):
        marks.judge_sse_answer(body, TEXT, ResponseFormat.WAV, sentences=2)


def test_audio_that_is_not_the_format_asked_for_fails_the_proof() -> None:
    events = wav_events()

    with pytest.raises(SystemExit):
        judged(events, ResponseFormat.OPUS)


def test_every_voice_is_proved_in_every_format_with_its_own_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    asked: list[tuple[str, VoiceId, ResponseFormat]] = []

    def speak(text: str, voice: VoiceId, response_format: ResponseFormat) -> bytes:
        asked.append((text, voice, response_format))
        events = (
            opus_events() if response_format is ResponseFormat.OPUS else wav_events()
        )
        return framed(events)

    record = marks.prove_marks(
        speak,
        [GERMAN_VOICE, ENGLISH_VOICE],
        {Language.DE: TEXT, Language.EN: TEXT},
        sentences=2,
    )

    assert [(voice, fmt) for _text, voice, fmt in asked] == [
        (voice, fmt)
        for voice in (GERMAN_VOICE, ENGLISH_VOICE)
        for fmt in ResponseFormat
    ]
    assert set(record) == {
        f"speak_sse_{voice}_{fmt.value}"
        for voice in (GERMAN_VOICE, ENGLISH_VOICE)
        for fmt in ResponseFormat
    }
    assert "words=4 sounds=10" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("words", "sounds", "passes"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_the_proof_needs_health_to_give_words_and_sounds(
    *, words: bool, sounds: bool, passes: bool
) -> None:
    health = Health.model_validate(
        {
            "speaking": {
                "ready": True,
                "voices": {
                    GERMAN_VOICE: {
                        "model": "piper",
                        "name": "n",
                        "languages": ["de"],
                        "takes_instructions": False,
                        "gives_marks": {"words": words, "sounds": sounds},
                        "ready": True,
                        "sample_rate": 22_050,
                    }
                },
            },
            "hearing": {"model": "small", "ready": False},
            "card_memory_mb": 0,
        }
    )

    if passes:
        marks.require_gives_marks(health, [GERMAN_VOICE])
    else:
        with pytest.raises(SystemExit):
            marks.require_gives_marks(health, [GERMAN_VOICE])
