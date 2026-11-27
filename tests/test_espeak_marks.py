"""The words of a Piper sentence, from espeak-ng's word events.

WIRE-152, WIRE-153, WIRE-154, WIRE-155, WIRE-156.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from speech_client.wire import SentenceMarks, Sound, Word
from speech_service.espeak_marks import (
    EspeakEvent,
    EventKind,
    Sentence,
    SentencePass,
    WordMarker,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

SAMPLES_PER_TOKEN = 256
SAMPLE_RATE = 1000


def word(position: int) -> EspeakEvent:
    return EspeakEvent(EventKind.WORD, position)


def sound(ipa: str) -> EspeakEvent:
    return EspeakEvent(EventKind.PHONEME, 0, ipa)


def spoken(tokens: Sequence[str]) -> Sentence:
    """A sentence of `tokens`, each token and its start and end the same samples."""
    token_samples = (SAMPLES_PER_TOKEN,) * (len(tokens) + 2)
    return Sentence(tuple(tokens), token_samples, sum(token_samples))


class ScriptedPasses:
    """The passes espeak would give, in order, and the starts each was asked for."""

    def __init__(self, *passes: SentencePass | None) -> None:
        """Script `passes`."""
        self._passes = list(passes)
        self.asked: list[int] = []

    def __call__(self, start: int) -> SentencePass | None:
        """The next scripted pass."""
        self.asked.append(start)
        return self._passes.pop(0)


def marked(
    text: str,
    sentences: Sequence[Sentence],
    passes: Sequence[SentencePass | None],
) -> list[SentenceMarks]:
    marker = WordMarker(text, ScriptedPasses(*passes), SAMPLE_RATE)
    return [marker.after(sentence) for sentence in sentences]


def words_of(marks: Sequence[SentenceMarks]) -> list[tuple[Word, ...] | None]:
    return [mark.words for mark in marks]


JA = spoken(["j", "\u02c8", "a", "."])
JA_EVENTS = (sound("j"), sound("\u02c8a"), sound(""))


def ja_pass(start: int, next_start: int | None) -> SentencePass:
    return SentencePass((word(start), *JA_EVENTS), next_start)


def test_the_words_of_each_sentence_are_input_spans_on_one_clock_across_sentences() -> (
    None
):
    """WIRE-152, WIRE-153: spans are absolute and times continue across sentences."""
    nein = spoken(["n", "\u02c8", "a", "i", "n", "."])
    passes = [
        SentencePass((word(1), sound("j"), sound("\u02c8a"), sound("")), 5),
        SentencePass(
            (word(5), sound("n"), sound("\u02c8ai"), sound("n"), sound("")), None
        ),
    ]

    marks = marked("Ja. Nein.", [JA, nein], passes)

    assert marks == [
        SentenceMarks(
            words=(Word(text="Ja", start_char=0, end_char=2, start=0.256, end=1.024),),
            sounds=(heard("j", 0.256, 0.512), heard("a", 0.512, 1.024)),
        ),
        SentenceMarks(
            words=(
                Word(text="Nein", start_char=4, end_char=8, start=1.792, end=3.072),
            ),
            sounds=(
                heard("n", 1.792, 2.048),
                heard("a", 2.048, 2.56),
                heard("i", 2.56, 2.816),
                heard("n", 2.816, 3.072),
            ),
        ),
    ]


def test_each_of_three_one_word_sentences_has_its_own_absolute_span() -> None:
    marks = marked(
        "Ja. Ja. Ja.",
        [JA, JA, JA],
        [ja_pass(1, 5), ja_pass(5, 9), ja_pass(9, None)],
    )

    spans = [(w.start_char, w.end_char) for m in marks for w in m.words or ()]
    assert spans == [(0, 2), (4, 6), (8, 10)]


def test_each_pass_starts_where_the_previous_sentence_ended() -> None:
    passes = ScriptedPasses(ja_pass(1, 5), ja_pass(5, 9), ja_pass(9, None))
    marker = WordMarker("Ja. Ja. Ja.", passes, SAMPLE_RATE)

    for _ in range(3):
        marker.after(JA)

    assert passes.asked == [1, 5, 9]


def test_a_word_starts_with_its_first_sound_not_with_the_stress_before_it() -> None:
    """WIRE-153: a stress mark's samples fall into the gap before the word."""
    stressed = spoken(["\u02c8", "a", "."])

    [mark] = marked(
        "Ah.", [stressed], [SentencePass((word(1), sound("\u02c8a"), sound("")), None)]
    )

    [only] = mark.words or ()
    assert only.start == (2 * SAMPLES_PER_TOKEN) / SAMPLE_RATE
    assert only.end == (3 * SAMPLES_PER_TOKEN) / SAMPLE_RATE


def test_a_word_ends_with_its_last_sound_not_with_the_pause_or_the_terminator() -> None:
    [mark] = marked("Ja.", [JA], [ja_pass(1, None)])

    [only] = mark.words or ()
    assert only.end == (4 * SAMPLES_PER_TOKEN) / SAMPLE_RATE


def test_times_are_truncated_to_milliseconds() -> None:
    marker = WordMarker("Ja.", ScriptedPasses(ja_pass(1, None)), 22_050)

    [only] = marker.after(JA).words or ()

    expected_start, expected_end = 0.011, 0.046  # 11.6 ms and 46.4 ms, cut off
    assert only.start == expected_start
    assert only.end == expected_end


def test_an_end_is_never_rounded_up_past_the_audio() -> None:
    """WIRE-153: 765 samples at 22050 Hz are 34.7 ms; the end is 34 ms, not 35."""
    marker = WordMarker("Ja.", ScriptedPasses(ja_pass(1, None)), 22_050)
    sentence = Sentence(("j", "a", "."), (SAMPLES_PER_TOKEN,) * 5, 765)

    [only] = marker.after(sentence).words or ()

    assert only.end <= sentence.audio_samples / 22_050


def test_a_word_never_ends_after_the_audio_of_its_sentence() -> None:
    short = Sentence(("j", "a", "."), (SAMPLES_PER_TOKEN,) * 5, 600)

    [mark] = marked("Ja.", [short], [ja_pass(1, None)])

    [only] = mark.words or ()
    assert only.end == short.audio_samples / SAMPLE_RATE


def test_a_word_that_merges_with_the_next_covers_both_in_its_exact_span() -> None:
    """WIRE-154: `in the` is one entry whose text is the exact input span."""
    merged = spoken(["i", "n", "ð", "ə", "."])
    events = (word(1), sound("i"), sound("n"), sound("ð"), sound("ə"), sound(""))

    [mark] = marked("in the.", [merged], [SentencePass(events, None)])

    [only] = mark.words or ()
    assert (only.text, only.start_char, only.end_char) == ("in the", 0, 6)


def test_a_word_excludes_the_whitespace_and_the_punctuation_after_it() -> None:
    sentence = spoken(["a", ",", " ", "b", "."])
    events = (word(1), sound("a"), sound(""), word(6), sound("b"), sound(""))

    [mark] = marked("Ah,  Be.", [sentence], [SentencePass(events, None)])

    assert [(w.text, w.end_char) for w in mark.words or ()] == [("Ah", 2), ("Be", 7)]


def test_a_word_event_on_whitespace_joins_the_word_before_it() -> None:
    """WIRE-154: an emoji spoken as two names is one entry, without its space."""
    sentence = spoken(["g", "r", " ", "g", "t", " ", "b", "."])
    events = (
        word(1),
        sound("gr"),
        word(2),
        sound("gt"),
        word(3),
        sound("b"),
        sound(""),
    )

    [mark] = marked("😀 b.", [sentence], [SentencePass(events, None)])

    assert [(w.text, w.start_char, w.end_char) for w in mark.words or ()] == [
        ("😀", 0, 1),
        ("b", 2, 3),
    ]


def test_several_events_inside_one_token_make_one_word() -> None:
    """A number spoken in several parts is one word."""
    sentence = spoken(["e", "i", "n", "s", "."])
    events = (word(1), sound("ei"), word(2), sound("n"), word(3), sound("s"))

    [mark] = marked("1990.", [sentence], [SentencePass(events, None)])

    [only] = mark.words or ()
    assert (only.text, only.start_char, only.end_char) == ("1990", 0, 4)


def test_language_tags_between_the_sounds_are_not_sounds() -> None:
    sentence = spoken(["b", "i", "z", "."])
    events = (word(1), sound("(en)"), sound("bi"), sound("z"), sound("(de)"))

    [mark] = marked("Biz.", [sentence], [SentencePass(events, None)])

    assert [w.text for w in mark.words or ()] == ["Biz"]


def test_a_stale_pause_before_the_first_word_is_dropped() -> None:
    events = (sound(""), word(1), *JA_EVENTS)

    [mark] = marked("Ja.", [JA], [SentencePass(events, None)])

    assert [w.text for w in mark.words or ()] == ["Ja"]


def test_a_composed_input_is_spanned_by_code_points() -> None:
    """WIRE-152: spans count code points of the input as received, never normalized."""
    decomposed = "Schön."
    sentence = spoken(["ʃ", "ø", "n", "."])
    events = (word(1), sound("ʃøn"), sound(""))

    [mark] = marked(decomposed, [sentence], [SentencePass(events, None)])

    [only] = mark.words or ()
    assert decomposed[only.start_char : only.end_char] == "Schön"


def test_a_word_split_across_a_merged_vowel_cluster_is_not_proven() -> None:
    merged = Sentence(("ai", "."), (SAMPLES_PER_TOKEN,) * 4, 4 * SAMPLES_PER_TOKEN)
    events = (word(1), sound("a"), word(4), sound("i"))

    [mark] = marked("Ah Ih.", [merged], [SentencePass(events, None)])

    assert mark.words is None


def test_a_merged_vowel_cluster_inside_one_word_is_proven() -> None:
    merged = Sentence(("ai", "."), (SAMPLES_PER_TOKEN,) * 4, 4 * SAMPLES_PER_TOKEN)

    [mark] = marked("Ei.", [merged], [SentencePass((word(1), sound("ai")), None)])

    [only] = mark.words or ()
    assert (only.start, only.end) == (0.256, 0.512)


def test_the_marks_of_input_with_a_raw_phoneme_block_are_null_for_every_sentence() -> (
    None
):
    passes = ScriptedPasses()
    marker = WordMarker("Ja [[ ja ]]. Ja.", passes, SAMPLE_RATE)

    marks = [marker.after(JA), marker.after(JA)]

    assert words_of(marks) == [None, None]
    assert passes.asked == []


def _differing_phonemes() -> list[SentencePass | None]:
    return [
        ja_pass(1, 5),
        SentencePass((word(5), sound("x"), sound("\u02c8a")), 9),
        ja_pass(9, None),
    ]


BAD_SECOND_PASSES = {
    "the callback failed": None,
    "positions relative to the sentence": SentencePass((word(1), *JA_EVENTS), 9),
    "a word after the sentence's end": SentencePass((word(5), *JA_EVENTS, word(9)), 9),
    "a falling position": SentencePass(
        (word(6), sound("j"), word(5), sound("\u02c8a")), 9
    ),
    "a sound before the first word": SentencePass(
        (sound("j"), word(5), sound("\u02c8a")), 9
    ),
    "a word opening on whitespace": SentencePass(
        (word(8), sound("j"), sound("\u02c8a")), 9
    ),
    "other sounds than Piper's": SentencePass(
        (word(5), sound("x"), sound("\u02c8a")), 9
    ),
    "a word without sounds": SentencePass((word(5), word(6)), 9),
    "no word": SentencePass((), 9),
    "a missing sentence event, the pass runs on": SentencePass(
        (word(5), *JA_EVENTS, word(9), *JA_EVENTS), None
    ),
}


@pytest.mark.parametrize(
    "second", list(BAD_SECOND_PASSES.values()), ids=list(BAD_SECOND_PASSES)
)
def test_a_sentence_that_fails_a_check_and_every_later_one_are_null(
    second: SentencePass | None,
) -> None:
    """WIRE-155: a failed check nulls its sentence and every later one."""
    passes = [ja_pass(1, 5), second, ja_pass(9, None)]

    marks = marked("Ja. Ja. Ja.", [JA, JA, JA], passes)

    first, *rest = words_of(marks)
    assert first is not None
    assert rest == [None, None]


def test_a_sentence_after_the_pass_that_reached_the_end_of_the_text_is_null() -> None:
    marks = marked("Ja.", [JA, JA], [ja_pass(1, None)])

    assert [m.words is not None for m in marks] == [True, False]


def test_a_later_sentence_is_not_asked_after_a_failure() -> None:
    passes = ScriptedPasses(ja_pass(1, 5), None)
    marker = WordMarker("Ja. Ja. Ja.", passes, SAMPLE_RATE)

    for _ in range(3):
        marker.after(JA)

    assert passes.asked == [1, 5]


def test_a_sentence_without_alignments_is_null_with_every_later_one() -> None:
    unaligned = replace(JA, token_samples=None)
    passes = [ja_pass(1, 5), ja_pass(5, 9), ja_pass(9, None)]

    marks = marked("Ja. Ja. Ja.", [JA, unaligned, JA], passes[:2])

    assert [m.words is not None for m in marks] == [True, False, False]


def test_a_sentence_whose_alignments_do_not_cover_its_phonemes_is_null() -> None:
    short = replace(JA, token_samples=JA.token_samples[:-1])  # type: ignore[index]

    [mark] = marked("Ja.", [short], [ja_pass(1, None)])

    assert mark.words is None


def test_a_sentence_of_the_marker_is_one_marks_event_even_when_it_is_null() -> None:
    marks = marked("Ja. Ja.", [JA, JA], [None])

    assert len(marks) == len([JA, JA])
    assert all(mark.sounds is None for mark in marks)


def test_a_unit_that_covers_no_text_joins_the_word_before_it() -> None:
    """WIRE-154: a word event on punctuation adds its sounds to the word before."""
    sentence = spoken(["j", "a", "x", "."])
    events = (word(1), sound("j"), sound("a"), word(4), sound("x"))

    [mark] = marked("Ja -.", [sentence], [SentencePass(events, None)])

    [only] = mark.words or ()
    assert (only.text, only.end) == ("Ja", 4 * SAMPLES_PER_TOKEN / SAMPLE_RATE)


@pytest.mark.parametrize(
    ("text", "events"),
    [
        ("- Ja.", (word(1), word(3), sound("j"), sound("a"))),
        ("Ja Ja.", (word(1), sound("j"), sound("a"), word(4))),
    ],
    ids=["no text for the first unit", "no sounds for a unit"],
)
def test_a_word_without_text_or_without_sounds_is_not_proven(
    text: str, events: tuple[EspeakEvent, ...]
) -> None:
    sentence = spoken(["j", "a", "."])

    [mark] = marked(text, [sentence], [SentencePass(events, None)])

    assert mark.words is None


TIE_BAR = "\u0361"
STRESS = "\u02c8"
COMBINING_TILDE = "\u0303"


def one_word_per_name(
    *words: Sequence[str], gap: str = " "
) -> tuple[str, Sentence, list[SentencePass]]:
    """A text of one-letter words, the sentence Piper spoke for it and its pass.

    Each word's tokens are the sounds espeak names for it (stress marks excepted);
    `gap` is the token Piper puts between two words.
    """
    text = " ".join("w" * (index + 1) for index in range(len(words))) + "."
    tokens: list[str] = []
    events: list[EspeakEvent] = []
    position = 1
    for index, tokens_of_word in enumerate(words):
        if index:
            tokens.append(gap)
        tokens.extend(tokens_of_word)
        events.append(word(position))
        events.append(sound("".join(tokens_of_word).replace(STRESS, "")))
        position += index + 2
    tokens.append(".")
    return text, spoken(tokens), [SentencePass(tuple(events), None)]


def sounds_of(*words: Sequence[str], gap: str = " ") -> tuple[Sound, ...] | None:
    text, sentence, passes = one_word_per_name(*words, gap=gap)
    [mark] = marked(text, [sentence], passes)
    return mark.sounds


def heard(ipa: str, begin: float, end: float) -> Sound:
    return Sound(ipa=ipa, start=begin, end=end)


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["a", "b"], [heard("a", 0.256, 0.512), heard("b", 0.512, 0.768)]),
        (
            ["t", TIE_BAR, "\u0283", "a"],
            [heard("t\u0361\u0283", 0.256, 1.024), heard("a", 1.024, 1.28)],
        ),
        (
            ["t", TIE_BAR, "\u02d0", "\u0283"],
            [heard("t\u0361\u02d0\u0283", 0.256, 1.28)],
        ),
        (
            ["a", "\u02d0", "b"],
            [heard("a\u02d0", 0.256, 0.768), heard("b", 0.768, 1.024)],
        ),
        (
            ["a", "\u02d1", "b"],
            [heard("a\u02d1", 0.256, 0.768), heard("b", 0.768, 1.024)],
        ),
        (
            ["a", COMBINING_TILDE, "b"],
            [heard("\u00e3", 0.256, 0.768), heard("b", 0.768, 1.024)],
        ),
        (["b", STRESS, "a"], [heard("b", 0.256, 0.512), heard("a", 0.512, 1.024)]),
        (
            ["a", "\u02cc", "b", "\u02d0"],
            [heard("a", 0.256, 0.512), heard("b\u02d0", 0.512, 1.28)],
        ),
        (["a", STRESS], [heard("a", 0.256, 0.512)]),
        (
            ["b", STRESS, "\u02cc", "a"],
            [heard("b", 0.256, 0.512), heard("a", 0.512, 1.28)],
        ),
        (
            ["t", TIE_BAR, STRESS, "a"],
            [heard("t\u0361", 0.256, 0.768), heard("a", 0.768, 1.28)],
        ),
        (
            ["a\u026a", "s"],
            [heard("a\u026a", 0.256, 0.512), heard("s", 0.512, 0.768)],
        ),
    ],
    ids=[
        "two bases",
        "tie bar joins the next base",
        "tie bar joins the next base past a length mark",
        "length mark",
        "half-length mark",
        "a combining mark is composed",
        "stress opens the following sound",
        "secondary stress opens a sound with its length mark",
        "stress with no sound after it is dropped",
        "two stress marks open the next sound at the first",
        "a stress mark after a tie bar opens the next sound",
        "a phoneme id of two bases is one sound with its own samples",
    ],
)
def test_a_sound_is_a_base_with_its_marks_and_a_stress_mark_opens_the_next(
    tokens: list[str], expected: list[Sound]
) -> None:
    """WIRE-159: the grouping, the NFC form and the stress rule."""
    assert sounds_of(tokens) == tuple(expected)


def test_a_sound_lasts_the_sum_of_the_samples_of_its_tokens() -> None:
    sentence = Sentence(
        ("t", TIE_BAR, "\u0283", "."), (100, 200, 300, 400, 50, 600), 1650
    )
    passes = [SentencePass((word(1), sound("t\u0361\u0283")), None)]

    [mark] = marked("Ja.", [sentence], passes)

    assert mark.sounds == (heard("t\u0361\u0283", 0.1, 1.0),)


def test_spaces_and_punctuation_separate_sounds_and_open_none() -> None:
    sentence_text, sentence, passes = one_word_per_name(["a"], ["b"], gap=",")

    [mark] = marked(sentence_text, [sentence], passes)

    assert mark.sounds == (heard("a", 0.256, 0.512), heard("b", 0.768, 1.024))


def test_stress_opens_the_sound_after_a_gap_and_precedes_the_word_it_belongs_to() -> (
    None
):
    """WIRE-159: the sound starts with its stress, the word with its sound."""
    text, sentence, passes = one_word_per_name(["a"], [STRESS, "b"])

    [mark] = marked(text, [sentence], passes)

    assert mark.sounds == (heard("a", 0.256, 0.512), heard("b", 0.768, 1.28))
    _, second = mark.words or ()
    assert second.start > mark.sounds[1].start  # type: ignore[index]


def test_the_sounds_of_a_word_lie_inside_it_but_for_the_stress_before_the_first() -> (
    None
):
    text, sentence, passes = one_word_per_name(["a", "b"], [STRESS, "c", "d"])

    [mark] = marked(text, [sentence], passes)

    first, second = mark.words or ()
    assert [(first.start, first.end), (second.start, second.end)] == [
        (0.256, 0.768),
        (1.28, 1.792),
    ]
    assert mark.sounds == (
        heard("a", 0.256, 0.512),
        heard("b", 0.512, 0.768),
        heard("c", 1.024, 1.536),
        heard("d", 1.536, 1.792),
    )


def test_the_sounds_of_each_sentence_continue_the_clock_of_the_one_before() -> None:
    marks = marked(
        "Ja. Ja.",
        [JA, JA],
        [ja_pass(1, 5), ja_pass(5, None)],
    )

    assert [mark.sounds for mark in marks] == [
        (heard("j", 0.256, 0.512), heard("a", 0.512, 1.024)),
        (heard("j", 1.792, 2.048), heard("a", 2.048, 2.56)),
    ]


def test_a_sentence_that_holds_no_sound_has_an_empty_list() -> None:
    [mark] = marked("...", [spoken(["."])], [SentencePass((), None)])

    assert (mark.words, mark.sounds) == ((), ())


def test_sounds_are_null_where_the_words_of_the_sentence_are_unproven() -> None:
    marks = marked("Ja. Ja. Ja.", [JA, JA, JA], [ja_pass(1, 5), None])

    assert [mark.sounds is not None for mark in marks] == [True, False, False]


def test_a_sound_ends_no_later_than_the_audio() -> None:
    sentence = Sentence(("a",), (10, 500, 0), 100)

    [mark] = marked("a", [sentence], [SentencePass((word(1), sound("a")), None)])

    assert mark.sounds == (heard("a", 0.01, 0.1),)
