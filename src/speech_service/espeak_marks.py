"""The words and sounds of a Piper sentence, read from espeak-ng's own word events.

Piper phonemizes with the espeak-ng it links into `piper/espeakbridge.so`. That
library also reports, while it speaks a text, which text position each word and each
phoneme belongs to. After a sentence's audio, one pass over that sentence ties the
phonemes Piper spoke to the words of the input. A sentence is mapped only when every
check holds; the first that fails leaves that sentence and every later one without
words, because the passes follow one another through the text. A sentence whose words
are proven has its sounds too: they are the same phonemes, grouped by their code points.
"""

from __future__ import annotations

import ctypes
import logging
import re
import threading
import unicodedata
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Final

from speech_client.wire import SentenceMarks, Sound, Word
from speech_service.failures import log_failure

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from piper import PiperVoice

logger = logging.getLogger(__name__)

# The symbols and the event layout are a build artefact of this exact bridge.
ESPEAK_VERSION: Final = "1.52.0.1"
_SELF_TEST_TEXT: Final = "One two. Three."
_SELF_TEST_VOICE: Final = "en"
# Per self-test pass: where it starts, the positions of its words, and where the
# next sentence starts.
_SELF_TEST_PASSES: Final = ((1, (1, 5), 10), (10, (10,), None))
_OUTPUTS_WITH_ALIGNMENT: Final = 2
# A text that espeak speaks to its end in an instant, to leave nothing half-spoken.
SETTLING_TEXT: Final = b"a"

_SYNCHRONOUS_OUTPUT: Final = 1
_CHARACTER_POSITIONS: Final = 1
_UTF8_TEXT: Final = 1
_LIST_TERMINATED: Final = 0
_CONTINUE: Final = 0
_ABORT: Final = 1

_RAW_PHONEMES: Final = "[["
_STRESS: Final = "\u02c8\u02cc"
_STRESS_MARKS: Final = str.maketrans("", "", _STRESS)
_TIE_BAR: Final = "\u0361"
# Marks that belong to the sound before them besides the combining ones.
_LENGTH_MARKS: Final = frozenset("\u02d0\u02d1")
# Piper's own removal of the language tags espeak puts around a foreign word.
_LANGUAGE_TAG: Final = re.compile(r"\([^)]+\)")
# Piper's tokens that are no sound: the space and the clause terminators.
_NOT_SOUNDS: Final = frozenset(" ,.:;?!")
_MILLISECONDS: Final = 1_000


class EventKind(IntEnum):
    """The espeak-ng event types a pass keeps, by their numbers."""

    WORD = 1
    SENTENCE = 2
    PHONEME = 7


@dataclass(frozen=True, slots=True)
class EspeakEvent:
    """A word or a phoneme at a 1-based code point position of the text."""

    kind: EventKind
    position: int
    name: str = ""


@dataclass(frozen=True, slots=True)
class SentencePass:
    """The word and phoneme events of one sentence, and where the next one starts.

    `next_start` is `None` when the sentence runs to the end of the text.
    """

    events: tuple[EspeakEvent, ...]
    next_start: int | None


@dataclass(frozen=True, slots=True)
class Sentence:
    """What Piper spoke for one sentence.

    `token_samples` holds the samples of the start token, each phoneme token and the
    end token in order, or is `None` where Piper's alignment failed.
    """

    phonemes: tuple[str, ...]
    token_samples: tuple[int, ...] | None
    audio_samples: int


class WordsOffReason(StrEnum):
    """Why the load check switched words off."""

    ALIGNMENT_OUTPUT = "alignment_output"
    LIBRARY = "library"
    LOCK = "lock"
    VERSION = "version"
    SELF_TEST = "self_test"


class _WordsOffError(Exception):
    def __init__(self, reason: WordsOffReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


class _EventId(ctypes.Union):
    _fields_ = (("number", ctypes.c_int), ("name", ctypes.c_char * 8))


class NativeEvent(ctypes.Structure):
    """One event of espeak-ng's callback, in the layout of the bridge's build."""

    _fields_ = (
        ("type", ctypes.c_int),
        ("unique_identifier", ctypes.c_uint),
        ("text_position", ctypes.c_int),
        ("length", ctypes.c_int),
        ("audio_position", ctypes.c_int),
        ("sample", ctypes.c_int),
        ("user_data", ctypes.c_void_p),
        ("id", _EventId),
    )


_Callback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_short),
    ctypes.c_int,
    ctypes.POINTER(NativeEvent),
)


class _Collector:
    """The synthesis callback of one pass: it keeps the events of one sentence.

    espeak announces a sentence before its first word; a pass that starts inside the
    text may or may not announce its own. The first announcement after a word is the
    next sentence's, where the pass ends.
    """

    def __init__(self) -> None:
        self.events: list[EspeakEvent] = []
        self.next_start: int | None = None
        self.failed = False
        self._saw_word = False

    def __call__(
        self, _wave: object, _samples: int, events: Sequence[NativeEvent]
    ) -> int:
        # An exception cannot unwind through the C caller; the pass reads `failed`.
        try:
            return self._collect(events)
        except Exception:  # noqa: BLE001
            self.failed = True
            return _ABORT

    def _collect(self, events: Sequence[NativeEvent]) -> int:
        index = 0
        while events[index].type != _LIST_TERMINATED:
            event = events[index]
            index += 1
            match event.type:
                case EventKind.SENTENCE if self._saw_word:
                    self.next_start = event.text_position
                    return _ABORT
                case EventKind.WORD:
                    self._saw_word = True
                    self.events.append(EspeakEvent(EventKind.WORD, event.text_position))
                case EventKind.PHONEME:
                    name = event.id.name.decode("utf-8")
                    self.events.append(
                        EspeakEvent(EventKind.PHONEME, event.text_position, name)
                    )
        return _CONTINUE


def _drop_events(*_arguments: object) -> int:
    return _CONTINUE


_DROP_EVENTS = _Callback(_drop_events)


class EspeakEvents:
    """Passes of espeak-ng over a text, run through the bridge Piper links in."""

    def __init__(self, library: ctypes.CDLL, lock: threading.Lock) -> None:
        """Hold the bridge's library and Piper's lock over espeak's state."""
        self._library = library
        self._lock = lock

    def read(self, text: str, espeak_voice: str, start: int) -> SentencePass | None:
        """The events of the sentence of `text` that begins at position `start`.

        `None` where the pass failed. Runs under the lock Piper holds while it
        phonemizes, because espeak keeps one voice and one callback for the process.
        """
        collector = _Collector()
        callback = _Callback(collector)
        data = text.encode("utf-8")
        try:
            with self._lock:
                from piper import espeakbridge

                espeakbridge.set_voice(espeak_voice)
                self._settle()
                self._library.espeak_SetSynthCallback(callback)
                try:
                    status = self._library.espeak_Synth(
                        data,
                        len(data) + 1,
                        start - 1,
                        _CHARACTER_POSITIONS,
                        0,
                        _UTF8_TEXT,
                        None,
                        None,
                    )
                finally:
                    self._settle()
                    self._library.espeak_SetSynthCallback(_Callback(0))
        except Exception as error:  # noqa: BLE001 - marks are optional; the audio goes on
            log_failure(logger, "marks.pass_failed", error, level=logging.WARNING)
            return None
        if status != 0 or collector.failed:
            return None
        return SentencePass(tuple(collector.events), collector.next_start)

    def _settle(self) -> None:
        """Speak a short text to the end into a callback that drops what it reports.

        espeak queues an event for each voice it sets and each text it phonemizes,
        and only speaking empties the queue; once enough pile up, the next pass
        loses its first events. A pass that ended early, at the next sentence,
        leaves the rest of its text half-spoken, which Piper's next phonemizing
        would take up as the start of its own text.
        """
        self._library.espeak_SetSynthCallback(_DROP_EVENTS)
        self._library.espeak_Synth(
            SETTLING_TEXT,
            len(SETTLING_TEXT) + 1,
            0,
            _CHARACTER_POSITIONS,
            0,
            _UTF8_TEXT,
            None,
            None,
        )


def load_espeak_events(
    voice: PiperVoice, open_library: Callable[[str], ctypes.CDLL] = ctypes.CDLL
) -> EspeakEvents | None:
    """The passes of `voice`'s espeak-ng, or `None` with an error line if words are off.

    `open_library` opens the bridge's shared library from its path.

    Words are off when the voice gives no alignment, or the bridge is not the one
    this module was written against, or a known text does not give its known
    positions. The audio of the voice is not affected.
    """
    try:
        return _checked(voice, open_library)
    except _WordsOffError as off:
        logger.exception("marks.words_off", extra={"reason": off.reason.value})
        return None


def _checked(
    voice: PiperVoice, open_library: Callable[[str], ctypes.CDLL]
) -> EspeakEvents:
    try:
        if len(voice.session.get_outputs()) != _OUTPUTS_WITH_ALIGNMENT:
            raise _WordsOffError(WordsOffReason.ALIGNMENT_OUTPUT)
        from piper import espeakbridge
        from piper import voice as piper_voice

        lock = piper_voice._ESPEAK_PHONEMIZER_LOCK  # noqa: SLF001
        if not isinstance(lock, type(threading.Lock())):
            raise _WordsOffError(WordsOffReason.LOCK)
        # Piper initializes espeak on its first phonemize; the passes need it.
        voice.phonemize(_SELF_TEST_TEXT)
        # ctypes.CDLL releases the GIL while espeak speaks: the event loop is not held.
        library = _configured_bridge(open_library(espeakbridge.__file__))
    except (ImportError, AttributeError, OSError) as error:
        raise _WordsOffError(WordsOffReason.LIBRARY) from error
    if library.espeak_Info(None) != ESPEAK_VERSION.encode("ascii"):
        raise _WordsOffError(WordsOffReason.VERSION)
    events = EspeakEvents(library, lock)
    if not _gives_the_known_positions(events):
        raise _WordsOffError(WordsOffReason.SELF_TEST)
    return events


def _configured_bridge(library: ctypes.CDLL) -> ctypes.CDLL:
    library.espeak_Info.restype = ctypes.c_char_p
    library.espeak_Info.argtypes = (ctypes.c_void_p,)
    library.espeak_ng_InitializeOutput.argtypes = (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
    )
    library.espeak_ng_SetPhonemeEvents.argtypes = (ctypes.c_int, ctypes.c_int)
    library.espeak_SetSynthCallback.argtypes = (_Callback,)
    library.espeak_Synth.argtypes = (
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    if library.espeak_ng_InitializeOutput(_SYNCHRONOUS_OUTPUT, 0, None) != 0:
        raise OSError
    # Phoneme events carry their IPA name.
    if library.espeak_ng_SetPhonemeEvents(1, 1) != 0:
        raise OSError
    return library


def _gives_the_known_positions(events: EspeakEvents) -> bool:
    for start, word_positions, next_start in _SELF_TEST_PASSES:
        result = events.read(_SELF_TEST_TEXT, _SELF_TEST_VOICE, start)
        if result is None or result.next_start != next_start:
            return False
        positions = tuple(
            event.position for event in result.events if event.kind is EventKind.WORD
        )
        if positions != word_positions:
            return False
    return True


class WordMarker:
    """The words and sounds of the sentences of one text, in Piper's speaking order."""

    def __init__(
        self,
        text: str,
        read_pass: Callable[[int], SentencePass | None],
        sample_rate: int,
    ) -> None:
        """Map `text`; `read_pass` gives the pass of the sentence at a position."""
        self._text = text
        self._read_pass = read_pass
        self._sample_rate = sample_rate
        self._next_start: int | None = 1
        self._proven = _RAW_PHONEMES not in text
        self._elapsed_samples = 0

    def after(self, sentence: Sentence) -> SentenceMarks:
        """The marks of `sentence`, which Piper has just spoken after the earlier."""
        words = self._words(sentence) if self._proven else None
        self._proven = words is not None
        sounds = self._sounds(sentence) if words is not None else None
        self._elapsed_samples += sentence.audio_samples
        return SentenceMarks(words=words, sounds=sounds)

    def _sounds(self, sentence: Sentence) -> tuple[Sound, ...]:
        """The sounds of a sentence whose words are proven, so its samples are whole."""
        return tuple(
            Sound(
                ipa=unicodedata.normalize("NFC", sound.ipa),
                start=self._seconds(min(sound.begin, sentence.audio_samples)),
                end=self._seconds(min(sound.end, sentence.audio_samples)),
            )
            for sound in _sounds_of(sentence)
        )

    def _words(self, sentence: Sentence) -> tuple[Word, ...] | None:
        pieces = _pieces(sentence)
        if pieces is None or self._next_start is None:
            return None
        start = self._next_start
        found = self._read_pass(start)
        if found is None:
            return None
        self._next_start = found.next_start
        end = found.next_start or len(self._text) + 1
        spans = _spans(self._text, _units(self._text, start, end, found.events), end)
        if spans is None:
            return None
        return self._timed(spans, pieces, sentence.audio_samples)

    def _timed(
        self, spans: Sequence[_Span], pieces: Sequence[_Piece], audio_samples: int
    ) -> tuple[Word, ...] | None:
        heard = "".join(piece.sounds for piece in pieces)
        said = "".join(span.sounds for span in spans)
        if heard != said:
            return None
        words: list[Word] = []
        index = 0
        for span in spans:
            if not span.sounds:
                return None
            first = index
            remaining = len(span.sounds)
            while remaining > 0:
                remaining -= len(pieces[index].sounds)
                index += 1
            begin = pieces[first].begin
            stop = min(pieces[index - 1].end, audio_samples)
            if remaining != 0 or begin >= stop:
                return None
            words.append(
                Word(
                    text=self._text[span.begin : span.stop],
                    start_char=span.begin,
                    end_char=span.stop,
                    start=self._seconds(begin),
                    end=self._seconds(stop),
                )
            )
        return tuple(words)

    def _seconds(self, sample: int) -> float:
        # Whole milliseconds, cut off: a rounded `end` could lie after the audio.
        milliseconds = (self._elapsed_samples + sample) * _MILLISECONDS
        return milliseconds // self._sample_rate / _MILLISECONDS


@dataclass(slots=True)
class _Unit:
    start: int
    sounds: str = ""


@dataclass(slots=True)
class _Span:
    begin: int
    stop: int
    sounds: str


@dataclass(frozen=True, slots=True)
class _Piece:
    sounds: str
    begin: int
    end: int


@dataclass(slots=True)
class _OpenSound:
    ipa: str
    begin: int
    end: int


def _normalized(sounds: str) -> str:
    return unicodedata.normalize("NFD", sounds.translate(_STRESS_MARKS))


def _tokens(sentence: Sentence) -> Iterator[tuple[str, int, int]] | None:
    """Piper's tokens with their samples, or `None` where the samples are not whole."""
    samples = sentence.token_samples
    if samples is None or len(samples) != len(sentence.phonemes) + 2:
        return None
    return _spans_of_tokens(sentence.phonemes, samples)


def _spans_of_tokens(
    phonemes: Sequence[str], samples: Sequence[int]
) -> Iterator[tuple[str, int, int]]:
    cursor = samples[0]
    for token, count in zip(phonemes, samples[1:-1], strict=True):
        yield token, cursor, cursor + count
        cursor += count


def _pieces(sentence: Sentence) -> list[_Piece] | None:
    """The sounds of Piper's tokens with the samples each one takes, stress apart.

    A stress mark's samples lie between two sounds, so they open no word.
    """
    tokens = _tokens(sentence)
    if tokens is None:
        return None
    pieces: list[_Piece] = []
    for token, begin, end in tokens:
        sounds = "" if token in _NOT_SOUNDS else _normalized(token)
        if sounds:
            pieces.append(_Piece(sounds, begin, end))
    return pieces


def _sounds_of(sentence: Sentence) -> list[_OpenSound]:
    """The sounds of a sentence whose tokens are whole.

    A sound is a base token with the combining marks and length marks after it; the
    tie bar joins the next base. Gaps end a sound. A stress mark's samples open the
    sound after it, so the mouth is ready before the vowel is heard.
    """
    sounds: list[_OpenSound] = []
    current: _OpenSound | None = None
    stress_begin: int | None = None
    joining = False
    for token, begin, end in _tokens(sentence) or ():
        if token in _NOT_SOUNDS:
            current, stress_begin, joining = None, None, False
        elif token in _STRESS:
            stress_begin = begin if stress_begin is None else stress_begin
            current = current if joining else None
        elif current is not None and (joining or _extends_a_sound(token)):
            current.ipa += token
            current.end = end
            stress_begin, joining = None, token == _TIE_BAR
        else:
            opening = begin if stress_begin is None else stress_begin
            current = _OpenSound(token, opening, end)
            sounds.append(current)
            stress_begin, joining = None, False
    return sounds


def _extends_a_sound(token: str) -> bool:
    return (
        token == _TIE_BAR
        or token in _LENGTH_MARKS
        or all(unicodedata.category(character) == "Mn" for character in token)
    )


def _units(
    text: str, start: int, end: int, events: Sequence[EspeakEvent]
) -> list[_Unit] | None:
    """The words of one pass, each with the sounds espeak spoke for it.

    A word event opens a unit, unless it lies inside the token of the unit before
    (a number spoken in parts) or on whitespace (a symbol spoken as two names);
    then it joins that unit.
    """
    units: list[_Unit] = []
    for event in events:
        if event.kind is EventKind.WORD:
            if not _took_word(text, range(start, end), units, event.position):
                return None
        elif sounds := _LANGUAGE_TAG.sub("", event.name):
            if not units:
                return None
            units[-1].sounds += _normalized(sounds)
    return units


def _took_word(text: str, positions: range, units: list[_Unit], position: int) -> bool:
    if position not in positions:
        return False
    on_whitespace = text[position - 1].isspace()
    if not units:
        if on_whitespace:
            return False
        units.append(_Unit(position))
        return True
    current = units[-1]
    if position < current.start:
        return False
    if not on_whitespace and position >= _token_end(text, current.start):
        units.append(_Unit(position))
    return True


def _token_end(text: str, start: int) -> int:
    """The position after the run of non-whitespace that begins at `start`."""
    index = start - 1
    while index < len(text) and not text[index].isspace():
        index += 1
    return index + 1


def _spans(text: str, units: list[_Unit] | None, end: int) -> list[_Span] | None:
    """The text each unit covers, up to the next unit, trailing punctuation cut off.

    A unit that covers no text joins the unit before it.
    """
    if units is None:
        return None
    spans: list[_Span] = []
    for index, unit in enumerate(units):
        following = units[index + 1].start if index + 1 < len(units) else end
        begin = unit.start - 1
        stop = _without_trailing_punctuation(text, begin, following - 1)
        if stop > begin:
            spans.append(_Span(begin, stop, unit.sounds))
        elif spans:
            spans[-1].sounds += unit.sounds
        else:
            return None
    return spans


def _without_trailing_punctuation(text: str, begin: int, stop: int) -> int:
    while stop > begin and (
        text[stop - 1].isspace() or unicodedata.category(text[stop - 1]).startswith("P")
    ):
        stop -= 1
    return stop
