"""The marks proof of speaking: a `sse` answer on a Piper voice, judged by the wire.

`speak_sse` reads the answer over raw HTTP, the way any caller that is no
`speech_client` does; `judge_sse_answer` holds it to the sentences docs/wire.md
rules; `prove_marks` runs both over every voice and format. A failure ends the
proof with a message that names the sentence broken and holds no spoken text.
"""

# ruff: noqa: T201 - the proof prints its stage lines as the command line output

from __future__ import annotations

import base64
import http.client
import json
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from pydantic import ValidationError

from proof_identity import mint_pair
from shipped_catalogue import CATALOGUE, first_language
from speech_client.wire import (
    SPEECH_PATH,
    AudioEventType,
    Health,
    ResponseFormat,
    SentenceMarks,
    Sound,
    SpeechRequest,
    StreamFormat,
    Word,
    speech_request_headers,
)
from speech_service.pcm import BYTES_PER_SAMPLE, HEARING_SAMPLE_RATE, pcm_from_wav
from speech_service.recording import decode_recording

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from proof_identity import Minting
    from speech_client.wire import Language, VoiceId

type SpeakSse = Callable[[str, VoiceId, ResponseFormat], bytes]

SPEAK_TIMEOUT_SECONDS: Final = 120
MAX_AUDIO_SECONDS: Final = 60.0
CONTAINER_SLACK_SECONDS: Final = 0.05
"""What a container's frames may add to the audio a mark's time stays within."""
_STRESS_MARKS: Final = "\u02c8\u02cc"
_SOUND_FIELDS: Final = {"ipa", "start", "end"}
_EVENT_SEPARATOR: Final = "\n\n"
_DATA_PREFIX: Final = "data: "


@dataclass(frozen=True)
class MarksFigures:
    """What a judged answer held: its events, marked words and sounds, and audio."""

    deltas: int
    marks: int
    words: int
    sounds: int
    audio_seconds: float


def speak_sse(
    base_url: str,
    text: str,
    voice: VoiceId,
    response_format: ResponseFormat,
    minting: Minting,
) -> bytes:
    """Speak `text` with `stream_format` `sse` and read the whole body.

    A body the service aborted raises `IncompleteRead`, and the proof fails: a
    stream without `done` is a failed one.
    """
    pair = mint_pair(minting)
    request = SpeechRequest(
        model=CATALOGUE[voice].engine.wire_model,
        input=text,
        voice=voice,
        response_format=response_format,
        language=first_language(voice),
        stream_format=StreamFormat.SSE,
    )
    address = urlsplit(base_url)
    connection = http.client.HTTPConnection(
        address.hostname, address.port, timeout=SPEAK_TIMEOUT_SECONDS
    )
    try:
        connection.request(
            "POST",
            SPEECH_PATH,
            request.model_dump_json().encode(),
            speech_request_headers(
                workload_token=pair.workload, person_token=pair.person
            ),
        )
        response = connection.getresponse()
        if response.status != http.client.OK:
            message = f"sse speak HTTP status: {response.status}"
            raise SystemExit(message)
        if response.getheader("Content-Type", "").split(";")[0] != "text/event-stream":
            message = "sse speak is not a text/event-stream"
            raise SystemExit(message)
        return response.read()
    except http.client.IncompleteRead:
        message = "sse speak body ended before its done event"
        raise SystemExit(message) from None
    finally:
        connection.close()


def read_events(body: bytes) -> list[dict[str, object]]:
    """The events of `body`: each a `data:` line with a JSON object and a blank line."""
    try:
        frames = body.decode("ascii").split(_EVENT_SEPARATOR)
    except UnicodeDecodeError:
        message = "sse body is not ASCII"
        raise SystemExit(message) from None
    if frames.pop() != "":
        message = "sse body ends inside an event"
        raise SystemExit(message)
    events: list[dict[str, object]] = []
    for frame in frames:
        try:
            event = json.loads(frame.removeprefix(_DATA_PREFIX))
        except json.JSONDecodeError:
            event = None
        if not frame.startswith(_DATA_PREFIX) or not isinstance(event, dict):
            message = "sse body holds a frame that is no `data:` JSON object"
            raise SystemExit(message)
        events.append(event)
    return events


def judge_sse_answer(
    body: bytes, text: str, response_format: ResponseFormat, *, sentences: int
) -> MarksFigures:
    """Hold the answer to docs/wire.md: its order, its audio and its marks.

    Each of the `sentences` of `text` has one marks event after audio, whose words
    are spans of `text` with strictly increasing seconds within the audio, and whose
    sounds are NFC text without stress on the same clock, each ending inside a word.
    """
    events = read_events(body)
    _require_order(events, sentences)
    audio = b"".join(
        base64.b64decode(str(event["audio"]))
        for event in events
        if event["type"] == AudioEventType.DELTA
    )
    seconds = _audio_seconds(audio, response_format)
    words, sounds = _require_marks(
        [event for event in events if event["type"] == AudioEventType.MARKS],
        text,
        seconds,
    )
    return MarksFigures(
        deltas=len(events) - sentences - 1,
        marks=sentences,
        words=words,
        sounds=sounds,
        audio_seconds=seconds,
    )


def _require_order(events: list[dict[str, object]], sentences: int) -> None:
    kinds = [event.get("type") for event in events]
    allowed = {item.value for item in AudioEventType}
    if not set(kinds) <= allowed or kinds.count(AudioEventType.DONE) != 1:
        message = "sse answer holds an unknown event or not one done"
        raise SystemExit(message)
    if kinds[-1] != AudioEventType.DONE or events[-1] != {"type": kinds[-1]}:
        message = "sse answer does not end with a bare done"
        raise SystemExit(message)
    if kinds.count(AudioEventType.MARKS) != sentences:
        message = f"sse answer holds {kinds.count(AudioEventType.MARKS)} marks events"
        raise SystemExit(message)
    heard = False
    for kind in kinds:
        if kind == AudioEventType.MARKS and not heard:
            message = "sse marks come before the audio of their sentence"
            raise SystemExit(message)
        heard = kind == AudioEventType.DELTA


def _audio_seconds(audio: bytes, response_format: ResponseFormat) -> float:
    if not response_format.begins_like_container(audio):
        message = f"sse deltas are not {response_format.value}"
        raise SystemExit(message)
    if response_format is ResponseFormat.WAV:
        rate, pcm = pcm_from_wav(audio)
    else:
        rate, pcm = HEARING_SAMPLE_RATE, decode_recording(audio, MAX_AUDIO_SECONDS)
    return len(pcm) / BYTES_PER_SAMPLE / rate


def _require_marks(
    marks: list[dict[str, object]], text: str, audio_seconds: float
) -> tuple[int, int]:
    covered: set[int] = set()
    previous_word_start = -1.0
    previous_sound_start = -1.0
    words = 0
    sounds = 0
    for event in marks:
        parsed = _marks_of(event)
        if parsed.words is None or parsed.sounds is None:
            message = (
                "sse marks hold no words or sounds for a sentence the voice proves"
            )
            raise SystemExit(message)
        if parsed.words and not parsed.sounds:
            message = "sse marks hold words for a sentence and no sound in it"
            raise SystemExit(message)
        for word in parsed.words:
            _require_word(word, text, audio_seconds, previous_word_start)
            previous_word_start = word.start
            covered.update(range(word.start_char, word.end_char))
            words += 1
        for sound in parsed.sounds:
            _require_sound(sound, parsed.words, audio_seconds, previous_sound_start)
            previous_sound_start = sound.start
            sounds += 1
    if not all(index in covered for index, char in enumerate(text) if char.isalnum()):
        message = "sse marks leave a word of the text out"
        raise SystemExit(message)
    return words, sounds


def _marks_of(event: dict[str, object]) -> SentenceMarks:
    fields = {key: value for key, value in event.items() if key != "type"}
    if set(fields) != {"words", "sounds"}:
        message = "sse marks event does not hold exactly words and sounds"
        raise SystemExit(message)
    try:
        marks = SentenceMarks.model_validate(fields)
    except ValidationError:
        message = "sse marks event is not words and sounds"
        raise SystemExit(message) from None
    if any(set(entry) != _SOUND_FIELDS for entry in fields["sounds"] or ()):  # type: ignore[attr-defined]
        message = "sse sound does not hold exactly ipa, start and end"
        raise SystemExit(message)
    return marks


def _require_word(word: Word, text: str, audio_seconds: float, before: float) -> None:
    if text[word.start_char : word.end_char] != word.text or not word.text:
        message = "sse word is not the span of the input it names"
        raise SystemExit(message)
    if not before < word.start < word.end <= audio_seconds + CONTAINER_SLACK_SECONDS:
        message = "sse word times do not increase within the audio"
        raise SystemExit(message)


def prove_marks(
    speak: SpeakSse,
    voices: Iterable[VoiceId],
    texts: Mapping[Language, str],
    *,
    sentences: int,
) -> dict[str, dict[str, float]]:
    """Speak each voice's text in each format as `sse` and judge every answer."""
    record: dict[str, dict[str, float]] = {}
    for voice in voices:
        text = texts[first_language(voice)]
        for response_format in ResponseFormat:
            figures = judge_sse_answer(
                speak(text, voice, response_format),
                text,
                response_format,
                sentences=sentences,
            )
            label = f"speak_sse_{voice}_{response_format.value}"
            print(
                label,
                f"deltas={figures.deltas}",
                f"marks={figures.marks}",
                f"words={figures.words}",
                f"sounds={figures.sounds}",
                f"audio_s={figures.audio_seconds:.3f}",
            )
            record[label] = {
                "words": figures.words,
                "sounds": figures.sounds,
                "audio_s": round(figures.audio_seconds, 3),
            }
    return record


def require_gives_marks(health: Health, voices: Iterable[VoiceId]) -> None:
    """Every voice in `voices` reports words and sounds as given."""
    for voice in voices:
        gives = health.speaking.voices[voice].gives_marks
        if not (gives.words and gives.sounds):
            message = f"health gives_marks of {voice} is {gives!r}"
            raise SystemExit(message)


def _require_sound(
    sound: Sound, words: tuple[Word, ...], audio_seconds: float, before: float
) -> None:
    if (
        not sound.ipa
        or unicodedata.normalize("NFC", sound.ipa) != sound.ipa
        or any(mark in sound.ipa for mark in _STRESS_MARKS)
    ):
        message = "sse sound is not NFC text without a stress mark"
        raise SystemExit(message)
    if (
        not before
        <= sound.start
        <= sound.end
        <= audio_seconds + CONTAINER_SLACK_SECONDS
    ):
        message = "sse sound times do not increase within the audio"
        raise SystemExit(message)
    if not any(word.start <= sound.end <= word.end for word in words):
        message = "sse sound does not end inside a word of its sentence"
        raise SystemExit(message)
