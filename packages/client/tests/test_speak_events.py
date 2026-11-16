"""The marked speaking stream against a raw listener the real service never matches."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

import pytest

from packages.client.tests.support.scripted_service_support import SAMPLE_RATE, wav_of
from speech_client import speak as speak_module
from speech_client.speak import (
    Cut,
    Finished,
    Malformed,
    OverCap,
    SentenceMarks,
    Word,
    speak_stream,
)
from speech_client.wire import AudioEventType, StreamFormat, VoiceId, audio_event

if TYPE_CHECKING:
    from speech_client.speak import MarkedStreamItem
    from support.listener_support import RawListener

DEADLINE_SECONDS = 30.0
CAP_BYTES = 1024 * 1024
PCM = b"\x01\x00" * 8
WAV = wav_of(PCM)
WORD = Word(text="Hallo", start_char=0, end_char=5, start=0.0, end=0.1)


def _chunk(data: bytes) -> bytes:
    return b"%x\r\n%b\r\n" % (len(data), data)


def _answer(
    *events: bytes,
    media_type: str = "text/event-stream",
    ended: bool = True,
) -> bytes:
    head = (
        f"HTTP/1.1 200 OK\r\nContent-Type: {media_type}\r\n"
        "Transfer-Encoding: chunked\r\n\r\n"
    ).encode()
    return head + b"".join(map(_chunk, events)) + (b"0\r\n\r\n" if ended else b"")


def _delta(audio: bytes) -> bytes:
    return audio_event(AudioEventType.DELTA, audio=base64.b64encode(audio).decode())


def _marks(*words: Word) -> bytes:
    return audio_event(
        AudioEventType.MARKS,
        words=[word.model_dump(mode="json") for word in words],
        sounds=None,
    )


DONE = audio_event(AudioEventType.DONE)


def _stream(base_url: str, *, cap_bytes: int = CAP_BYTES) -> list[MarkedStreamItem]:
    return list(
        speak_stream(
            base_url,
            "Hallo",
            VoiceId.THORSTEN,
            stream_format=StreamFormat.SSE,
            workload_token="workload-token",  # noqa: S106 - a stand-in the raw listener never checks
            person_token="person-token",  # noqa: S106 - a stand-in the raw listener never checks
            deadline_seconds=DEADLINE_SECONDS,
            cap_bytes=cap_bytes,
        )
    )


def test_the_items_are_the_audio_and_the_marks_in_arrival_order_then_finished(
    raw_listener: RawListener,
) -> None:
    answer = _answer(_delta(WAV[:20]), _delta(WAV[20:]), _marks(WORD), DONE)

    with raw_listener(answer) as (base_url, _):
        items = _stream(base_url)

    assert items == [
        WAV[:20],
        WAV[20:],
        SentenceMarks(words=(WORD,)),
        Finished(len(PCM) // 2 / SAMPLE_RATE),
    ]


def test_a_marks_item_without_words_or_sounds_holds_none(
    raw_listener: RawListener,
) -> None:
    unproven = audio_event(AudioEventType.MARKS, words=None, sounds=None)

    with raw_listener(_answer(_delta(WAV), unproven, DONE)) as (base_url, _):
        items = _stream(base_url)

    assert SentenceMarks(words=None, sounds=None) in items


def test_a_field_the_reader_does_not_know_is_ignored(
    raw_listener: RawListener,
) -> None:
    """WIRE-158: a reader ignores a field it does not know."""
    extended = audio_event(AudioEventType.DONE, usage={"tokens": 3})

    with raw_listener(_answer(_delta(WAV), extended)) as (base_url, _):
        items = _stream(base_url)

    assert isinstance(items[-1], Finished)


def test_the_request_asks_for_sse_and_the_default_request_does_not_name_it(
    raw_listener: RawListener,
) -> None:
    with raw_listener(_answer(_delta(WAV), DONE)) as (base_url, requests):
        _stream(base_url)
    with raw_listener(_answer(_delta(WAV), DONE)) as (base_url, default_requests):
        list(
            speak_stream(
                base_url,
                "Hallo",
                VoiceId.THORSTEN,
                workload_token="workload-token",  # noqa: S106 - a stand-in the raw listener never checks
                person_token="person-token",  # noqa: S106 - a stand-in the raw listener never checks
                deadline_seconds=DEADLINE_SECONDS,
                cap_bytes=CAP_BYTES,
            )
        )

    asked = json.loads(requests[0].partition(b"\r\n\r\n")[2])
    default_asked = json.loads(default_requests[0].partition(b"\r\n\r\n")[2])
    assert asked["stream_format"] == StreamFormat.SSE.value
    assert "stream_format" not in default_asked


@pytest.mark.parametrize(
    ("answer", "failure"),
    [
        pytest.param(_answer(_delta(WAV), _marks(WORD)), Cut(), id="no-done"),
        pytest.param(
            _answer(_delta(WAV), _marks(WORD), ended=False), Cut(), id="no-terminator"
        ),
        pytest.param(_answer(DONE), Malformed(), id="done-before-any-audio"),
        pytest.param(
            _answer(_delta(WAV), DONE, _delta(WAV)), Malformed(), id="audio-after-done"
        ),
        pytest.param(_answer(_delta(WAV), DONE, DONE), Malformed(), id="a-second-done"),
        pytest.param(
            _answer(_marks(WORD), _delta(WAV), DONE),
            Malformed(),
            id="marks-before-audio",
        ),
        pytest.param(
            _answer(_delta(b"not a wav at all"), DONE),
            Malformed(),
            id="audio-that-is-no-wav",
        ),
        pytest.param(
            _answer(_delta(WAV), DONE, media_type="audio/wav"),
            Malformed(),
            id="not-an-event-stream",
        ),
        pytest.param(
            _answer(_delta(WAV), b'data: {"type": "speech.audio.other"}\n\n', DONE),
            Malformed(),
            id="unknown-type",
        ),
        pytest.param(
            _answer(_delta(WAV), b"data: not json\n\n", DONE),
            Malformed(),
            id="not-json",
        ),
        pytest.param(
            _answer(_delta(WAV), b'{"type": "speech.audio.done"}\n\n'),
            Malformed(),
            id="no-data-prefix",
        ),
        pytest.param(
            _answer(
                _delta(WAV),
                audio_event(AudioEventType.DELTA, audio="not base64!"),
                DONE,
            ),
            Malformed(),
            id="audio-that-is-not-base64",
        ),
        pytest.param(
            _answer(
                _delta(WAV),
                audio_event(AudioEventType.MARKS, words=[{"text": "x"}], sounds=None),
                DONE,
            ),
            Malformed(),
            id="a-word-without-its-times",
        ),
        pytest.param(
            _answer(_delta(WAV), b'data: {"type": "speech.audio.done"}\n', ended=True),
            Malformed(),
            id="ends-inside-an-event",
        ),
    ],
)
def test_a_stream_that_breaks_the_wire_ends_as_that_failure_and_never_finished(
    answer: bytes, failure: object, raw_listener: RawListener
) -> None:
    """WIRE-158: a stream without `done` failed; what is no event breaks the wire."""
    with raw_listener(answer) as (base_url, _):
        items = _stream(base_url)

    assert items[-1] == failure
    assert not any(isinstance(item, Finished) for item in items)


def test_an_event_that_never_ends_stops_over_the_cap(raw_listener: RawListener) -> None:
    """The reader holds no more than an event of the cap's audio can fill."""
    endless = b"data: " + b"A" * (CAP_BYTES // 2)

    with raw_listener(_answer(endless, endless, endless, endless)) as (base_url, _):
        items = _stream(base_url, cap_bytes=1024)

    assert items == [OverCap()]


def test_audio_past_the_cap_ends_over_the_cap(raw_listener: RawListener) -> None:
    with raw_listener(_answer(_delta(WAV), _delta(WAV), DONE)) as (base_url, _):
        items = _stream(base_url, cap_bytes=len(WAV) + 1)

    assert items[-1] == OverCap()


def test_every_marks_record_is_exported_from_the_speak_module() -> None:
    assert {"SentenceMarks", "Word", "MarkedStreamItem"} <= set(speak_module.__all__)
