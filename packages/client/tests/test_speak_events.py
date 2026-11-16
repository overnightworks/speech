"""The marked speaking stream against a raw listener the real service never matches."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
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
    Sound,
    Word,
    speak_stream,
)
from speech_client.wire import AudioEventType, StreamFormat, VoiceId, audio_event

if TYPE_CHECKING:
    from collections.abc import Callable

    from speech_client.speak import MarkedStreamItem
    from support.listener_support import RawListener


def _chunk(data: bytes) -> bytes:
    return b"%x\r\n%b\r\n" % (len(data), data)


@dataclass(frozen=True)
class Scenario:
    """One marked answer's audio and word, with the events and streams built on them."""

    deadline_seconds: float = 30.0
    cap_bytes: int = 1024 * 1024
    pcm: bytes = b"\x01\x00" * 8
    word: Word = field(
        default_factory=lambda: Word(
            text="Hallo", start_char=0, end_char=5, start=0.0, end=0.1
        )
    )

    @property
    def wav(self) -> bytes:
        """The WAV the answer carries."""
        return wav_of(self.pcm)

    @property
    def done(self) -> bytes:
        """The last event."""
        return audio_event(AudioEventType.DONE)

    def answer(
        self,
        *events: bytes,
        media_type: str = "text/event-stream",
        ended: bool = True,
    ) -> bytes:
        """A chunked 200 whose chunks are `events`."""
        head = (
            f"HTTP/1.1 200 OK\r\nContent-Type: {media_type}\r\n"
            "Transfer-Encoding: chunked\r\n\r\n"
        ).encode()
        return head + b"".join(map(_chunk, events)) + (b"0\r\n\r\n" if ended else b"")

    def delta(self, audio: bytes) -> bytes:
        """An audio event of `audio`."""
        return audio_event(AudioEventType.DELTA, audio=base64.b64encode(audio).decode())

    def marks(self, *words: Word) -> bytes:
        """A marks event of `words` and no sounds."""
        return audio_event(
            AudioEventType.MARKS,
            words=[word.model_dump(mode="json") for word in words],
            sounds=None,
        )

    def stream(
        self, base_url: str, *, cap_bytes: int | None = None
    ) -> list[MarkedStreamItem]:
        """Every item of the sse stream read from `base_url`."""
        return list(
            speak_stream(
                base_url,
                "Hallo",
                VoiceId.THORSTEN,
                stream_format=StreamFormat.SSE,
                workload_token="workload-token",  # noqa: S106 - a stand-in the raw listener never checks
                person_token="person-token",  # noqa: S106 - a stand-in the raw listener never checks
                deadline_seconds=self.deadline_seconds,
                cap_bytes=self.cap_bytes if cap_bytes is None else cap_bytes,
            )
        )


@pytest.fixture
def scenario() -> Scenario:
    return Scenario()


def test_the_items_are_the_audio_and_the_marks_in_arrival_order_then_finished(
    scenario: Scenario, raw_listener: RawListener
) -> None:
    wav = scenario.wav
    answer = scenario.answer(
        scenario.delta(wav[:20]),
        scenario.delta(wav[20:]),
        scenario.marks(scenario.word),
        scenario.done,
    )

    with raw_listener(answer) as (base_url, _):
        items = scenario.stream(base_url)

    assert items == [
        wav[:20],
        wav[20:],
        SentenceMarks(words=(scenario.word,)),
        Finished(len(scenario.pcm) // 2 / SAMPLE_RATE),
    ]


def test_a_marks_item_without_words_or_sounds_holds_none(
    scenario: Scenario, raw_listener: RawListener
) -> None:
    unproven = audio_event(AudioEventType.MARKS, words=None, sounds=None)
    answer = scenario.answer(scenario.delta(scenario.wav), unproven, scenario.done)

    with raw_listener(answer) as (base_url, _):
        items = scenario.stream(base_url)

    assert SentenceMarks(words=None, sounds=None) in items


def test_a_marks_item_carries_the_sounds_with_their_times(
    scenario: Scenario, raw_listener: RawListener
) -> None:
    sound = Sound(ipa="a", start=0.0, end=0.05, estimated=True)
    with_sounds = audio_event(
        AudioEventType.MARKS, words=None, sounds=[sound.model_dump(mode="json")]
    )
    answer = scenario.answer(scenario.delta(scenario.wav), with_sounds, scenario.done)

    with raw_listener(answer) as (base_url, _):
        items = scenario.stream(base_url)

    assert SentenceMarks(sounds=(sound,)) in items


def test_a_field_the_reader_does_not_know_is_ignored(
    scenario: Scenario, raw_listener: RawListener
) -> None:
    """WIRE-158: a reader ignores a field it does not know."""
    extended = audio_event(AudioEventType.DONE, usage={"tokens": 3})

    with raw_listener(scenario.answer(scenario.delta(scenario.wav), extended)) as (
        base_url,
        _,
    ):
        items = scenario.stream(base_url)

    assert isinstance(items[-1], Finished)


def test_the_request_asks_for_sse_and_the_default_request_does_not_name_it(
    scenario: Scenario, raw_listener: RawListener
) -> None:
    answer = scenario.answer(scenario.delta(scenario.wav), scenario.done)
    with raw_listener(answer) as (base_url, requests):
        scenario.stream(base_url)
    with raw_listener(answer) as (base_url, default_requests):
        list(
            speak_stream(
                base_url,
                "Hallo",
                VoiceId.THORSTEN,
                workload_token="workload-token",  # noqa: S106 - a stand-in the raw listener never checks
                person_token="person-token",  # noqa: S106 - a stand-in the raw listener never checks
                deadline_seconds=scenario.deadline_seconds,
                cap_bytes=scenario.cap_bytes,
            )
        )

    asked = json.loads(requests[0].partition(b"\r\n\r\n")[2])
    default_asked = json.loads(default_requests[0].partition(b"\r\n\r\n")[2])
    assert asked["stream_format"] == StreamFormat.SSE.value
    assert "stream_format" not in default_asked


@pytest.mark.parametrize(
    ("answer", "failure"),
    [
        pytest.param(
            lambda s: s.answer(s.delta(s.wav), s.marks(s.word)), Cut(), id="no-done"
        ),
        pytest.param(
            lambda s: s.answer(s.delta(s.wav), s.marks(s.word), ended=False),
            Cut(),
            id="no-terminator",
        ),
        pytest.param(
            lambda s: s.answer(s.done), Malformed(), id="done-before-any-audio"
        ),
        pytest.param(
            lambda s: s.answer(s.delta(s.wav), s.done, s.delta(s.wav)),
            Malformed(),
            id="audio-after-done",
        ),
        pytest.param(
            lambda s: s.answer(s.delta(s.wav), s.done, s.done),
            Malformed(),
            id="a-second-done",
        ),
        pytest.param(
            lambda s: s.answer(s.marks(s.word), s.delta(s.wav), s.done),
            Malformed(),
            id="marks-before-audio",
        ),
        pytest.param(
            lambda s: s.answer(s.delta(b"not a wav at all"), s.done),
            Malformed(),
            id="audio-that-is-no-wav",
        ),
        pytest.param(
            lambda s: s.answer(s.delta(s.wav), s.done, media_type="audio/wav"),
            Malformed(),
            id="not-an-event-stream",
        ),
        pytest.param(
            lambda s: s.answer(
                s.delta(s.wav), b'data: {"type": "speech.audio.other"}\n\n', s.done
            ),
            Malformed(),
            id="unknown-type",
        ),
        pytest.param(
            lambda s: s.answer(s.delta(s.wav), b"data: not json\n\n", s.done),
            Malformed(),
            id="not-json",
        ),
        pytest.param(
            lambda s: s.answer(s.delta(s.wav), b'{"type": "speech.audio.done"}\n\n'),
            Malformed(),
            id="no-data-prefix",
        ),
        pytest.param(
            lambda s: s.answer(
                s.delta(s.wav),
                audio_event(AudioEventType.DELTA, audio="not base64!"),
                s.done,
            ),
            Malformed(),
            id="audio-that-is-not-base64",
        ),
        pytest.param(
            lambda s: s.answer(
                s.delta(s.wav),
                audio_event(AudioEventType.MARKS, words=[{"text": "x"}], sounds=None),
                s.done,
            ),
            Malformed(),
            id="a-word-without-its-times",
        ),
        pytest.param(
            lambda s: s.answer(
                s.delta(s.wav),
                audio_event(
                    AudioEventType.MARKS,
                    words=None,
                    sounds=[{"ipa": "a", "start": 0, "end": 1}],
                ),
                s.done,
            ),
            Malformed(),
            id="a-sound-without-its-flag",
        ),
        pytest.param(
            lambda s: s.answer(
                s.delta(s.wav), b'data: {"type": "speech.audio.done"}\n', ended=True
            ),
            Malformed(),
            id="ends-inside-an-event",
        ),
    ],
)
def test_a_stream_that_breaks_the_wire_ends_as_that_failure_and_never_finished(
    answer: Callable[[Scenario], bytes],
    failure: object,
    scenario: Scenario,
    raw_listener: RawListener,
) -> None:
    """WIRE-158: a stream without `done` failed; what is no event breaks the wire."""
    with raw_listener(answer(scenario)) as (base_url, _):
        items = scenario.stream(base_url)

    assert items[-1] == failure
    assert not any(isinstance(item, Finished) for item in items)


def test_an_event_that_never_ends_stops_over_the_cap(
    scenario: Scenario, raw_listener: RawListener
) -> None:
    """The reader holds no more than an event of the cap's audio can fill."""
    endless = b"data: " + b"A" * (scenario.cap_bytes // 2)

    with raw_listener(scenario.answer(endless, endless, endless, endless)) as (
        base_url,
        _,
    ):
        items = scenario.stream(base_url, cap_bytes=1024)

    assert items == [OverCap()]


def test_a_whole_event_over_the_size_of_the_cap_stops_over_the_cap(
    scenario: Scenario, raw_listener: RawListener
) -> None:
    """A whole event over the limit is refused as a partial one is."""
    padded = audio_event(AudioEventType.DONE, padding="A" * 70_000)
    answer = scenario.answer(scenario.delta(scenario.wav), padded)

    with raw_listener(answer) as (base_url, _):
        items = scenario.stream(base_url, cap_bytes=1024)

    assert items[-1] == OverCap()


def test_audio_past_the_cap_ends_over_the_cap(
    scenario: Scenario, raw_listener: RawListener
) -> None:
    wav = scenario.wav
    answer = scenario.answer(scenario.delta(wav), scenario.delta(wav), scenario.done)

    with raw_listener(answer) as (base_url, _):
        items = scenario.stream(base_url, cap_bytes=len(wav) + 1)

    assert items[-1] == OverCap()


def test_every_marks_record_is_exported_from_the_speak_module() -> None:
    assert {"SentenceMarks", "Word", "Sound", "MarkedStreamItem"} <= set(
        speak_module.__all__
    )
