"""The client's marked speaking stream against the real service over TCP (WIRE-158)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from speech_client.speak import (
    Finished,
    SentenceMarks,
    speak_stream,
)
from speech_client.voice_preference import (
    VoiceChosen,
    speak_stream_with_preference,
)
from speech_client.wire import Language, StreamFormat, VoiceId
from speech_service.service import Runtime
from tests.support.fakes_support import (
    GATE_TIMEOUT_SECONDS,
    MARKED_SENTENCES,
    FakeMarkingSpeaking,
    FakeSpeaking,
)
from tests.support.identity_support import valid_pair
from tests.support.tcp_server_support import serving_over_tcp

if TYPE_CHECKING:
    from collections.abc import Callable

    from speech_client.speak import MarkedStreamItem, SpeakStreamItem
    from speech_client.voice_preference import (
        MarkedPreferenceStreamItem,
        PreferenceStreamItem,
    )

CAP_BYTES = 1024 * 1024


def _served[Result](voice: FakeSpeaking, consume: Callable[[str], Result]) -> Result:
    runtime = Runtime({VoiceId.THORSTEN: voice}, memory_probe=lambda: 0)

    async def scenario() -> Result:
        async with serving_over_tcp(runtime) as server:
            return await asyncio.to_thread(consume, server.base_url)

    return asyncio.run(scenario())


def _marked(voice: FakeSpeaking) -> list[MarkedStreamItem]:
    pair = valid_pair()
    return _served(
        voice,
        lambda base_url: list(
            speak_stream(
                base_url,
                "Hallo",
                VoiceId.THORSTEN,
                stream_format=StreamFormat.SSE,
                workload_token=pair.workload,
                person_token=pair.person,
                deadline_seconds=GATE_TIMEOUT_SECONDS,
                cap_bytes=CAP_BYTES,
            )
        ),
    )


def _plain(voice: FakeSpeaking) -> list[SpeakStreamItem]:
    pair = valid_pair()
    return _served(
        voice,
        lambda base_url: list(
            speak_stream(
                base_url,
                "Hallo",
                VoiceId.THORSTEN,
                workload_token=pair.workload,
                person_token=pair.person,
                deadline_seconds=GATE_TIMEOUT_SECONDS,
                cap_bytes=CAP_BYTES,
            )
        ),
    )


def _marked_preference(
    voice: FakeSpeaking,
) -> list[MarkedPreferenceStreamItem]:
    pair = valid_pair()
    return _served(
        voice,
        lambda base_url: list(
            speak_stream_with_preference(
                base_url,
                "Hallo",
                [VoiceId.RYAN, VoiceId.THORSTEN],
                language=Language.DE,
                instructions="Sprich ruhig",
                stream_format=StreamFormat.SSE,
                workload_token=pair.workload,
                person_token=pair.person,
                deadline_seconds=GATE_TIMEOUT_SECONDS,
                cap_bytes=CAP_BYTES,
            )
        ),
    )


def _audio(items: list[MarkedStreamItem]) -> bytes:
    return b"".join(item for item in items if isinstance(item, bytes))


def test_a_marked_stream_hands_over_each_sentences_audio_then_its_marks() -> None:
    """WIRE-157, WIRE-158: audio and marks arrive in the order the service sent."""
    *items, end = _marked(FakeMarkingSpeaking())

    assert [type(item) for item in items] == [
        bytes,
        SentenceMarks,
        bytes,
        SentenceMarks,
    ]
    assert [item for item in items if isinstance(item, SentenceMarks)] == [
        SentenceMarks(words=(word,)) for word in MARKED_SENTENCES
    ]
    assert isinstance(end, Finished)


def test_the_audio_of_a_marked_stream_is_the_audio_of_the_default_stream() -> None:
    """WIRE-158: the deltas joined are the audio the default answer carries."""
    *plain_chunks, plain_end = _plain(FakeMarkingSpeaking())

    *_, marked_end = marked = _marked(FakeMarkingSpeaking())

    assert _audio(marked) == b"".join(
        chunk for chunk in plain_chunks if isinstance(chunk, bytes)
    )
    assert marked_end == plain_end


def test_the_default_stream_of_a_voice_that_marks_carries_no_marks() -> None:
    """WIRE-151: without `stream_format` the items are the audio and its end."""
    *items, end = _plain(FakeMarkingSpeaking())

    assert all(isinstance(item, bytes) for item in items)
    assert isinstance(end, Finished)


def test_a_voice_that_marks_nothing_sends_one_marks_item_without_words() -> None:
    """WIRE-152: the marks are there, their words unproven."""
    *items, end = _marked(FakeSpeaking())

    assert [item for item in items if isinstance(item, SentenceMarks)] == [
        SentenceMarks(words=None)
    ]
    assert isinstance(end, Finished)


def test_a_marked_preference_names_the_voice_first_then_audio_marks_and_the_end() -> (
    None
):
    """WIRE-125, WIRE-142: the preference chooses as without `sse`."""
    first, *items, end = _marked_preference(FakeMarkingSpeaking())

    assert first == VoiceChosen(VoiceId.THORSTEN, instructions_left_out=True)
    assert [type(item) for item in items] == [
        bytes,
        SentenceMarks,
        bytes,
        SentenceMarks,
    ]
    assert isinstance(end, Finished)


def test_the_default_preference_stream_keeps_its_item_types() -> None:
    pair = valid_pair()

    def stream(base_url: str) -> list[PreferenceStreamItem]:
        return list(
            speak_stream_with_preference(
                base_url,
                "Hallo",
                [VoiceId.THORSTEN],
                workload_token=pair.workload,
                person_token=pair.person,
                deadline_seconds=GATE_TIMEOUT_SECONDS,
                cap_bytes=CAP_BYTES,
            )
        )

    first, *items, end = _served(FakeMarkingSpeaking(), stream)

    assert isinstance(first, VoiceChosen)
    assert all(isinstance(item, bytes) for item in items)
    assert isinstance(end, Finished)
