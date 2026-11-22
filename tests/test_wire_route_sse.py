"""POST /v1/audio/speech with `stream_format` `sse`: audio, marks, done (WIRE-158)."""

from __future__ import annotations

import asyncio
import json
import logging
from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient
from logline import StepTimer

from speech_client.wire import (
    RefusalCode,
    ResponseFormat,
    SpeechRequest,
    StreamFormat,
)
from speech_service.failures import SpeakingFailedError
from speech_service.service import audio_chunks
from tests.support.fakes_support import (
    MARKED_SENTENCES,
    FakeMarkingSpeaking,
    FakeSpeaking,
)
from tests.support.speak_support import (
    ASGI_TIMEOUT_SECONDS,
    FailingSpeaking,
    assert_refusal,
    post_speech,
    speak_app,
    speak_body,
    speak_scope,
    sse_audio,
    sse_body,
    sse_events,
)
from wav_audio import pcm_from_wav

AUDIO_TYPE = "speech.audio.delta"
MARKS_TYPE = "speech.audio.marks"
DONE_TYPE = "speech.audio.done"


def _speak(
    speaking: FakeSpeaking,
    stream_format: StreamFormat,
    response_format: ResponseFormat = ResponseFormat.WAV,
) -> bytes:
    response = post_speech(
        TestClient(speak_app(speaking)),
        _body(stream_format, response_format),
    )
    assert response.status_code == HTTPStatus.OK
    return response.content


def test_the_default_answer_is_the_audio_whatever_a_voice_gives_as_marks() -> None:
    """WIRE-151: without `stream_format` the answer is the audio, as before."""
    answer = _speak(FakeMarkingSpeaking(), StreamFormat.AUDIO)

    assert pcm_from_wav(answer)[1] == b"".join(FakeSpeaking().pcm_chunks("", "de"))


def test_the_event_stream_answers_as_text_event_stream() -> None:
    response = post_speech(
        TestClient(speak_app(FakeMarkingSpeaking())),
        sse_body(),
    )

    assert response.headers["content-type"].split(";")[0] == "text/event-stream"


def test_the_deltas_of_the_event_stream_are_the_audio_the_default_answer_holds() -> (
    None
):
    """WIRE-158: the deltas are today's bytes in order."""
    events = sse_events(_speak(FakeMarkingSpeaking(), StreamFormat.SSE))

    assert sse_audio(events) == _speak(FakeMarkingSpeaking(), StreamFormat.AUDIO)


def test_each_sentence_is_its_audio_then_its_marks_and_the_stream_ends_with_done() -> (
    None
):
    """WIRE-158: one marks event per sentence after its audio; `done` comes last."""
    events = sse_events(_speak(FakeMarkingSpeaking(), StreamFormat.SSE))

    kinds = [event["type"] for event in events]
    assert kinds == [AUDIO_TYPE, MARKS_TYPE, AUDIO_TYPE, MARKS_TYPE, DONE_TYPE]
    marks = [event for event in events if event["type"] == MARKS_TYPE]
    assert [mark["words"] for mark in marks] == [
        [word.model_dump(mode="json")] for word in MARKED_SENTENCES
    ]


def test_a_done_event_carries_no_usage() -> None:
    events = sse_events(_speak(FakeMarkingSpeaking(), StreamFormat.SSE))

    assert events[-1] == {"type": DONE_TYPE}


def test_an_opus_event_stream_carries_the_ogg_and_the_marks() -> None:
    events = sse_events(
        _speak(FakeMarkingSpeaking(), StreamFormat.SSE, ResponseFormat.OPUS)
    )

    assert sse_audio(events)[:4] == b"OggS"
    kinds = [event["type"] for event in events]
    assert kinds.count(MARKS_TYPE) == len(MARKED_SENTENCES)
    assert kinds[-1] == DONE_TYPE
    assert kinds.index(MARKS_TYPE) > kinds.index(AUDIO_TYPE)


def test_a_voice_that_gives_no_marks_ends_its_audio_with_one_marks_event_of_nulls() -> (
    None
):
    """WIRE-152: a kind a voice does not give is `null`, never missing."""
    events = sse_events(_speak(FakeSpeaking(), StreamFormat.SSE))

    kinds = [event["type"] for event in events]
    assert kinds[-2:] == [MARKS_TYPE, DONE_TYPE]
    assert set(kinds[:-2]) == {AUDIO_TYPE}
    assert events[-2] == {"type": MARKS_TYPE, "words": None, "sounds": None}


def test_a_marks_event_is_null_for_sounds_that_no_voice_gives() -> None:
    events = sse_events(_speak(FakeMarkingSpeaking(), StreamFormat.SSE))

    marks = [event for event in events if event["type"] == MARKS_TYPE]
    assert all(mark["sounds"] is None for mark in marks)


def test_the_first_audio_leaves_before_the_marks_of_its_sentence_are_made() -> None:
    """The timings never delay the first audio: marks follow their audio."""
    speaking = FakeMarkingSpeaking()
    request = _request(StreamFormat.SSE)

    stream = audio_chunks(speaking, request, _step())
    first = next(stream)

    assert json.loads(first.removeprefix(b"data: "))["type"] == AUDIO_TYPE
    assert speaking.produced == [f"audio {MARKED_SENTENCES[0].text}"]
    stream.close()


def test_the_marks_of_a_sentence_leave_before_the_next_sentence_is_made() -> None:
    speaking = FakeMarkingSpeaking()
    stream = audio_chunks(speaking, _request(StreamFormat.SSE), _step())
    next(stream)

    marks = next(stream)

    assert json.loads(marks.removeprefix(b"data: "))["type"] == MARKS_TYPE
    assert speaking.produced[-1] == f"marks {MARKED_SENTENCES[0].text}"
    assert f"audio {MARKED_SENTENCES[1].text}" not in speaking.produced
    stream.close()


def test_a_stream_that_fails_is_cut_before_any_done() -> None:
    """WIRE-90: a stream without `done` failed; the body never ends."""
    messages: list[dict[str, object]] = []

    async def scenario() -> BaseException | None:
        sent = False
        never = asyncio.Event()

        async def receive() -> dict[str, object]:
            nonlocal sent
            if not sent:
                sent = True
                body = sse_body()
                return {"type": "http.request", "body": body, "more_body": False}
            await never.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        app = speak_app(FailingSpeaking(chunks_before_failing=1))
        try:
            await asyncio.wait_for(
                app(speak_scope(), receive, send), timeout=ASGI_TIMEOUT_SECONDS
            )
        except SpeakingFailedError as error:
            return error
        return None

    failure = asyncio.run(scenario())

    sent_body = b"".join(
        message["body"]
        for message in messages
        if message["type"] == "http.response.body"
    )
    assert isinstance(failure, SpeakingFailedError)
    assert all(DONE_TYPE not in event["type"] for event in sse_events(sent_body))
    assert all(
        message.get("more_body") is True
        for message in messages
        if message["type"] == "http.response.body"
    )


@pytest.mark.parametrize("value", ["mp3", "SSE", "", None, 1])
def test_a_stream_format_outside_the_wire_is_refused_on_stream_format(
    value: object,
) -> None:
    """WIRE-94, WIRE-151: any other `stream_format` is `invalid_request`."""
    body = json.dumps(json.loads(speak_body()) | {"stream_format": value}).encode()

    response = post_speech(TestClient(speak_app()), body)

    assert_refusal(
        response,
        HTTPStatus.UNPROCESSABLE_ENTITY,
        RefusalCode.INVALID_REQUEST,
        "stream_format",
    )


def _body(
    stream_format: StreamFormat, response_format: ResponseFormat = ResponseFormat.WAV
) -> bytes:
    if stream_format is StreamFormat.SSE:
        return sse_body(response_format=response_format)
    return speak_body(response_format=response_format)


def _request(stream_format: StreamFormat) -> SpeechRequest:
    return SpeechRequest.model_validate_json(_body(stream_format))


def _step() -> StepTimer:
    return StepTimer("speak.ended", logging.getLogger(__name__))
