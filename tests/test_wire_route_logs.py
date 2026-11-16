"""POST /v1/audio/speech: what a spoken, refused or failed request writes to the log."""

import asyncio
import json
import logging
import traceback
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from logline.testing import configured
from starlette.status import HTTP_200_OK

from speech_client.wire import SPEECH_PATH, RefusalCode, SentenceMarks, Sound, VoiceId
from speech_service.config import Settings
from speech_service.failures import SpeakingFailedError
from tests.support.fakes_support import FakeMarkingSpeaking, FakeSpeaking
from tests.support.identity_support import valid_headers
from tests.support.speak_support import (
    ASGI_TIMEOUT_SECONDS,
    DisconnectingReceive,
    HoldingSend,
    app_over,
    fake_runtime,
    post_speech,
    speak_app,
    speak_body,
    speak_once,
    speak_scope,
    sse_body,
    status_of,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi import FastAPI


def _wrong_voice_body() -> bytes:
    return json.dumps(json.loads(speak_body()) | {"voice": _SPOKEN}).encode()


_SPOKEN = "Nur dieser Satz darf nicht in den Logs stehen"


class _SentenceCarryingError(RuntimeError):
    """A third-party failure that holds the input in its message."""


def _error_holding_the_sentence() -> _SentenceCarryingError:
    cause = KeyError(_SPOKEN)
    error = _SentenceCarryingError(_SPOKEN)
    error.__cause__ = cause
    return error


class _FailingOnceSpeaking(FakeSpeaking):
    """Fails with the sentence in its error, optionally after one chunk."""

    def __init__(self, *, after_first_chunk: bool = False) -> None:
        super().__init__()
        self.fails = True
        self._after_first_chunk = after_first_chunk

    def pcm_chunks(
        self, text: str, language: str, instructions: str | None = None
    ) -> Iterator[bytes]:
        if not self.fails:
            yield from super().pcm_chunks(text, language, instructions)
            return
        if self._after_first_chunk:
            yield next(iter(super().pcm_chunks(text, language, instructions)))
        raise _error_holding_the_sentence()


def _assert_failure_logged_without_text(
    caplog: pytest.LogCaptureFixture, raised: BaseException
) -> None:
    (failure,) = (r for r in caplog.records if r.getMessage() == "speaking.failed")
    assert not hasattr(failure, "failure_chain")
    exception = logging.Formatter().formatException(failure.exc_info)
    assert "_SentenceCarryingError" in exception
    assert "KeyError" in exception
    assert 'File "' in exception
    assert _SPOKEN not in f"{caplog.text} {vars(failure)} {exception}"
    assert _SPOKEN not in "".join(traceback.format_exception(raised))


def _speak_collecting(
    app: FastAPI,
) -> tuple[list[dict[str, object]], BaseException | None]:
    async def scenario() -> tuple[list[dict[str, object]], BaseException | None]:
        messages: list[dict[str, object]] = []
        sent = False
        never = asyncio.Event()

        async def receive() -> dict[str, object]:
            nonlocal sent
            if not sent:
                sent = True
                return {
                    "type": "http.request",
                    "body": speak_body(),
                    "more_body": False,
                }
            await never.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        try:
            await asyncio.wait_for(
                app(speak_scope(), receive, send), timeout=ASGI_TIMEOUT_SECONDS
            )
        except SpeakingFailedError as error:
            return messages, error
        return messages, None

    return asyncio.run(scenario())


def test_speak_writes_no_sentence_to_the_log_whatever_the_environment_says(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEECH_DEBUG", "true")
    caplog.set_level(logging.DEBUG)

    response = post_speech(TestClient(speak_app()), speak_body(text=_SPOKEN))

    assert response.status_code == HTTP_200_OK
    assert _SPOKEN not in caplog.text


def test_speak_failure_before_the_first_chunk_logs_types_and_never_the_sentence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    _messages, raised = _speak_collecting(speak_app(_FailingOnceSpeaking()))

    assert isinstance(raised, SpeakingFailedError)
    _assert_failure_logged_without_text(caplog, raised)


def test_speak_failure_after_a_chunk_never_ends_the_body_and_the_next_request_is_served(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    speaking = _FailingOnceSpeaking(after_first_chunk=True)
    app = speak_app(speaking)

    messages, raised = _speak_collecting(app)
    speaking.fails = False
    later = asyncio.run(speak_once(app))

    assert isinstance(raised, SpeakingFailedError)
    _assert_failure_logged_without_text(caplog, raised)
    bodies = [m for m in messages if m["type"] == "http.response.body"]
    assert bodies
    assert all(body.get("more_body") is True for body in bodies)
    assert status_of(later) == HTTP_200_OK


def test_speak_cleanup_failure_after_a_disconnect_logs_no_sentence_and_frees_the_voice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    class CleanupFails(FakeSpeaking):
        failing_cleanup = True

        def pcm_chunks(
            self, text: str, language: str, instructions: str | None = None
        ) -> Iterator[bytes]:
            try:
                yield from super().pcm_chunks(text, language, instructions)
            finally:
                if self.failing_cleanup:
                    raise _error_holding_the_sentence()

    speaking = CleanupFails()
    app = speak_app(speaking)

    async def scenario() -> tuple[BaseException | None, list[dict[str, object]]]:
        receive = DisconnectingReceive()
        send = HoldingSend(hold_after_first_chunk=True)
        call = asyncio.create_task(app(speak_scope(), receive, send))
        await asyncio.wait_for(send.entered_hold.wait(), timeout=ASGI_TIMEOUT_SECONDS)
        receive.disconnect()
        failure: BaseException | None = None
        try:
            await asyncio.wait_for(call, timeout=ASGI_TIMEOUT_SECONDS)
        except SpeakingFailedError as error:
            failure = error
        speaking.failing_cleanup = False
        return failure, await speak_once(app)

    failure, later = asyncio.run(scenario())

    assert isinstance(failure, SpeakingFailedError)
    _assert_failure_logged_without_text(caplog, failure)
    assert status_of(later) == HTTP_200_OK


def _speak_lines(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == "speak.ended"]


def test_a_spoken_request_writes_one_speak_line_with_its_voice_and_first_audio(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    response = post_speech(TestClient(speak_app()), speak_body(text=_SPOKEN))

    assert response.status_code == HTTP_200_OK
    (line,) = _speak_lines(caplog)
    assert line.outcome == "ok"
    assert line.voice == VoiceId.THORSTEN.value
    assert 0 <= line.first_output_ms <= line.duration_ms
    assert _SPOKEN not in f"{caplog.text} {vars(line)}"


class _SoundMarkingSpeaking(FakeMarkingSpeaking):
    """A marking voice that also marks two sounds in each sentence."""

    def marked_chunks(
        self, text: str, language: str, instructions: str | None = None
    ) -> Iterator[bytes | SentenceMarks]:
        """Each sentence's marks gain two sounds."""
        for item in super().marked_chunks(text, language, instructions):
            if isinstance(item, SentenceMarks):
                yield SentenceMarks(
                    words=item.words,
                    sounds=(
                        Sound(ipa="h", start=0.0, end=0.05),
                        Sound(ipa="a", start=0.05, end=0.1),
                    ),
                )
            else:
                yield item


@pytest.mark.parametrize(
    ("body", "stream_format", "marked_words", "marked_sounds"),
    [
        (speak_body(text=_SPOKEN), "audio", 0, 0),
        (sse_body(text=_SPOKEN), "sse", 2, 4),
    ],
    ids=["audio", "sse"],
)
def test_a_speak_line_counts_the_words_and_sounds_it_marked_and_never_names_them(
    caplog: pytest.LogCaptureFixture,
    body: bytes,
    stream_format: str,
    marked_words: int,
    marked_sounds: int,
) -> None:
    """WIRE-67: `speak.ended` carries its stream format and counts only."""
    caplog.set_level(logging.DEBUG)

    response = post_speech(TestClient(speak_app(_SoundMarkingSpeaking())), body)

    assert response.status_code == HTTP_200_OK
    (line,) = _speak_lines(caplog)
    assert (line.stream_format, line.marks_words, line.marks_sounds) == (
        stream_format,
        marked_words,
        marked_sounds,
    )
    assert _SPOKEN not in f"{caplog.text} {vars(line)}"


@pytest.mark.parametrize(
    "body",
    [_wrong_voice_body(), speak_body(text=_SPOKEN * 100)],
    ids=["unparsed-request", "too-long-input"],
)
def test_a_request_refused_before_its_voice_is_known_writes_a_speak_line_without_one(
    caplog: pytest.LogCaptureFixture, body: bytes
) -> None:
    caplog.set_level(logging.INFO)
    client = TestClient(speak_app(settings=Settings(max_input_characters=1000)))

    post_speech(client, body)

    (line,) = _speak_lines(caplog)
    assert (line.outcome, line.reason) == (
        "refused",
        RefusalCode.INVALID_REQUEST.value,
    )
    assert not hasattr(line, "voice")


def test_a_request_refused_before_streaming_writes_a_speak_line_with_its_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = fake_runtime()
    runtime.begin_shutdown()
    caplog.set_level(logging.INFO)

    post_speech(TestClient(app_over(runtime)), speak_body())

    (line,) = _speak_lines(caplog)
    assert (line.outcome, line.reason) == ("refused", RefusalCode.UNAVAILABLE.value)
    assert line.voice == VoiceId.THORSTEN.value
    assert not hasattr(line, "first_output_ms")


def test_a_failed_speaking_writes_a_speak_line_with_the_error_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    _speak_collecting(speak_app(_FailingOnceSpeaking()))

    (line,) = _speak_lines(caplog)
    assert (line.outcome, line.error_type) == ("failed", "SpeakingFailedError")


def test_a_refused_unauthenticated_request_writes_a_speak_line_with_its_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    post_speech(TestClient(speak_app()), speak_body(), headers={})

    (line,) = _speak_lines(caplog)
    assert (line.outcome, line.reason) == ("refused", RefusalCode.UNAUTHORIZED.value)


def test_a_failed_speaking_writes_one_error_line_the_speak_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    _speak_collecting(speak_app(_FailingOnceSpeaking()))

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert [r.getMessage() for r in errors] == ["speak.ended"]


def test_a_client_that_leaves_before_the_first_chunk_writes_its_speak_line_as_cancelled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    app = speak_app()

    async def scenario() -> None:
        receive = DisconnectingReceive()
        started = asyncio.Event()
        never = asyncio.Event()

        async def send(message: dict[str, object]) -> None:
            if message["type"] == "http.response.start":
                started.set()
                await never.wait()

        call = asyncio.create_task(app(speak_scope(), receive, send))
        await asyncio.wait_for(started.wait(), timeout=ASGI_TIMEOUT_SECONDS)
        receive.disconnect()
        await asyncio.wait_for(call, timeout=ASGI_TIMEOUT_SECONDS)

    asyncio.run(scenario())

    (line,) = _speak_lines(caplog)
    assert line.outcome == "cancelled"
    assert not hasattr(line, "first_output_ms")


def test_the_speak_line_and_the_request_line_carry_the_trace_of_the_callers_traceparent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    traceparent = f"00-{trace_id}-00f067aa0ba902b7-01"

    with configured("speech"):
        response = post_speech(
            TestClient(speak_app()),
            speak_body(),
            headers={**valid_headers(), "traceparent": traceparent},
        )

    assert response.status_code == HTTP_200_OK
    written = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (speak,) = (line for line in written if line["event"] == "speak.ended")
    (request,) = (line for line in written if line["event"] == "request.ended")
    assert speak["trace_id"] == trace_id
    assert request["route"] == SPEECH_PATH
