"""The speaking calls: one `POST /v1/audio/speech`, bounded in time and size.

`speak_stream` hands the WAV's chunks over as they arrive; `speak` is that
stream read to its end, so both share one transport and one set of refusals.
Both take every input from their call site.

`speak_stream` yields the `bytes` chunks of the WAV (WIRE-88, WIRE-90) and then
exactly one last item, which is never missing:

- `Finished`: the body ended whole; it names the exact seconds of audio it
  carried, counted from the PCM bytes at the sample rate in the WAV header. The
  stream promises no length before then.
- a failure, any outcome below but `Spoken`; it is the only item when it comes
  before any audio, as every refusal does. A body that broke off is a failure,
  never `Finished`.

With `stream_format` `StreamFormat.SSE` the same stream carries the event
stream (WIRE-158): the audio chunks in order, each sentence's `SentenceMarks`
(WIRE-152) after the audio of that sentence, then `Finished` once `done` came.
A body that ends without `done` is `Cut`, as one that broke off is; an event the
wire does not know, one after `done` and audio that is no WAV are `Malformed`.
Without it every item is as above, and no `SentenceMarks` comes.

The deadline starts with the first item asked for. Close the stream
(`contextlib.closing`) to end the call early.

`speak` returns one `SpeakOutcome`, a closed union a caller matches with
`assert_never`:

- `Spoken`: the audio bytes exactly as received, in the ordered format, read to
  the end of the chunked body (WIRE-90).
- `Refused`: a refusal of the wire, with its status, code and param.
- `Busy`: the 429 and the wait the service asks for; the app decides to retry.
- `Malformed`: the answer breaks the wire, for example a 200 that is not
  chunked, is not the ordered format's media type or does not begin with its
  container signature (WIRE-88, WIRE-133), a WAV with no `data` audio or a zero
  sample rate, a refusal outside the error form, a 429 without a whole-second
  `Retry-After`, or a refusal body over 64 KiB.
- `Cut`, `PastDeadline`, `OverCap` and `Unreachable`, shared with the health
  fetch (`speech_client.outcomes`); here `OverCap` is the audio.

A defect of the caller raises and is no outcome: an input the request refuses
(`pydantic.ValidationError`), a deadline or cap of 0 or below, and a base URL
that is not `http`, has no host or carries a path, query or fragment
(`ValueError`). The service
is plain HTTP on internal networks.

The calls block and hold no process-global state, so they run in a worker
thread (`asyncio.to_thread`); the thread ends by the deadline. They log nothing,
and no outcome carries the text or a token. The name lookup is bounded by the
system resolver, not by the deadline.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING, Final, Literal, overload

from pydantic import ValidationError

from speech_client._traceparent import traceparent_headers
from speech_client._transport import (
    body_pieces,
    checked_origin,
    exchange_stream,
    read_within,
)
from speech_client.outcomes import (
    Busy,
    Cut,
    Malformed,
    OverCap,
    PastDeadline,
    RetryAfterError,
    Unreachable,
    busy_wait,
)
from speech_client.wire import (
    EVENT_END,
    SPEECH_PATH,
    Language,
    Refusal,
    RefusalCode,
    RequestField,
    ResponseFormat,
    SentenceMarks,
    Sound,
    StreamFormat,
    VoiceId,
    Word,
    answer_media_type,
    read_audio_event,
    refusal,
    speech_request_body,
    speech_request_headers,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable
    from http.client import HTTPConnection, HTTPResponse

__all__ = [
    "Busy",
    "Cut",
    "Finished",
    "Malformed",
    "MarkedStreamItem",
    "OverCap",
    "PastDeadline",
    "Refused",
    "SentenceMarks",
    "Sound",
    "SpeakOutcome",
    "SpeakStreamItem",
    "Spoken",
    "Unreachable",
    "Word",
    "speak",
    "speak_stream",
]

_REFUSAL_BODY_LIMIT: Final = 64 * 1024
# The longest container signature: `RIFF`, a size and `WAVE`.
_SIGNATURE_BYTES: Final = 12
# WIRE-88: 16-bit mono PCM.
_BYTES_PER_SAMPLE: Final = 2
_CHUNK_HEADER_BYTES: Final = 8
_RIFF_HEADER_BYTES: Final = 12
_SAMPLE_RATE_AT: Final = slice(24, 28)
_EVENT_END_BYTES: Final = EVENT_END.encode()
# What an event may carry beyond its audio: the envelope and one sentence's marks.
_EVENT_EXTRA_BYTES: Final = 64 * 1024


@dataclass(frozen=True, slots=True)
class Spoken:
    """The audio, exactly as the service sent it."""

    audio: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class Refused:
    """A refusal of the wire."""

    refusal: Refusal

    @property
    def status(self) -> HTTPStatus:
        """The HTTP status the service answered."""
        return self.refusal.status

    @property
    def code(self) -> RefusalCode:
        """Why the service refused."""
        return self.refusal.error.code

    @property
    def param(self) -> RequestField | None:
        """The request field at fault, if any."""
        return self.refusal.error.param


type StreamFailure = (
    Refused | Busy | Cut | PastDeadline | OverCap | Malformed | Unreachable
)
type SpeakOutcome = Spoken | StreamFailure
type SpeakStreamItem = bytes | Finished | StreamFailure
type MarkedStreamItem = SpeakStreamItem | SentenceMarks
type _Content = bytes | SentenceMarks | OverCap | Malformed | Cut


@dataclass(frozen=True, slots=True)
class Finished:
    """The body ended whole, and the audio it carried lasted exactly `seconds`."""

    seconds: float


def _judged_head(head: bytearray, response_format: ResponseFormat) -> bytes | Malformed:
    signature = bytes(head)
    if response_format.begins_like_container(signature):
        return signature
    return Malformed()


def _event_limit(cap_bytes: int) -> int:
    """The most one event holds: `cap_bytes` of audio in base64 and its marks."""
    return (cap_bytes + 2) // 3 * 4 + _EVENT_EXTRA_BYTES


def _frames(
    response: HTTPResponse, limit_bytes: int
) -> Generator[bytes | OverCap | Malformed]:
    """The events of the body, each without its blank line."""
    pending = bytearray()
    for piece in body_pieces(response):
        pending += piece
        while (end := pending.find(_EVENT_END_BYTES)) >= 0:
            if end > limit_bytes:
                yield OverCap()
                return
            yield bytes(pending[:end])
            del pending[: end + len(_EVENT_END_BYTES)]
        if len(pending) > limit_bytes:
            yield OverCap()
            return
    if pending:
        yield Malformed()


def _events(frames: Iterable[bytes | OverCap | Malformed]) -> Generator[_Content]:
    """The audio and marks the events carry, ended by `done` or by a failure."""
    done = False
    for frame in frames:
        if not isinstance(frame, bytes):
            yield frame
            return
        if done:
            yield Malformed()
            return
        try:
            content = read_audio_event(frame)
        except ValueError:
            yield Malformed()
            return
        if content is None:
            done = True
        else:
            yield content
    if not done:
        yield Cut()


def _within_cap(items: Iterable[_Content], cap_bytes: int) -> Generator[_Content]:
    received = 0
    for item in items:
        if isinstance(item, bytes):
            received += len(item)
            if received > cap_bytes:
                yield OverCap()
                return
        yield item


def _signed(
    items: Iterable[_Content], response_format: ResponseFormat
) -> Generator[_Content]:
    """`items`, once the first audio bytes begin like the ordered container.

    Marks that come before a whole signature make the answer `Malformed`: no
    container signature is shorter than that.
    """
    remaining = iter(items)
    head = bytearray()
    for item in remaining:
        if isinstance(item, OverCap | Malformed | Cut):
            yield item
            return
        if isinstance(item, bytes):
            head += item
            if len(head) < _SIGNATURE_BYTES:
                continue
        judged = _judged_head(head, response_format)
        yield judged
        if not isinstance(judged, Malformed):
            yield from remaining
        return
    yield _judged_head(head, response_format)


def _delivered(
    response: HTTPResponse,
    cap_bytes: int,
    response_format: ResponseFormat,
    stream_format: StreamFormat,
) -> Generator[_Content]:
    marked = stream_format is StreamFormat.SSE
    media_type = answer_media_type(response_format, stream_format)
    if response.headers.get_content_type() != media_type or not response.chunked:
        yield Malformed()
        return
    items = (
        _events(_frames(response, _event_limit(cap_bytes)))
        if marked
        else body_pieces(response)
    )
    yield from _signed(_within_cap(items, cap_bytes), response_format)


def _refusal(response: HTTPResponse) -> SpeakOutcome:
    body = read_within(response, _REFUSAL_BODY_LIMIT)
    if body is None:
        return Malformed()
    try:
        parsed = Refusal.model_validate_json(body)
        wait = busy_wait(response.status, dict(response.getheaders()))
    except ValidationError, RetryAfterError:
        return Malformed()
    if parsed.status != response.status:
        return Malformed()
    if parsed != refusal(parsed.error.code, parsed.error.param):
        return Malformed()
    return Refused(parsed) if wait is None else Busy(wait)


def _answer(
    connection: HTTPConnection,
    body: bytes,
    headers: dict[str, str],
    delivered: Callable[[HTTPResponse], Generator[_Content]],
) -> Generator[bytes | SentenceMarks | StreamFailure]:
    connection.request("POST", SPEECH_PATH, body, headers)
    response = connection.getresponse()
    if response.status == HTTPStatus.OK:
        yield from delivered(response)
    else:
        yield _refusal(response)


def _data_offset(head: bytes) -> int | None:
    """Where the PCM begins in `head`, or `None` while the `data` header is partial."""
    position = _RIFF_HEADER_BYTES
    formatted = False
    while position + _CHUNK_HEADER_BYTES <= len(head):
        chunk_id = head[position : position + 4]
        size = int.from_bytes(
            head[position + 4 : position + _CHUNK_HEADER_BYTES], "little"
        )
        if chunk_id == b"data":
            return position + _CHUNK_HEADER_BYTES if formatted else None
        formatted = formatted or chunk_id == b"fmt "
        # RIFF chunks are padded to an even size.
        position += _CHUNK_HEADER_BYTES + size + size % 2
    return None


class _WavLength:
    """The audio length of a WAV that arrives in pieces, counted from its PCM bytes."""

    def __init__(self) -> None:
        self._head = bytearray()
        self._pcm_bytes = 0
        self._sample_rate = 0
        self._in_data = False

    def feed(self, piece: bytes) -> None:
        if self._in_data:
            self._pcm_bytes += len(piece)
            return
        self._head += piece
        pcm_at = _data_offset(bytes(self._head))
        if pcm_at is not None:
            self._in_data = True
            self._pcm_bytes = len(self._head) - pcm_at
            self._sample_rate = int.from_bytes(self._head[_SAMPLE_RATE_AT], "little")
            self._head.clear()

    def seconds(self) -> float | None:
        """The exact seconds fed, or `None` if the body holds no usable `data`."""
        if not self._in_data or self._sample_rate == 0:
            return None
        return self._pcm_bytes // _BYTES_PER_SAMPLE / self._sample_rate


def _chunks(  # noqa: PLR0913 - each wire option is its own argument
    base_url: str,
    text: str,
    voice: VoiceId,
    *,
    language: Language | None,
    instructions: str | None,
    response_format: ResponseFormat,
    workload_token: str,
    person_token: str,
    deadline_seconds: float,
    cap_bytes: int,
    traceparent: str | None = None,
    stream_format: StreamFormat = StreamFormat.AUDIO,
) -> Generator[bytes | SentenceMarks | StreamFailure]:
    """The one transport: the body's chunks, or the failure that ends them."""
    origin = checked_origin(base_url, deadline_seconds, cap_bytes)
    body = speech_request_body(
        text,
        voice,
        response_format,
        language,
        instructions,
        stream_format=stream_format,
    )
    headers = {
        **speech_request_headers(
            workload_token=workload_token, person_token=person_token
        ),
        **traceparent_headers(traceparent),
    }
    return exchange_stream(
        origin,
        deadline_seconds,
        lambda connection: _answer(
            connection,
            body,
            headers,
            lambda response: _delivered(
                response, cap_bytes, response_format, stream_format
            ),
        ),
    )


def _with_length(
    chunks: Generator[bytes | SentenceMarks | StreamFailure],
) -> Generator[MarkedStreamItem]:
    length = _WavLength()
    with closing(chunks):
        for item in chunks:
            yield item
            if isinstance(item, SentenceMarks):
                continue
            if not isinstance(item, bytes):
                return
            length.feed(item)
    seconds = length.seconds()
    yield Malformed() if seconds is None else Finished(seconds)


# Every input comes from the call site; only `language`, `instructions` and
# `stream_format` default.
@overload
def speak_stream(
    base_url: str,
    text: str,
    voice: VoiceId,
    *,
    language: Language | None = None,
    instructions: str | None = None,
    stream_format: Literal[StreamFormat.AUDIO] = StreamFormat.AUDIO,
    workload_token: str,
    person_token: str,
    deadline_seconds: float,
    cap_bytes: int,
    traceparent: str | None = None,
) -> Generator[SpeakStreamItem]: ...


@overload
def speak_stream(
    base_url: str,
    text: str,
    voice: VoiceId,
    *,
    language: Language | None = None,
    instructions: str | None = None,
    stream_format: Literal[StreamFormat.SSE],
    workload_token: str,
    person_token: str,
    deadline_seconds: float,
    cap_bytes: int,
    traceparent: str | None = None,
) -> Generator[MarkedStreamItem]: ...


def speak_stream(  # noqa: PLR0913 - each wire option is its own argument
    base_url: str,
    text: str,
    voice: VoiceId,
    *,
    language: Language | None = None,
    instructions: str | None = None,
    stream_format: StreamFormat = StreamFormat.AUDIO,
    workload_token: str,
    person_token: str,
    deadline_seconds: float,
    cap_bytes: int,
    traceparent: str | None = None,
) -> Generator[MarkedStreamItem]:
    """Speak `text` with `voice` as a WAV, its chunks yielded as they arrive.

    `language` is required for a voice of several languages. `instructions` name
    the tone, for a voice that takes them. `stream_format` `sse` adds each
    sentence's `SentenceMarks` to the items. `deadline_seconds` bounds the whole
    stream, `cap_bytes` the audio. `traceparent` joins the service's lines to a
    trace; it defaults to the trace running at the call. See the module for the
    last item and for what raises, which happens at the call.
    """
    return _with_length(
        _chunks(
            base_url,
            text,
            voice,
            language=language,
            instructions=instructions,
            response_format=ResponseFormat.WAV,
            workload_token=workload_token,
            person_token=person_token,
            deadline_seconds=deadline_seconds,
            cap_bytes=cap_bytes,
            traceparent=traceparent,
            stream_format=stream_format,
        )
    )


# Every input comes from the call site; only `language` and `instructions` default.
def speak(  # noqa: PLR0913 - each wire option is its own argument
    base_url: str,
    text: str,
    voice: VoiceId,
    *,
    language: Language | None = None,
    instructions: str | None = None,
    response_format: ResponseFormat,
    workload_token: str,
    person_token: str,
    deadline_seconds: float,
    cap_bytes: int,
    traceparent: str | None = None,
) -> SpeakOutcome:
    """Speak `text` with `voice` at the service on `base_url`, within the bounds.

    `language` is required for a voice of several languages. `instructions` name
    the tone, for a voice that takes them. `deadline_seconds` bounds the whole
    call, `cap_bytes` the audio. `traceparent` joins the service's lines to a
    trace; it defaults to the trace running at the call. See the module for the
    outcomes and for what raises.
    """
    chunks = _chunks(
        base_url,
        text,
        voice,
        language=language,
        instructions=instructions,
        response_format=response_format,
        workload_token=workload_token,
        person_token=person_token,
        deadline_seconds=deadline_seconds,
        cap_bytes=cap_bytes,
        traceparent=traceparent,
    )
    # Only a WAV carries a length to check; Ogg is judged by its signature.
    items = _with_length(chunks) if response_format is ResponseFormat.WAV else chunks
    audio = bytearray()
    failure: StreamFailure | None = None
    with closing(items):
        for item in items:
            if isinstance(item, bytes):
                audio += item
            elif not isinstance(item, Finished):
                failure = item
                break
    return Spoken(bytes(audio)) if failure is None else failure
