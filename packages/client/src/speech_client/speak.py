"""The speaking call: one blocking `POST /v1/audio/speech`, bounded in time and size.

`speak` takes every input from its call site and returns one `SpeakOutcome`,
a closed union a caller matches with `assert_never`:

- `Spoken`: the audio bytes exactly as received, in the ordered format, read to
  the end of the chunked body (WIRE-90).
- `Refused`: a refusal of the wire, with its status, code and param.
- `Busy`: the 429 and the wait the service asks for; the app decides to retry.
- `Malformed`: the answer breaks the wire, for example a 200 that is not
  chunked, is not the ordered format's media type or does not begin with its
  container signature (WIRE-88, WIRE-133), a refusal outside the error form, a 429
  without a whole-second `Retry-After`, or a refusal body over 64 KiB.
- `Cut`, `PastDeadline`, `OverCap` and `Unreachable`, shared with the health
  fetch (`speech_client.outcomes`); here `OverCap` is the audio.

A defect of the caller raises and is no outcome: an input the request refuses
(`pydantic.ValidationError`), a deadline or cap of 0 or below, and a base URL
that is not `http`, has no host or carries a path, query or fragment
(`ValueError`). The service
is plain HTTP on internal networks.

The call blocks and holds no process-global state, so it runs in a worker
thread (`asyncio.to_thread`); the thread ends by the deadline. It logs nothing,
and no outcome carries the text or a token. The name lookup is bounded by the
system resolver, not by the deadline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING, Final

from pydantic import ValidationError

from speech_client._transport import checked_origin, exchange, read_within
from speech_client.outcomes import (
    Busy,
    Cut,
    Malformed,
    OverCap,
    PastDeadline,
    Unreachable,
)
from speech_client.wire import (
    SPEECH_PATH,
    Refusal,
    RefusalCode,
    RequestField,
    ResponseFormat,
    RetryAfterError,
    VoiceId,
    busy_wait,
    refusal,
    speech_request_body,
    speech_request_headers,
)

if TYPE_CHECKING:
    from http.client import HTTPConnection, HTTPResponse

__all__ = [
    "Busy",
    "Cut",
    "Malformed",
    "OverCap",
    "PastDeadline",
    "Refused",
    "SpeakOutcome",
    "Spoken",
    "Unreachable",
    "speak",
]

_REFUSAL_BODY_LIMIT: Final = 64 * 1024


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


type SpeakOutcome = (
    Spoken | Refused | Busy | Cut | PastDeadline | OverCap | Malformed | Unreachable
)


def _spoken(
    response: HTTPResponse, cap_bytes: int, response_format: ResponseFormat
) -> SpeakOutcome:
    if (
        response.headers.get_content_type() != response_format.media_type
        or not response.chunked
    ):
        return Malformed()
    audio = read_within(response, cap_bytes)
    if audio is None:
        return OverCap()
    if not response_format.begins_like_container(audio):
        return Malformed()
    return Spoken(audio)


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
    cap_bytes: int,
    response_format: ResponseFormat,
) -> SpeakOutcome:
    connection.request("POST", SPEECH_PATH, body, headers)
    response = connection.getresponse()
    if response.status == HTTPStatus.OK:
        return _spoken(response, cap_bytes, response_format)
    return _refusal(response)


# Every input comes from the call site, none has a default.
def speak(  # noqa: PLR0913
    base_url: str,
    text: str,
    voice: VoiceId,
    *,
    response_format: ResponseFormat,
    workload_token: str,
    person_token: str,
    deadline_seconds: float,
    cap_bytes: int,
) -> SpeakOutcome:
    """Speak `text` with `voice` at the service on `base_url`, within the bounds.

    `deadline_seconds` bounds the whole call, `cap_bytes` the audio. See the module
    for the outcomes and for what raises.
    """
    origin = checked_origin(base_url, deadline_seconds, cap_bytes)
    body = speech_request_body(text, voice, response_format)
    headers = speech_request_headers(
        workload_token=workload_token, person_token=person_token
    )
    return exchange(
        origin,
        deadline_seconds,
        lambda connection: _answer(
            connection, body, headers, cap_bytes, response_format
        ),
    )
