"""The outcomes the bounded calls share: what went wrong on the wire or the clock.

`speak`, `fetch_health` and the hearing session each return their own success
and refusal records and these, so a caller matches one closed union per call:

- `Cut`: the connection was up and the answer ended, closed or was reset
  before it was whole.
- `PastDeadline`: the total deadline passed, at connect, send or any read.
- `OverCap`: the body would pass the byte cap; reading stops there.
- `Malformed`: the answer breaks the wire.
- `Unreachable`: the name did not resolve, no address accepted the connection, or
  the service did not open the session within the opening bound.

`Busy` (the 429 and the wait the service asks for; the app decides to retry)
is shared by `speak` and the hearing session only; `busy_wait` reads it
from a response.

A timeout is `PastDeadline` before anything else, except the hearing session's
opening bound, which is `Unreachable`; then a closed or reset
connection is `Cut`; only then a parse failure is `Malformed`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from speech_client.wire import REFUSAL_STATUS, RETRY_AFTER_HEADER, RefusalCode

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class Cut:
    """The answer ended, closed or was reset before it was whole."""


@dataclass(frozen=True, slots=True)
class PastDeadline:
    """The total deadline passed."""


@dataclass(frozen=True, slots=True)
class OverCap:
    """The body would pass the byte cap."""


@dataclass(frozen=True, slots=True)
class Malformed:
    """The answer breaks the wire."""


@dataclass(frozen=True, slots=True)
class Unreachable:
    """The connection never came up, or the service never opened the session."""


@dataclass(frozen=True, slots=True)
class Busy:
    """The service is busy and asks for a wait."""

    wait: BusyWait


class RetryAfterError(ValueError):
    """A `busy` answer whose `Retry-After` is not a whole number of seconds."""


@dataclass(frozen=True)
class BusyWait:
    """How long the service asks the app to wait after a `busy` refusal."""

    retry_after_seconds: int


def busy_wait(status: int, headers: Mapping[str, str]) -> BusyWait | None:
    """The wait a `busy` answer asks for, or `None` for any other status.

    Raises `RetryAfterError` when a `busy` answer lacks a `Retry-After` of whole
    seconds, at least 1: the service never sends a date.
    """
    if status != REFUSAL_STATUS[RefusalCode.BUSY]:
        return None
    wanted = RETRY_AFTER_HEADER.lower()
    value = next((v for name, v in headers.items() if name.lower() == wanted), None)
    if value is None or not (value.isascii() and value.isdigit()) or int(value) < 1:
        message = (
            f"{RETRY_AFTER_HEADER} must be whole seconds, at least 1, got {value!r}"
        )
        raise RetryAfterError(message)
    return BusyWait(retry_after_seconds=int(value))
