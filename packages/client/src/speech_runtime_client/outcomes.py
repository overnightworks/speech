"""The outcomes the bounded calls share: what went wrong on the wire or the clock.

`speak`, `fetch_health` and the hearing session each return their own success
and refusal records and these, so a caller matches one closed union per call:

- `Cut`: the connection was up and the answer ended, closed or was reset
  before it was whole.
- `PastDeadline`: the total deadline passed, at connect, send or any read.
- `OverCap`: the body would pass the byte cap; reading stops there.
- `Malformed`: the answer breaks the wire.
- `Unreachable`: the name did not resolve or no address accepted the connection.

`Busy` (the 429 and the wait the service asks for; the app decides to retry)
is shared by `speak` and the hearing session only.

A timeout is `PastDeadline` before anything else; then a closed or reset
connection is `Cut`; only then a parse failure is `Malformed`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from speech_runtime_client.wire import BusyWait


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
    """The connection never came up."""


@dataclass(frozen=True, slots=True)
class Busy:
    """The service is busy and asks for a wait."""

    wait: BusyWait
