"""The health fetch: one blocking `GET /health` (WIRE-122), bounded in time and size.

`fetch_health` takes every input from its call site and returns one
`HealthOutcome`, a closed union a caller matches with `assert_never`:

- `Health`: a 200 whose body validates as `wire.Health`; unknown keys are
  ignored (WIRE-127). The content type is not checked, the body decides.
- `UnexpectedStatus`: any other status.
- `Malformed`: a 200 whose body is not a valid `Health`.
- `OverCap`: the body would pass the byte cap; reading stops there.
- `Cut`, `PastDeadline` and `Unreachable`, as for `speak`
  (`speech_client.outcomes`).

A timeout is `PastDeadline` before anything else; then a closed or reset
connection is `Cut`; only then a parse failure is `Malformed`.

A defect of the caller raises and is no outcome: a deadline or cap of 0 or
below, and a base URL that is not `http`, has no host or carries a path, query or
fragment (`ValueError`).

The call sends no token, blocks and holds no process-global state, so it runs
in a worker thread (`asyncio.to_thread`); the thread ends by the deadline. It
logs nothing. The name lookup is bounded by the system resolver, not by the
deadline.
"""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

from pydantic import ValidationError

from speech_client._transport import checked_origin, exchange, read_within
from speech_client.outcomes import (
    Cut,
    Malformed,
    OverCap,
    PastDeadline,
    Unreachable,
)
from speech_client.wire import HEALTH_PATH, Health

if TYPE_CHECKING:
    from http.client import HTTPConnection


@dataclass(frozen=True, slots=True)
class UnexpectedStatus:
    """The service answered a status other than 200."""

    status: int


type HealthOutcome = (
    Health | UnexpectedStatus | Malformed | OverCap | PastDeadline | Cut | Unreachable
)


def _answer(connection: HTTPConnection, cap_bytes: int) -> HealthOutcome:
    connection.request("GET", HEALTH_PATH)
    response = connection.getresponse()
    if response.status != HTTPStatus.OK:
        return UnexpectedStatus(response.status)
    body = read_within(response, cap_bytes)
    if body is None:
        return OverCap()
    try:
        return Health.model_validate_json(body)
    except ValidationError:
        return Malformed()


def fetch_health(
    base_url: str, *, deadline_seconds: float, cap_bytes: int
) -> HealthOutcome:
    """Read the health of the service on `base_url`, within the bounds.

    `deadline_seconds` bounds the whole call, `cap_bytes` the body. See the module
    for the outcomes and for what raises.
    """
    origin = checked_origin(base_url, deadline_seconds, cap_bytes)
    return exchange(
        origin, deadline_seconds, lambda connection: _answer(connection, cap_bytes)
    )
