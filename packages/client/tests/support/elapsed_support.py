"""How far below a long wait a test's elapsed time must stay to rule that wait out."""

from __future__ import annotations

# The websockets close timeout a peer that ignores the close handshake costs.
CLOSE_TIMEOUT_SECONDS = 10

# The hearing deadline a test hands a session it expects to end long before.
SESSION_DEADLINE_SECONDS = 30


def well_before(long_wait_seconds: float) -> float:
    """An upper bound for a run that must not sit through `long_wait_seconds`."""
    return long_wait_seconds / 2
