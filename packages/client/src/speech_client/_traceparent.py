"""The `traceparent` header the caller gives, which joins the service to a trace.

Only the caller's explicit value is sent from here; the running trace reaches the
service through `logline`'s hooks once the host has called `logline.configure`.
"""

from __future__ import annotations

from speech_client.wire import TRACEPARENT_HEADER


def given_traceparent_headers(traceparent: str | None) -> dict[str, str]:
    """The caller's `traceparent` as a header, else none."""
    return {} if traceparent is None else {TRACEPARENT_HEADER: traceparent}
