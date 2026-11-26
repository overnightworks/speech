"""The `traceparent` header that joins the service's lines to a trace.

A speaking call sends only the caller's explicit value; the running trace is the
default of the hearing's websocket handshake alone.
"""

from __future__ import annotations

from logline import current_traceparent

from speech_client.wire import TRACEPARENT_HEADER


def given_traceparent_headers(traceparent: str | None) -> dict[str, str]:
    """The caller's `traceparent` as a header, else none."""
    return {} if traceparent is None else {TRACEPARENT_HEADER: traceparent}


def traceparent_headers(traceparent: str | None) -> dict[str, str]:
    """The caller's `traceparent` as a header, else the running trace's, else none."""
    return given_traceparent_headers(
        current_traceparent() if traceparent is None else traceparent
    )
