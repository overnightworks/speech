"""The `traceparent` header a call sends, which joins the service's lines to a trace."""

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
