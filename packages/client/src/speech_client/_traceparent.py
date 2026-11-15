"""The `traceparent` header a call sends, which joins the service's lines to a trace."""

from __future__ import annotations

from logline import current_traceparent

from speech_client.wire import TRACEPARENT_HEADER


def traceparent_headers(traceparent: str | None) -> dict[str, str]:
    """The caller's `traceparent` as a header, else the running trace's, else none."""
    sent = current_traceparent() if traceparent is None else traceparent
    return {} if sent is None else {TRACEPARENT_HEADER: sent}
