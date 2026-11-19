"""The hand-written HTTP/1.1 framing the raw listener tests answer with."""

from __future__ import annotations


def chunk(data: bytes) -> bytes:
    """One chunk of a chunked body."""
    return b"%x\r\n%b\r\n" % (len(data), data)


def head(status: str, *headers: str) -> bytes:
    """The status line and `headers` of a response, closed by the empty line."""
    return "\r\n".join([f"HTTP/1.1 {status}", *headers, "", ""]).encode()
