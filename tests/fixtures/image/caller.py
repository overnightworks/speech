"""Send one request from inside the proof's caller container; print the reply as JSON.

The request arrives on standard input: a JSON object with `url`, `method`,
`headers` and `body` (base64, empty for none). A refused or failed connection is
a reply with an `error`.
"""

# ruff: noqa: T201 - this script's output is the reply the proof reads

import base64
import json
import sys
import time
import urllib.error
import urllib.request

TIMEOUT_SECONDS = 120


def reply(request: urllib.request.Request) -> dict[str, object]:
    """The status, headers, body and timing answered, or the error that stopped it.

    The timing is the seconds from the request's start to the body's first byte
    and to its end.
    """
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310 - the proof's own http URL
            status, headers, first = response.status, response.headers, response.read(1)
            first_byte = time.perf_counter()
            body = first + response.read()
    except urllib.error.HTTPError as refusal:
        status, headers, body = refusal.code, refusal.headers, refusal.read()
        first_byte = time.perf_counter()
    except OSError as error:
        # urlopen wraps a refused connection in a URLError that holds the cause.
        return {"error": type(getattr(error, "reason", error)).__name__}
    return {
        "status": status,
        "headers": dict(headers.items()),
        "body": base64.b64encode(body).decode("ascii"),
        "first_byte_seconds": first_byte - started,
        "seconds": time.perf_counter() - started,
    }


specification = json.load(sys.stdin)
print(
    json.dumps(
        reply(
            urllib.request.Request(  # noqa: S310 - the proof's own http URL
                specification["url"],
                data=base64.b64decode(specification["body"]) or None,
                headers=specification["headers"],
                method=specification["method"],
            )
        )
    )
)
