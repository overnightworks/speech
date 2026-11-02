"""Send one request from inside the proof's caller container; print the reply as JSON.

The request arrives as the one argument: a JSON object with `url`, `method`,
`headers` and `body` (base64, empty for none). A refused or failed connection is
a reply with an `error`.
"""

# ruff: noqa: T201 - this script's output is the reply the proof reads

import base64
import json
import sys
import urllib.error
import urllib.request

TIMEOUT_SECONDS = 120


def reply(request: urllib.request.Request) -> dict[str, object]:
    """The status and body the server answered, or the error that stopped it."""
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310 - the proof's own http URL
            status, body = response.status, response.read()
    except urllib.error.HTTPError as refusal:
        status, body = refusal.code, refusal.read()
    except OSError as error:
        return {"error": type(error).__name__}
    return {"status": status, "body": base64.b64encode(body).decode("ascii")}


specification = json.loads(sys.argv[1])
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
