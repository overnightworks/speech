"""Send one request from inside the proof's caller container; print the reply as JSON.

The request arrives on standard input: a JSON object with `url`, `method`,
`headers` and `body` (base64, empty for none). A refused or failed connection is
a reply with an `error`.
"""

# ruff: noqa: T201 - this script's output is the reply the proof reads

import base64
import contextlib
import http.client
import json
import sys
import urllib.parse

TIMEOUT_SECONDS = 120


def reply(
    specification: dict[str, str], timeout_seconds: float = TIMEOUT_SECONDS
) -> dict[str, object]:
    """The status, headers and body answered, or the error that stopped the request."""
    url = urllib.parse.urlsplit(specification["url"])
    target = urllib.parse.urlunsplit(("", "", url.path or "/", url.query, ""))
    connection = http.client.HTTPConnection(
        url.hostname, url.port, timeout=timeout_seconds
    )
    try:
        connection.connect()
        # A service may refuse by the headers and drop the rest of the upload; its
        # answer is already on the way.
        with contextlib.suppress(OSError):
            connection.request(
                specification["method"],
                target,
                body=base64.b64decode(specification["body"]) or None,
                headers=specification["headers"],
            )
        response = connection.getresponse()
        return {
            "status": response.status,
            "headers": dict(response.getheaders()),
            "body": base64.b64encode(response.read()).decode("ascii"),
        }
    except (OSError, http.client.HTTPException) as error:
        return {"error": type(error).__name__}
    finally:
        connection.close()


if __name__ == "__main__":
    print(json.dumps(reply(json.load(sys.stdin))))
