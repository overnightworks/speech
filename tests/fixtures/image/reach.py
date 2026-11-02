"""Try a TCP connection to each `host:port` argument; print who answered, as JSON."""

# ruff: noqa: T201 - this script's output is the answer the proof reads

import json
import socket
import sys

TIMEOUT_SECONDS = 3


def reachable(address: str) -> bool:
    """Whether a TCP connection to `host:port` opens."""
    host, _, port = address.rpartition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=TIMEOUT_SECONDS):
            return True
    except OSError:
        return False


print(json.dumps({address: reachable(address) for address in sys.argv[1:]}))
