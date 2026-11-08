"""The types of the raw listener the client tests share."""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager

type Answer = Callable[[socket.socket, threading.Event], None]
type RawListener = Callable[
    [Answer | bytes], AbstractContextManager[tuple[str, list[bytes]]]
]
