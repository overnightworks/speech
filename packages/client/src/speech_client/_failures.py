"""How a hearing task's failure maps to an outcome, and how its tasks settle."""

from __future__ import annotations

import asyncio
import contextlib

from websockets.exceptions import ConnectionClosed, InvalidHandshake

from speech_client.outcomes import Cut, Malformed, PastDeadline, Unreachable
from speech_client.realtime_wire import ServerEventError

type FailureOutcome = PastDeadline | Malformed | Cut | Unreachable


class SourceFailedError(Exception):
    """Carries an exception of the token source past the outcome mapping."""

    def __init__(self, original: Exception) -> None:
        super().__init__()
        self.original = original


# In order: a timeout is an `OSError` too. A dropped handshake is the one
# `InvalidMessage` that is `Cut`; `_wire_session` handles it first.
_FAILURE_OUTCOMES: tuple[tuple[type[Exception], FailureOutcome], ...] = (
    (TimeoutError, PastDeadline()),
    (InvalidHandshake, Malformed()),
    (ConnectionClosed, Cut()),
    (ServerEventError, Malformed()),
    (OSError, Unreachable()),
)
FAILURES = tuple(kind for kind, _ in _FAILURE_OUTCOMES)


def failure_outcome(failure: BaseException) -> FailureOutcome:
    return next(
        outcome for kind, outcome in _FAILURE_OUTCOMES if isinstance(failure, kind)
    )


def left_under_cancellation(exception: BaseException | None) -> bool:
    # A cancel the caller's own code turned into another exception still counts.
    task = asyncio.current_task()
    return isinstance(exception, asyncio.CancelledError) or (
        task is not None and task.cancelling() > 0
    )


async def _wait_until_done(*tasks: asyncio.Task[object]) -> None:
    # A loop, not a recursion: each further cancel would otherwise deepen the stack.
    while not all(task.done() for task in tasks):
        for task in tasks:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait(tasks)


def _raise_unexpected_failure(tasks: tuple[asyncio.Task[object], ...]) -> None:
    for task in tasks:
        failure = None if task.cancelled() else task.exception()
        if failure is not None and not isinstance(failure, FAILURES):
            raise failure


async def settled(*tasks: asyncio.Task[object]) -> None:
    """Cancel `tasks` and wait until each is done, retrieving every result.

    A failure outside the outcome mapping is raised, even over a cancel that
    landed meanwhile, as in a task group; otherwise that cancel is raised again.
    The cancel count is left as it is, for `asyncio.timeout` and the caller.
    """
    for task in tasks:
        task.cancel()
    try:
        await asyncio.wait(tasks)
    except asyncio.CancelledError:
        await _wait_until_done(*tasks)
        _raise_unexpected_failure(tasks)
        raise
    _raise_unexpected_failure(tasks)
