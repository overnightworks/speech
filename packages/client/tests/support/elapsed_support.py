"""How far below a long wait a test's elapsed time must stay to rule that wait out."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from contextlib import AbstractAsyncContextManager

# The websockets close timeout a peer that ignores the close handshake costs.
CLOSE_TIMEOUT_SECONDS = 10

# The hearing deadline a test hands a session it expects to end long before.
SESSION_DEADLINE_SECONDS = 30


def well_before(long_wait_seconds: float) -> float:
    """An upper bound for a run that must not sit through `long_wait_seconds`."""
    return long_wait_seconds / 2


async def seconds_to_cancel_after_yields(
    task: asyncio.Task[object], yields: int
) -> float:
    """Yield `yields` times, cancel `task`; the seconds it took to end cancelled."""
    for _ in range(yields):
        await asyncio.sleep(0)
    started = time.monotonic()
    task.cancel()
    await asyncio.wait({task}, timeout=CLOSE_TIMEOUT_SECONDS)
    elapsed = time.monotonic() - started
    assert task.cancelled()
    return elapsed


async def seconds_to_cancel_after_end[Target](
    stalled_service: AbstractAsyncContextManager[tuple[Target, asyncio.Event]],
    call_to: Callable[[Target, asyncio.Event], Awaitable[object]],
    yields: int,
) -> float:
    """Cancel a call `yields` yields after it was told to end; the seconds it took.

    The service opens the call, then stalls; the call ends once its event is set.
    """
    async with stalled_service as (target, opened):
        ended = asyncio.Event()
        call = asyncio.ensure_future(call_to(target, ended))
        await opened.wait()
        ended.set()
        elapsed = await seconds_to_cancel_after_yields(call, yields)
    assert asyncio.all_tasks() == {asyncio.current_task()}
    return elapsed
