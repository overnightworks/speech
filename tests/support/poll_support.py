"""The bounded poll every test uses to wait for another process, the OS or a file."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

POLL_INTERVAL_SECONDS = 0.01


def poll(
    condition: Callable[[], bool],
    *,
    within: float,
    interval: float = POLL_INTERVAL_SECONDS,
) -> bool:
    """Whether `condition` held, checked now and every `interval` s for `within` s."""
    deadline = time.monotonic() + within
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)
    return True


def wait_until(
    condition: Callable[[], bool],
    *,
    within: float,
    waited_for: str,
    interval: float = POLL_INTERVAL_SECONDS,
) -> None:
    """Return once `condition` holds; fail naming `waited_for` after `within` s."""
    assert poll(condition, within=within, interval=interval), waited_for


async def poll_async(
    condition: Callable[[], bool],
    *,
    within: float,
    interval: float = POLL_INTERVAL_SECONDS,
) -> bool:
    """`poll` for a coroutine: `condition` runs in a thread, the loop stays free."""
    deadline = time.monotonic() + within
    while not await asyncio.to_thread(condition):
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(interval)
    return True


async def wait_until_async(
    condition: Callable[[], bool],
    *,
    within: float,
    waited_for: str,
    interval: float = POLL_INTERVAL_SECONDS,
) -> None:
    """`wait_until` for a coroutine."""
    assert await poll_async(condition, within=within, interval=interval), waited_for
