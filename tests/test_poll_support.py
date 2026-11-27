"""The shared bounded poll waits for a late condition and fails loud past its bound."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from tests.support.poll_support import poll, poll_async, wait_until, wait_until_async

if TYPE_CHECKING:
    from collections.abc import Callable

WITHIN_SECONDS = 0.05
LATE_WITHIN_SECONDS = 5.0
INTERVAL_SECONDS = 0.001
WAITED_FOR = "the thing that never came"


def _holds_on_call(number: int) -> Callable[[], bool]:
    calls = 0

    def condition() -> bool:
        nonlocal calls
        calls += 1
        return calls >= number

    return condition


def _sync_poll(condition: Callable[[], bool], within: float = WITHIN_SECONDS) -> bool:
    return poll(condition, within=within, interval=INTERVAL_SECONDS)


def _async_poll(condition: Callable[[], bool], within: float = WITHIN_SECONDS) -> bool:
    return asyncio.run(poll_async(condition, within=within, interval=INTERVAL_SECONDS))


def _sync_wait(condition: Callable[[], bool]) -> bool:
    wait_until(
        condition,
        within=WITHIN_SECONDS,
        waited_for=WAITED_FOR,
        interval=INTERVAL_SECONDS,
    )
    return True


def _async_wait(condition: Callable[[], bool]) -> bool:
    asyncio.run(
        wait_until_async(
            condition,
            within=WITHIN_SECONDS,
            waited_for=WAITED_FOR,
            interval=INTERVAL_SECONDS,
        )
    )
    return True


@pytest.mark.parametrize("wait", [_sync_poll, _async_poll], ids=["sync", "async"])
def test_poll_returns_true_when_the_condition_holds_late(
    wait: Callable[..., bool],
) -> None:
    assert wait(_holds_on_call(3), within=LATE_WITHIN_SECONDS) is True


@pytest.mark.parametrize("wait", [_sync_poll, _async_poll], ids=["sync", "async"])
def test_poll_returns_false_when_the_condition_never_holds(
    wait: Callable[[Callable[[], bool]], bool],
) -> None:
    assert wait(lambda: False) is False


@pytest.mark.parametrize("wait", [_sync_wait, _async_wait], ids=["sync", "async"])
def test_wait_until_fails_naming_what_it_waited_for(
    wait: Callable[[Callable[[], bool]], bool],
) -> None:
    with pytest.raises(AssertionError, match=WAITED_FOR):
        wait(lambda: False)
