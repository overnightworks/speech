"""The hearing proof's wait for the service to finish loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from shipped_catalogue import PIPER_VOICES
from speech_client.outcomes import Unreachable
from tests.support.clock_support import FakeClock
from tests.support.hearing_health_support import (
    HEALTH_LOADING,
    HEALTH_NO_HEARING,
    HEALTH_READY,
)
from tests.support.script_support import script_module
from tests.support.tcp_server_support import free_loopback_port

if TYPE_CHECKING:
    from proof_minting import Minting
    from speech_client.wire import Health

HEARING = script_module("prove_hearing")


def wait_loaded_with(health: Health, *, speaks: bool) -> Health:
    clock = FakeClock()
    return HEARING.wait_loaded(
        "http://127.0.0.1:1",
        lambda: None,
        HEARING.LoadWait(lambda: speaks, lambda *_: health, clock.jump, clock.now),
    )


@pytest.mark.parametrize(
    ("health", "speaks"),
    [(HEALTH_READY, False), (HEALTH_NO_HEARING, True)],
    ids=["hearing ready", "speaking served after a load without hearing"],
)
def test_the_service_has_loaded_when_hearing_is_ready_or_speaking_is_served(
    health: Health, *, speaks: bool
) -> None:
    assert wait_loaded_with(health, speaks=speaks) == health


@pytest.mark.parametrize(
    ("health", "speaks"),
    [(HEALTH_LOADING, True), (HEALTH_NO_HEARING, False)],
    ids=["still loading", "speaking ready but not served"],
)
def test_a_service_that_never_finishes_loading_fails_the_wait(
    health: Health, *, speaks: bool
) -> None:
    with pytest.raises(SystemExit, match="did not finish loading"):
        wait_loaded_with(health, speaks=speaks)


def test_a_load_whose_fresh_health_fails_waits_on() -> None:
    clock = FakeClock()
    answers = iter([HEALTH_READY, Unreachable(), HEALTH_READY, HEALTH_READY])
    wait = HEARING.LoadWait(
        lambda: False, lambda *_: next(answers), clock.jump, clock.now
    )

    health = HEARING.wait_loaded("http://127.0.0.1:1", lambda: None, wait)

    assert (health, clock.now()) == (HEALTH_READY, HEARING.POLL_SECONDS)


def test_a_service_nothing_listens_on_does_not_speak(minting: Minting) -> None:
    base_url = f"http://127.0.0.1:{free_loopback_port()}"

    assert not HEARING.speaks(base_url, minting, PIPER_VOICES[0])


def test_a_wait_with_slow_health_answers_still_ends_at_its_time_limit() -> None:
    clock = FakeClock()
    slow_seconds = 35.0

    def slow_health(*_: object) -> Health:
        clock.jump(slow_seconds)
        return HEALTH_LOADING

    wait = HEARING.LoadWait(lambda: True, slow_health, clock.jump, clock.now)
    with pytest.raises(SystemExit, match="did not finish loading"):
        HEARING.wait_loaded("http://127.0.0.1:1", lambda: None, wait)

    assert clock.now() < HEARING.READY_TIMEOUT_SECONDS + slow_seconds + 1


def test_a_wait_that_never_heard_the_service_names_no_health() -> None:
    clock = FakeClock()

    def unreachable(*_: object) -> Unreachable:
        return Unreachable()

    wait = HEARING.LoadWait(lambda: True, unreachable, clock.jump, clock.now)
    with pytest.raises(SystemExit) as failure:
        HEARING.wait_loaded("http://127.0.0.1:1", lambda: None, wait)

    assert "None" not in str(failure.value)


def test_a_wait_ends_at_once_with_the_last_health_when_the_server_exited() -> None:
    clock = FakeClock()
    polls = iter([None, None, 1])

    def exit_status() -> int | None:
        return next(polls)

    wait = HEARING.LoadWait(
        lambda: True, lambda *_: HEALTH_LOADING, clock.jump, clock.now
    )
    with pytest.raises(SystemExit, match="exited with status 1") as failure:
        HEARING.wait_loaded("http://127.0.0.1:1", exit_status, wait)

    assert str(HEALTH_LOADING) in str(failure.value)
    assert clock.now() < HEARING.READY_TIMEOUT_SECONDS
