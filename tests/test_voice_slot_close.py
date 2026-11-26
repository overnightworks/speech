"""A slot's close and its shared load order themselves around an engine's slow close."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

from shipped_catalogue import CATALOGUE
from speech_client.wire import VoiceId
from speech_service.clock import LoopClock
from speech_service.service import RuntimeDependencies
from speech_service.voice_slot import VoiceSlot
from tests.support.fakes_support import (
    GATE_TIMEOUT_SECONDS,
    GatedEngine,
    noting_slot_waits,
)

if TYPE_CHECKING:
    import threading
    from collections.abc import Iterator

    from speech_service.service import SpeakingEngine


class _Slot:
    """A slot whose factory hands out `engines` in order and records its builds."""

    def __init__(
        self, *engines: GatedEngine, speaking: GatedEngine | None = None
    ) -> None:
        self.built: list[SpeakingEngine] = []
        self.fatal_calls = 0
        queue = iter(engines)

        def build(_voice: VoiceId) -> SpeakingEngine:
            engine = next(queue)
            self.built.append(engine)
            return engine

        self.slot = VoiceSlot(
            CATALOGUE[VoiceId.THORSTEN],
            RuntimeDependencies(
                catalogue=CATALOGUE,
                engine_factory=build,
                missing_artifacts=lambda _voice: (),
            ),
            LoopClock(),
            speaking,
        )
        self.slot.set_fatal_callback(self._fatal)

    def _fatal(self) -> None:
        self.fatal_calls += 1


@pytest.fixture
def pool() -> Iterator[ThreadPoolExecutor]:
    with ThreadPoolExecutor(max_workers=3) as executor:
        yield executor


@pytest.fixture
def waited(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    return noting_slot_waits(monkeypatch)


def _slot_whose_engine_failed(
    *fresh: GatedEngine, close_fails: bool, hold_close: bool = False
) -> tuple[_Slot, GatedEngine]:
    failed = GatedEngine(close_fails=close_fails, hold_close=hold_close)
    failed.ready = True
    owner = _Slot(*fresh, speaking=failed)
    failed.ready = False
    return owner, failed


@pytest.mark.parametrize("close_fails", [False, True], ids=["closed", "close-failed"])
def test_a_close_waits_for_the_close_of_a_failed_engine_and_raises_when_it_failed(
    pool: ThreadPoolExecutor, waited: threading.Event, *, close_fails: bool
) -> None:
    owner, failed = _slot_whose_engine_failed(close_fails=close_fails, hold_close=True)
    reconciling = pool.submit(owner.slot.reconcile_failed_active)
    assert failed.entered_close.wait(GATE_TIMEOUT_SECONDS)

    closing = pool.submit(owner.slot.close)
    assert waited.wait(GATE_TIMEOUT_SECONDS)
    assert not closing.done()
    failed.release_all()

    reconciling.result(GATE_TIMEOUT_SECONDS)
    if close_fails:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            closing.result(GATE_TIMEOUT_SECONDS)
        assert owner.fatal_calls == 1
    else:
        closing.result(GATE_TIMEOUT_SECONDS)
        assert owner.fatal_calls == 0


@pytest.mark.parametrize("close_fails", [False, True], ids=["closed", "close-failed"])
def test_a_close_waits_for_a_running_load_and_raises_when_cleanup_failed(
    pool: ThreadPoolExecutor, waited: threading.Event, *, close_fails: bool
) -> None:
    building = GatedEngine(hold_load=True, close_fails=close_fails)
    owner = _Slot(building)
    load = owner.slot.shared_load()
    assert load is not None
    assert building.entered_load.wait(GATE_TIMEOUT_SECONDS)

    closing = pool.submit(owner.slot.close)
    assert waited.wait(GATE_TIMEOUT_SECONDS)
    assert not closing.done()
    building.release_load.set()

    if close_fails:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            closing.result(GATE_TIMEOUT_SECONDS)
    else:
        closing.result(GATE_TIMEOUT_SECONDS)
    assert load.result(GATE_TIMEOUT_SECONDS) is None


def test_a_dead_engine_whose_close_fails_is_fatal_and_no_engine_is_built() -> None:
    dead = GatedEngine(close_fails=True)
    owner = _Slot(speaking=dead)

    load = owner.slot.shared_load()

    assert load is not None
    assert load.result(GATE_TIMEOUT_SECONDS) is None
    assert owner.fatal_calls == 1
    assert owner.built == []
    assert owner.slot.capture() is None


def test_a_shared_load_builds_no_engine_until_the_failed_engines_close_ends(
    pool: ThreadPoolExecutor, waited: threading.Event
) -> None:
    fresh = GatedEngine()
    owner, failed = _slot_whose_engine_failed(fresh, close_fails=False, hold_close=True)
    reconciling = pool.submit(owner.slot.reconcile_failed_active)
    assert failed.entered_close.wait(GATE_TIMEOUT_SECONDS)

    load = owner.slot.shared_load()
    assert load is not None
    assert waited.wait(GATE_TIMEOUT_SECONDS)
    assert owner.built == []
    failed.release_close.set()

    reconciling.result(GATE_TIMEOUT_SECONDS)
    assert load.result(GATE_TIMEOUT_SECONDS) is fresh
    assert owner.built == [fresh]


def test_a_load_asked_after_an_engine_was_published_returns_it_and_builds_none() -> (
    None
):
    published = GatedEngine()
    owner = _Slot(published)
    first = owner.slot.shared_load()
    assert first is not None
    assert first.result(GATE_TIMEOUT_SECONDS) is published

    again = owner.slot.shared_load()

    assert again is not None
    assert again.result(GATE_TIMEOUT_SECONDS) is published
    assert owner.built == [published]
