"""One voice's engine lifecycle and its admission gate."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from speech_runtime_service.admission_gate import AdmissionGate

if TYPE_CHECKING:
    from collections.abc import Callable

    from speech_runtime_service.clock import LoopClock
    from speech_runtime_service.service import RuntimeDependencies, SpeakingEngine

logger = logging.getLogger(__name__)


class VoiceSlot:
    """The engine one voice speaks with, from its build to its close."""

    def __init__(
        self,
        dependencies: RuntimeDependencies,
        clock: LoopClock,
        speaking: SpeakingEngine | None = None,
    ) -> None:
        """Hold `speaking` if it is ready; otherwise nothing runs until `load`."""
        self.speaking = speaking
        self.gate = AdmissionGate(clock)
        self.loading = False
        self._dependencies = dependencies
        self._state_lock = threading.Lock()
        self._state_changed = threading.Condition(self._state_lock)
        self._ready_speaking = speaking if speaking and speaking.ready else None
        self._pending_engine: SpeakingEngine | None = None
        self._closing_engine: SpeakingEngine | None = None
        self._stopping = False
        self._fatal_callback: Callable[[], None] = lambda: None

    def begin_loading(self) -> None:
        """Publish startup before its background task can be scheduled."""
        with self._state_lock:
            self.loading = True

    def load(self, after_publish: Callable[[], None]) -> None:
        """Build and load the engine, publish it, then run `after_publish`."""
        engine: SpeakingEngine | None = None
        registered = False
        try:
            engine = self._construct_engine()
            registered = self._register_pending_engine(engine)
            if registered:
                _load_or_raise(engine)
                self._publish(engine)
                after_publish()
        finally:
            self._finish_loading(engine, registered=registered)

    def capture(self) -> SpeakingEngine | None:
        """Capture one stable engine for an entire streaming response.

        It takes only the state lock, which every holder keeps briefly, so a
        caller on the event loop is never held up by a failed engine's close;
        `reconcile_failed_active` does that, after the refusal.
        """
        with self._state_lock:
            speaking = self.speaking
            if self.loading or self._stopping or speaking is None or not speaking.ready:
                return None
            return speaking

    def begin_shutdown(self) -> None:
        """Refuse new work before the server starts its own shutdown."""
        with self._state_changed:
            self._stopping = True
            self._state_changed.notify_all()

    def set_fatal_callback(self, callback: Callable[[], None]) -> None:
        """Route an owned engine failure to the service orchestrator."""
        with self._state_lock:
            self._fatal_callback = callback
            engines = [self._pending_engine, self.speaking]
        for engine in _unique_engines(engines):
            setter = getattr(engine, "set_fatal_callback", None)
            if callable(setter):
                setter(callback)

    def close(self, while_released: Callable[[], None]) -> None:
        """Close the pending and the active engine outside the state lock.

        `while_released` runs once the slot refuses work and before its engines close.
        """
        with self._state_changed:
            self._stopping = True
            self._state_changed.notify_all()
            while self._closing_engine is not None:
                self._state_changed.wait()
            engines = [self._pending_engine, self.speaking]
            self._pending_engine = None
            self.speaking = None
            self._ready_speaking = None
        while_released()
        failed = False
        for engine in _unique_engines(engines):
            try:
                engine.close()
            except Exception:
                failed = True
                logger.exception("speaking engine cleanup failed")
        if failed:
            self._fatal_callback()
            message = "speaking engine cleanup failed"
            raise RuntimeError(message)

    def reconcile_failed_active(self) -> None:
        """Close an active engine that stopped being ready."""
        with self._state_lock:
            active = self.speaking
            if active is None:
                return
            if active.ready:
                self._ready_speaking = active
                return
            if active is not self._ready_speaking or self.loading or self._stopping:
                return
            self.speaking = None
            self._ready_speaking = None
            self._closing_engine = active
        try:
            active.close()
        except Exception:
            logger.exception("failed speaking engine cleanup failed")
            self._fatal_callback()
        finally:
            with self._state_changed:
                self._closing_engine = None
                self._state_changed.notify_all()

    def health(self) -> dict[str, object]:
        """The speaking part of the contract body for GET /health."""
        with self._state_lock:
            speaking = self.speaking
        if speaking is None:
            return {
                "model": "unavailable",
                "ready": False,
                "streams": False,
                "sample_rate": 0,
            }
        return {
            "model": speaking.model_name,
            "ready": speaking.ready,
            "streams": speaking.streams,
            "sample_rate": speaking.sample_rate,
        }

    def _construct_engine(self) -> SpeakingEngine:
        voice = self._dependencies.voice
        engine_factory = self._dependencies.engine_factory
        artifact_checker = self._dependencies.artifact_checker
        if engine_factory is None or artifact_checker is None:
            message = "the runtime has no engine factory"
            raise RuntimeError(message)
        if not artifact_checker(voice):
            message = f"voice {voice.value} is not installed"
            raise RuntimeError(message)
        engine = engine_factory(voice)
        setter = getattr(engine, "set_fatal_callback", None)
        if callable(setter):
            setter(self._fatal_callback)
        return engine

    def _register_pending_engine(self, engine: SpeakingEngine) -> bool:
        with self._state_lock:
            if self._stopping:
                return False
            self._pending_engine = engine
            return True

    def _publish(self, engine: SpeakingEngine) -> None:
        with self._state_lock:
            if self._stopping:
                return
            self.speaking = engine
            self._ready_speaking = engine if engine.ready else None
            self._pending_engine = None

    def _finish_loading(
        self, engine: SpeakingEngine | None, *, registered: bool
    ) -> None:
        with self._state_lock:
            self.loading = False
            still_pending = registered and self._pending_engine is engine
            if still_pending:
                self._pending_engine = None
        if engine is not None and (still_pending or not registered):
            try:
                engine.close()
            except Exception:
                logger.exception("uncommitted speaking engine cleanup failed")
                self._fatal_callback()


def _unique_engines(
    engines: list[SpeakingEngine | None],
) -> list[SpeakingEngine]:
    unique: list[SpeakingEngine] = []
    identities: set[int] = set()
    for engine in engines:
        if engine is not None and id(engine) not in identities:
            identities.add(id(engine))
            unique.append(engine)
    return unique


def _load_or_raise(engine: SpeakingEngine) -> None:
    try:
        engine.load()
    except Exception:
        message = f"speaking model {engine.model_name} failed to load"
        logger.exception("%s", message)
        raise RuntimeError(message) from None
