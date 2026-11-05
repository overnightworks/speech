"""HTTP contract: health and the runtime that speaks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from speech_runtime_client.wire import (
    HEALTH_PATH,
    Health,
    HearingHealth,
    SpeakingHealth,
)
from speech_runtime_service.card import card_memory_mb
from speech_runtime_service.clock import LoopClock
from speech_runtime_service.failures import PayloadFreeFailure, SpeakingFailedError
from speech_runtime_service.hearing import Hearing, hearing_from_settings
from speech_runtime_service.pcm import wav_header
from speech_runtime_service.speaking import speaking_for_voice
from speech_runtime_service.voice_slot import VoiceSlot
from speech_runtime_service.voices import RESIDENT_VOICE, VoiceId, missing_artifacts

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from concurrent.futures import Future

    from fastapi import FastAPI

    from speech_runtime_service.config import Settings

logger = logging.getLogger(__name__)

_CLOCK: Final = LoopClock()


class SpeakingEngine(Protocol):
    """The engine one voice speaks with."""

    model_name: str
    sample_rate: int
    ready: bool

    def load(self) -> None:
        """Load weights and become ready, or raise."""

    def pcm_chunks(self, text: str, language: str) -> Generator[bytes]:
        """16-bit mono PCM as soon as the model produces it."""

    def close(self) -> None:
        """Release owned resources once."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeDependencies:
    """The GPU voices offered beside Piper, and the seams that check and build them."""

    gpu_voices: frozenset[VoiceId] = frozenset()
    engine_factory: Callable[[VoiceId], SpeakingEngine] | None = None
    missing_artifacts: Callable[[VoiceId], tuple[str, ...]] | None = None


class Runtime:
    """The voice slots, hearing, and the one readiness both wires read."""

    def __init__(
        self,
        speaking: SpeakingEngine | None,
        *,
        memory_probe: Callable[[], int] | None = None,
        dependencies: RuntimeDependencies | None = None,
        hearing: Hearing | None = None,
    ) -> None:
        """Hold Piper's slot, holding `speaking`, and one idle slot per GPU voice.

        Nothing is loaded until `load`, and a GPU voice not until its first request.
        """
        dependencies = dependencies or RuntimeDependencies()
        self.hearing = hearing or Hearing(None, _CLOCK)
        self.clock = _CLOCK
        self._resident = VoiceSlot(RESIDENT_VOICE, dependencies, _CLOCK, speaking)
        self.slots: dict[VoiceId, VoiceSlot] = {RESIDENT_VOICE: self._resident} | {
            voice: VoiceSlot(voice, dependencies, _CLOCK)
            for voice in sorted(dependencies.gpu_voices)
        }
        self._memory_probe = memory_probe or card_memory_mb

    @classmethod
    def from_settings(cls, settings: Settings) -> Runtime:
        """Compose the offer: Piper, and the GPU voices `SPEECH_GPU_VOICES` names."""
        return cls(
            None,
            hearing=hearing_from_settings(settings, _CLOCK),
            dependencies=RuntimeDependencies(
                gpu_voices=settings.gpu_voices,
                engine_factory=lambda voice: speaking_for_voice(settings, voice),
                missing_artifacts=lambda voice: missing_artifacts(settings, voice),
            ),
        )

    @property
    def loading(self) -> bool:
        """Whether the start load is still running."""
        return self._resident.loading

    def load(self) -> None:
        """Load Piper, then hearing; a GPU voice loads on its first request."""
        self.begin_loading()
        self._resident.load(self.hearing.load)

    def begin_loading(self) -> None:
        """Publish startup before its background task can be scheduled."""
        self._resident.begin_loading()

    def capture(self, voice: VoiceId) -> SpeakingEngine | None:
        """The engine `voice` speaks with now; `None` while it cannot speak.

        This is the one readiness: no voice speaks during the start load or once
        shutdown began, and /health reports exactly what this answers.
        """
        if self._resident.loading:
            return None
        return self.slots[voice].capture()

    def load_on_demand(self, voice: VoiceId) -> Future[SpeakingEngine | None] | None:
        """The GPU voice's shared load, started if none runs; `None` if none may."""
        if voice is RESIDENT_VOICE or self._resident.loading:
            return None
        return self.slots[voice].shared_load()

    def health(self) -> Health:
        """The contract body for GET /health."""
        self.reconcile_failed_active()
        voices = {
            voice.value: slot.health(ready=self.capture(voice) is not None)
            for voice, slot in self.slots.items()
        }
        return Health(
            speaking=SpeakingHealth(
                ready=voices[RESIDENT_VOICE.value].ready, voices=voices
            ),
            hearing=HearingHealth.model_validate(self.hearing.health()),
            card_memory_mb=self._memory_probe(),
        )

    def begin_shutdown(self) -> None:
        """Refuse new work before the server starts its own shutdown."""
        for slot in self.slots.values():
            slot.begin_shutdown()

    def set_fatal_callback(self, callback: Callable[[], None]) -> None:
        """Route an owned speaking-engine failure to the service orchestrator."""
        for slot in self.slots.values():
            slot.set_fatal_callback(callback)

    def close(self) -> None:
        """Close every GPU voice and its load, then Piper and hearing."""
        self.begin_shutdown()
        failed = False
        for slot in self.slots.values():
            if slot is not self._resident:
                try:
                    slot.close()
                except RuntimeError:
                    failed = True
        self._resident.close(self.hearing.close)
        if failed:
            message = "speaking engine cleanup failed"
            raise RuntimeError(message)

    def reconcile_failed_active(self) -> None:
        """Close an active engine that stopped being ready."""
        for slot in self.slots.values():
            slot.reconcile_failed_active()


def wav_chunks(
    speaking: SpeakingEngine,
    text: str,
    language: str,
    sample_rate: int | None = None,
) -> Iterator[bytes]:
    """One WAV from `speaking`: its header leaves with the first PCM chunk."""
    first = True
    with PayloadFreeFailure(logger, "speaking failed", reraise_as=SpeakingFailedError):
        synthesis = speaking.pcm_chunks(text, language)
        try:
            for pcm in synthesis:
                if first:
                    yield wav_header(sample_rate or speaking.sample_rate) + pcm
                    first = False
                else:
                    yield pcm
        finally:
            synthesis.close()


def mount_health_route(app: FastAPI, runtime: Runtime) -> None:
    """Mount /health over the one runtime."""

    @app.get(HEALTH_PATH)
    def health() -> Health:
        return runtime.health()
