"""HTTP contract: health and the runtime that speaks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from speech_runtime_client.wire import Health, HearingHealth, SpeakingHealth
from speech_runtime_service.card import card_memory_mb
from speech_runtime_service.clock import LoopClock
from speech_runtime_service.failures import PayloadFreeFailure, SpeakingFailedError
from speech_runtime_service.hearing import Hearing, hearing_from_settings
from speech_runtime_service.pcm import wav_header
from speech_runtime_service.speaking import speaking_for_voice
from speech_runtime_service.voice_slot import VoiceSlot
from speech_runtime_service.voices import VoiceId, installed_voice_ids, voice_for_model

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator

    from fastapi import FastAPI

    from speech_runtime_service.config import Settings

logger = logging.getLogger(__name__)

_CLOCK: Final = LoopClock()


class SpeakingEngine(Protocol):
    """A resident voice."""

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
    """The configured voice and the seams that check and build its engine."""

    voice: VoiceId = VoiceId.PIPER
    engine_factory: Callable[[VoiceId], SpeakingEngine] | None = None
    artifact_checker: Callable[[VoiceId], bool] | None = None


class Runtime:
    """The voice slot, hearing, and the flags a caller waits on."""

    def __init__(
        self,
        speaking: SpeakingEngine | None,
        *,
        memory_probe: Callable[[], int] | None = None,
        dependencies: RuntimeDependencies | None = None,
        hearing: Hearing | None = None,
    ) -> None:
        """Hold the voice slot and hearing. Neither is loaded until `load`."""
        dependencies = dependencies or RuntimeDependencies()
        self.hearing = hearing or Hearing(None, _CLOCK)
        self.clock = _CLOCK
        self.slots = {dependencies.voice: VoiceSlot(dependencies, _CLOCK, speaking)}
        self._voice = dependencies.voice
        self._memory_probe = memory_probe or card_memory_mb

    @classmethod
    def from_settings(cls, settings: Settings) -> Runtime:
        """Build the production engines the environment configures."""
        return cls(
            None,
            hearing=hearing_from_settings(settings, _CLOCK),
            dependencies=RuntimeDependencies(
                voice=voice_for_model(settings.speaking_model),
                engine_factory=lambda voice: speaking_for_voice(settings, voice),
                artifact_checker=lambda voice: voice in installed_voice_ids(settings),
            ),
        )

    @property
    def loading(self) -> bool:
        """Whether startup is still running."""
        return self.slots[self._voice].loading

    def load(self) -> None:
        """Start the configured voice, then hearing; the orchestrator owns processes."""
        self.begin_loading()
        self.slots[self._voice].load(self.hearing.load)

    def begin_loading(self) -> None:
        """Publish startup before its background task can be scheduled."""
        self.slots[self._voice].begin_loading()

    def health(self) -> Health:
        """The contract body for GET /health."""
        self.reconcile_failed_active()
        voice = self.slots[self._voice].health()
        return Health(
            speaking=SpeakingHealth(
                ready=voice.ready, voices={self._voice.value: voice}
            ),
            hearing=HearingHealth.model_validate(self.hearing.health()),
            card_memory_mb=self._memory_probe(),
        )

    def capture_speaking(self) -> VoiceAdmission | None:
        """Capture one stable engine for an entire streaming response."""
        speaking = self.slots[self._voice].capture()
        if speaking is None:
            return None
        return VoiceAdmission(speaking, self._voice)

    def begin_shutdown(self) -> None:
        """Refuse new work before the server starts its own shutdown."""
        self.slots[self._voice].begin_shutdown()

    def set_fatal_callback(self, callback: Callable[[], None]) -> None:
        """Route an owned speaking-engine failure to the service orchestrator."""
        self.slots[self._voice].set_fatal_callback(callback)

    def close(self) -> None:
        """Close the voice slot's engines and hearing."""
        self.slots[self._voice].close(self.hearing.close)

    def reconcile_failed_active(self) -> None:
        """Close an active engine that stopped being ready."""
        self.slots[self._voice].reconcile_failed_active()


@dataclass(frozen=True, slots=True)
class VoiceAdmission:
    """One captured engine and the voice it speaks for."""

    speaking: SpeakingEngine
    voice: VoiceId

    @property
    def sample_rate(self) -> int:
        """The captured engine's rate."""
        return self.speaking.sample_rate


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

    @app.get("/health")
    def health() -> Health:
        return runtime.health()
