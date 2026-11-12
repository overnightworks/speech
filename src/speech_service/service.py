"""HTTP contract: health and the runtime that speaks."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from speech_client.wire import (
    HEALTH_PATH,
    Health,
    HearingHealth,
    ResponseFormat,
    SpeakingHealth,
)
from speech_service.card import card_memory_mb
from speech_service.clock import LoopClock
from speech_service.failures import PayloadFreeFailure, SpeakingFailedError
from speech_service.hearing import Hearing, hearing_from_settings
from speech_service.ogg_opus import ogg_opus_chunks
from speech_service.pcm import wav_header
from speech_service.speaking import speaking_for_voice
from speech_service.voice_slot import VoiceSlot
from speech_service.voices import RESIDENT_VOICE, VoiceId, missing_artifacts

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Mapping
    from concurrent.futures import Future

    from fastapi import FastAPI

    from speech_service.config import Settings

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
    """The seams that check and build the voices to load."""

    engine_factory: Callable[[VoiceId], SpeakingEngine] | None = None
    missing_artifacts: Callable[[VoiceId], tuple[str, ...]] | None = None


def _is_start_voice(voice: VoiceId) -> bool:
    """Whether the start loads `voice` rather than its first request."""
    return voice is RESIDENT_VOICE


class Runtime:
    """The voice slots, hearing, the start phase, and the one readiness they share."""

    def __init__(
        self,
        voices: Mapping[VoiceId, SpeakingEngine | None],
        *,
        dependencies: RuntimeDependencies | None = None,
        hearing: Hearing | None = None,
        memory_probe: Callable[[], int] | None = None,
    ) -> None:
        """Offer the keys of `voices`, a slot each, in the order of `VoiceId`.

        A value is an engine already loaded, `None` a voice to load. Nothing is
        loaded until `load`, and a voice that is not a start voice not until its
        first request.
        """
        dependencies = dependencies or RuntimeDependencies()
        self.hearing = hearing or Hearing(None, _CLOCK)
        self.clock = _CLOCK
        self.slots: dict[VoiceId, VoiceSlot] = {
            voice: VoiceSlot(voice, dependencies, _CLOCK, voices[voice])
            for voice in sorted(voices)
        }
        self._start_voices = tuple(filter(_is_start_voice, self.slots))
        self._memory_probe = memory_probe or card_memory_mb
        self._phase_lock = threading.Lock()
        self._loading = False
        self._stopping = False

    @classmethod
    def from_settings(cls, settings: Settings) -> Runtime:
        """Compose the offer: Piper, and the GPU voices `SPEECH_GPU_VOICES` names."""
        return cls(
            {RESIDENT_VOICE: None} | dict.fromkeys(settings.gpu_voices),
            hearing=hearing_from_settings(settings, _CLOCK),
            dependencies=RuntimeDependencies(
                engine_factory=lambda voice: speaking_for_voice(settings, voice),
                missing_artifacts=lambda voice: missing_artifacts(settings, voice),
            ),
        )

    @property
    def loading(self) -> bool:
        """Whether the start load is still running."""
        with self._phase_lock:
            return self._loading

    def load(self) -> None:
        """Load each start voice in order, then hearing, unless shutdown began.

        A voice that is not a start voice loads on its first request.
        """
        self.begin_loading()
        steps = [self.slots[voice].load for voice in self._start_voices]
        try:
            for step in (*steps, self.hearing.load):
                if self._shutdown_begun():
                    return
                step()
        finally:
            with self._phase_lock:
                self._loading = False

    def begin_loading(self) -> None:
        """Publish startup before its background task can be scheduled."""
        with self._phase_lock:
            self._loading = True

    def capture(self, voice: VoiceId) -> SpeakingEngine | None:
        """The engine `voice` speaks with now; `None` while it cannot speak.

        This is the one readiness: no voice speaks during the start load or once
        shutdown began, and /health reports exactly what this answers.
        """
        if self.loading:
            return None
        return self.slots[voice].capture()

    def load_on_demand(self, voice: VoiceId) -> Future[SpeakingEngine | None] | None:
        """The voice's shared load, started if none runs; `None` if none may.

        A start voice never loads on demand, and no voice does during the start.
        """
        if _is_start_voice(voice) or self.loading:
            return None
        return self.slots[voice].shared_load()

    def health(self) -> Health:
        """The contract body for GET /health."""
        self.reconcile_failed_active()
        voices = {
            voice.value: slot.health(ready=self.capture(voice) is not None)
            for voice, slot in self.slots.items()
        }
        start_ended = not (self.loading or self._shutdown_begun())
        return Health(
            speaking=SpeakingHealth(
                ready=start_ended
                and all(voices[voice.value].ready for voice in self._start_voices),
                voices=voices,
            ),
            hearing=HearingHealth.model_validate(self.hearing.health()),
            card_memory_mb=self._memory_probe(),
        )

    def begin_shutdown(self) -> None:
        """Refuse new work before the server starts its own shutdown."""
        with self._phase_lock:
            self._stopping = True
        for slot in self.slots.values():
            slot.begin_shutdown()

    def set_fatal_callback(self, callback: Callable[[], None]) -> None:
        """Route an owned speaking-engine failure to the service orchestrator."""
        for slot in self.slots.values():
            slot.set_fatal_callback(callback)

    def close(self) -> None:
        """Close every slot and its load, then hearing; raise if a slot failed."""
        self.begin_shutdown()
        failed = False
        try:
            for slot in self.slots.values():
                try:
                    slot.close()
                except RuntimeError:
                    failed = True
        finally:
            self.hearing.close()
        if failed:
            message = "speaking engine cleanup failed"
            raise RuntimeError(message)

    def reconcile_failed_active(self) -> None:
        """Close an active engine that stopped being ready."""
        for slot in self.slots.values():
            slot.reconcile_failed_active()

    def _shutdown_begun(self) -> bool:
        with self._phase_lock:
            return self._stopping


def audio_chunks(
    speaking: SpeakingEngine,
    text: str,
    language: str,
    response_format: ResponseFormat,
) -> Iterator[bytes]:
    """`text` spoken by `speaking` in `response_format`, as the voice produces it."""
    with PayloadFreeFailure(logger, "speaking failed", reraise_as=SpeakingFailedError):
        synthesis = speaking.pcm_chunks(text, language)
        try:
            if response_format is ResponseFormat.OPUS:
                yield from ogg_opus_chunks(synthesis, speaking.sample_rate)
            else:
                yield from _wav_chunks(synthesis, speaking.sample_rate)
        finally:
            synthesis.close()


def _wav_chunks(pcm_chunks: Iterator[bytes], sample_rate: int) -> Iterator[bytes]:
    """One WAV: its header leaves with the first PCM chunk."""
    first = True
    for pcm in pcm_chunks:
        if first:
            yield wav_header(sample_rate) + pcm
            first = False
        else:
            yield pcm


def mount_health_route(app: FastAPI, runtime: Runtime) -> None:
    """Mount /health over the one runtime."""

    @app.get(HEALTH_PATH)
    def health() -> Health:
        return runtime.health()
