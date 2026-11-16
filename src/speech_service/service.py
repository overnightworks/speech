"""HTTP contract: health and the runtime that speaks."""

from __future__ import annotations

import base64
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from speech_client.wire import (
    HEALTH_PATH,
    AudioEventType,
    Health,
    HearingHealth,
    ResponseFormat,
    SentenceMarks,
    SpeakingHealth,
    StreamFormat,
    VoiceId,
    audio_event,
)
from speech_service.card import card_memory_mb
from speech_service.clock import LoopClock
from speech_service.failures import PayloadFreeFailure, SpeakingFailedError
from speech_service.hearing import Hearing, hearing_from_settings
from speech_service.ogg_opus import ogg_opus_chunks
from speech_service.pcm import wav_header
from speech_service.speaking import MarkingEngine, speaking_for_voice
from speech_service.voice_slot import VoiceSlot
from speech_service.voices import loads_at_start, missing_artifacts

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Mapping
    from concurrent.futures import Future

    from fastapi import FastAPI
    from logline import StepTimer

    from speech_client.wire import SpeechRequest
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

    def pcm_chunks(
        self, text: str, language: str, instructions: str | None = None
    ) -> Generator[bytes]:
        """16-bit mono PCM as soon as the model produces it, in the tone asked."""

    def close(self) -> None:
        """Release owned resources once."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeDependencies:
    """The seams that check and build the voices to load."""

    engine_factory: Callable[[VoiceId], SpeakingEngine] | None = None
    missing_artifacts: Callable[[VoiceId], tuple[str, ...]] | None = None


class Runtime:
    """The voice slots, hearing, the start phase, and the one readiness they share."""

    def __init__(
        self,
        voices: Mapping[VoiceId, SpeakingEngine | None],
        *,
        dependencies: RuntimeDependencies | None = None,
        hearing: Hearing | None = None,
        memory_probe: Callable[[], int] | None = None,
        load_gpu_voices_at_start: bool = False,
    ) -> None:
        """Offer the keys of `voices`, a slot each, in catalogue order.

        A value is an engine already loaded, `None` a voice to load. Nothing is
        loaded until `load`, and a voice that is not a start voice not until its
        first request. Every Piper voice is a start voice, and a GPU voice is one
        when `load_gpu_voices_at_start`.
        """
        dependencies = dependencies or RuntimeDependencies()
        self.hearing = hearing or Hearing(None, _CLOCK)
        self.clock = _CLOCK
        self.slots: dict[VoiceId, VoiceSlot] = {
            voice: VoiceSlot(voice, dependencies, _CLOCK, voices[voice])
            for voice in VoiceId
            if voice in voices
        }
        self._start_voices = tuple(
            voice
            for voice in self.slots
            if loads_at_start(voice, gpu_voices=load_gpu_voices_at_start)
        )
        self._memory_probe = memory_probe or card_memory_mb
        self._phase_lock = threading.Lock()
        self._loading = False
        self._stopping = False

    @classmethod
    def from_settings(cls, settings: Settings) -> Runtime:
        """Compose the offer `SPEECH_VOICES` names."""
        return cls(
            dict.fromkeys(settings.voices),
            hearing=hearing_from_settings(settings, _CLOCK),
            load_gpu_voices_at_start=settings.load_gpu_voices_at_start,
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

        A Piper voice never loads on demand, and no voice does during the start.
        A GPU voice the start loaded reloads here once its worker died.
        """
        if loads_at_start(voice) or self.loading:
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
        """Refuse new work and report not ready; work already admitted runs on.

        Engines stay open until `close`, so a stream, a recording or a live
        session already admitted can end whole.
        """
        with self._phase_lock:
            self._stopping = True
        for slot in self.slots.values():
            slot.begin_shutdown()
        self.hearing.begin_drain()

    def abandon_loads(self) -> None:
        """Stop the voice loads under way, so no request waits on one through a drain.

        Blocks while a building engine closes; call it off the event loop.
        """
        for slot in self.slots.values():
            slot.abandon_load()

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
    speaking: SpeakingEngine, request: SpeechRequest, step: StepTimer
) -> Iterator[bytes]:
    """`request` spoken by `speaking` in its format, as the voice produces it.

    The chunks are the audio itself, or for `sse` its events: the audio as deltas,
    each sentence's marks after its audio, and `done` once the voice finished.
    `step` times the speaking: it marks the first chunk, counts the words marked
    and ends with the stream, whatever ends it, a close before the first chunk
    included.
    """
    stream = _timed_chunks(speaking, request, step)
    # Started here, a close before the first chunk still ends `step`: a generator
    # closed unstarted never runs its body.
    next(stream)
    return stream


def _timed_chunks(
    speaking: SpeakingEngine, request: SpeechRequest, step: StepTimer
) -> Iterator[bytes]:
    with (
        step,
        PayloadFreeFailure(
            logger,
            "speaking.failed",
            level=logging.WARNING,
            reraise_as=SpeakingFailedError,
        ),
    ):
        yield b""
        synthesis = (
            _marked_chunks(speaking, request)
            if request.stream_format is StreamFormat.SSE
            else speaking.pcm_chunks(
                request.input, request.spoken_language, request.instructions
            )
        )
        try:
            chunks = (
                ogg_opus_chunks(synthesis, speaking.sample_rate)
                if request.response_format is ResponseFormat.OPUS
                else _wav_chunks(synthesis, speaking.sample_rate)
            )
            if request.stream_format is StreamFormat.SSE:
                yield from _events(chunks, step)
            else:
                for chunk in chunks:
                    step.first_output()
                    yield chunk
        finally:
            synthesis.close()


def _marked_chunks(
    speaking: SpeakingEngine, request: SpeechRequest
) -> Generator[bytes | SentenceMarks]:
    """The voice's PCM with its marks after each sentence's audio.

    A voice that marks nothing ends its audio with one marks event of nulls.
    """
    if isinstance(speaking, MarkingEngine):
        yield from speaking.marked_chunks(
            request.input, request.spoken_language, request.instructions
        )
        return
    yield from speaking.pcm_chunks(
        request.input, request.spoken_language, request.instructions
    )
    yield SentenceMarks()


def _events(
    chunks: Iterator[bytes | SentenceMarks], step: StepTimer
) -> Iterator[bytes]:
    """The events of `chunks`, then `done`; `step` counts the words marked."""
    words = 0
    for chunk in chunks:
        if isinstance(chunk, SentenceMarks):
            words += len(chunk.words or ())
            step.add(marks_words=words)
            yield audio_event(AudioEventType.MARKS, **chunk.model_dump(mode="json"))
        else:
            step.first_output()
            yield audio_event(
                AudioEventType.DELTA, audio=base64.b64encode(chunk).decode("ascii")
            )
    yield audio_event(AudioEventType.DONE)


def _wav_chunks(
    pcm_chunks: Iterator[bytes | SentenceMarks], sample_rate: int
) -> Iterator[bytes | SentenceMarks]:
    """One WAV: its header leaves with the first PCM chunk; marks pass through."""
    first = True
    for pcm in pcm_chunks:
        if isinstance(pcm, SentenceMarks):
            yield pcm
        elif first:
            yield wav_header(sample_rate) + pcm
            first = False
        else:
            yield pcm


def mount_health_route(app: FastAPI, runtime: Runtime) -> None:
    """Mount /health over the one runtime."""

    @app.get(HEALTH_PATH)
    def health() -> Health:
        return runtime.health()
