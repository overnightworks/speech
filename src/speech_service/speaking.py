"""The closed factory for resident and isolated local speaking engines."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final, Protocol, assert_never, runtime_checkable

from speech_provider_contract import (
    QWEN_DEVICE,
    QWEN_MODEL_ID,
    QWEN_SAMPLE_RATE,
    QwenSpeaker,
)

from speech_client.wire import (
    VOICES,
    GivesMarks,
    SentenceMarks,
    SpeakingModel,
    VoiceId,
)
from speech_service.espeak_marks import (
    EspeakEvents,
    Sentence,
    WordMarker,
    load_espeak_events,
)
from speech_service.provider_process import (
    ProviderId,
    ProviderLaunch,
    ProviderProcess,
    provider_entrypoint,
)
from speech_service.voices import piper_voice_paths

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from piper import AudioChunk, PiperVoice

    from speech_service.config import Settings


@runtime_checkable
class MarkingEngine(Protocol):
    """An engine that marks, after each sentence's audio, what it spoke."""

    gives_word_marks: bool

    def marked_chunks(
        self, text: str, language: str, instructions: str | None = None
    ) -> Generator[bytes | SentenceMarks]:
        """16-bit mono PCM as `pcm_chunks` gives it, each sentence's marks after it."""


def gives_marks(speaking: object) -> GivesMarks:
    """The marks `speaking` gives; none before it is loaded."""
    words = isinstance(speaking, MarkingEngine) and speaking.gives_word_marks
    return GivesMarks(words=words, sounds=False)


class PiperSpeaking:
    """One Piper voice, loaded once and held for the process."""

    def __init__(self, model_name: str, cache: Path) -> None:
        """Remember the catalogue name and where ONNX files are kept."""
        self.model_name = model_name
        self.sample_rate = 0
        self.ready = False
        self.gives_word_marks = False
        self._cache = cache
        self._voice: PiperVoice | None = None
        self._events: EspeakEvents | None = None

    def load(self) -> None:
        """Load the exact local Piper artifacts and keep them resident."""
        from piper import PiperVoice

        onnx, metadata = piper_voice_paths(self._cache, self.model_name)
        if not onnx.is_file() or not metadata.is_file():
            raise FileNotFoundError
        voice = PiperVoice.load(onnx, include_alignments=True)
        self._voice = voice
        self._events = load_espeak_events(voice)
        self.gives_word_marks = self._events is not None
        self.sample_rate = voice.config.sample_rate
        self.ready = True

    def pcm_chunks(
        self, text: str, language: str, instructions: str | None = None
    ) -> Generator[bytes]:
        """Yield 16-bit mono PCM as soon as Piper produces a chunk."""
        del language, instructions
        for chunk in self._loaded_voice().synthesize(text):
            yield chunk.audio_int16_bytes

    def marked_chunks(
        self, text: str, language: str, instructions: str | None = None
    ) -> Generator[bytes | SentenceMarks]:
        """Yield each sentence's PCM, then its marks, which are null if words are off.

        A sentence's marks are made after its audio left, so they never delay it.
        """
        del language, instructions
        voice = self._loaded_voice()
        marker = (
            None
            if self._events is None
            else WordMarker(
                text,
                partial(self._events.read, text, voice.config.espeak_voice),
                voice.config.sample_rate,
            )
        )
        for chunk in voice.synthesize(text, include_alignments=True):
            yield chunk.audio_int16_bytes
            yield SentenceMarks() if marker is None else marker.after(_spoken(chunk))

    def _loaded_voice(self) -> PiperVoice:
        if self._voice is None:
            message = "speaking model is not loaded"
            raise RuntimeError(message)
        return self._voice

    def close(self) -> None:
        """Piper stays retained for the process and needs no explicit cleanup."""


def _spoken(chunk: AudioChunk) -> Sentence:
    alignments = chunk.phoneme_alignments
    return Sentence(
        phonemes=tuple(chunk.phonemes),
        token_samples=(
            None
            if alignments is None
            else tuple(alignment.num_samples for alignment in alignments)
        ),
        audio_samples=len(chunk.audio_float_array),
    )


LocalSpeaking = PiperSpeaking | ProviderProcess


_QWEN_SPEAKERS: Final = {VoiceId.RYAN: QwenSpeaker.RYAN}


def speaking_for_voice(settings: Settings, voice: VoiceId) -> LocalSpeaking:
    """Construct one supported local engine from the closed catalogue id."""
    match VOICES[voice].model:
        case SpeakingModel.PIPER:
            return PiperSpeaking(voice.value, settings.voice_cache)
        case SpeakingModel.QWEN:
            return ProviderProcess(
                ProviderLaunch(
                    model_name=QWEN_MODEL_ID,
                    sample_rate=QWEN_SAMPLE_RATE,
                    executable=provider_entrypoint(
                        settings.provider_root, ProviderId.QWEN
                    ),
                    arguments=(
                        "--device",
                        QWEN_DEVICE,
                        "--cache",
                        str(settings.huggingface_cache),
                        "--speaker",
                        _QWEN_SPEAKERS[voice].value,
                    ),
                ),
            )
        case unreachable:
            assert_never(unreachable)
