"""The closed factory for resident and isolated local speaking engines."""

from __future__ import annotations

import http.client
import json
import logging
from functools import partial
from http import HTTPStatus
from typing import TYPE_CHECKING, Protocol, assert_never, runtime_checkable

from speech_provider_contract import (
    QWEN_LANGUAGE_NAMES,
    QWEN_MODEL_ID,
    QWEN_SAMPLE_RATE,
    QwenSpeaker,
)

from speech_client.wire import GivesMarks, SentenceMarks
from speech_service.catalogue import PiperEntry, QwenEntry
from speech_service.espeak_marks import (
    EspeakEvents,
    Sentence,
    WordMarker,
    load_espeak_events,
)
from speech_service.pcm import BYTES_PER_SAMPLE
from speech_service.provider_process import (
    ProviderId,
    ProviderLaunch,
    ProviderProcess,
    ProviderSynthesisFailedError,
    UnsupportedLanguageError,
    provider_entrypoint,
)
from speech_service.voices import piper_voice_paths

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from pathlib import Path

    from piper import AudioChunk, PiperVoice

    from speech_service.catalogue import CatalogueEntry
    from speech_service.config import Settings

logger = logging.getLogger(__name__)


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
    marks = isinstance(speaking, MarkingEngine) and speaking.gives_word_marks
    # Piper's sounds come from the pass that proves its words.
    return GivesMarks(words=marks, sounds=marks)


def load_piper_voice(onnx: Path) -> PiperVoice:
    """The voice of an ONNX file, with the alignment output the word marks need."""
    from piper import PiperVoice

    return PiperVoice.load(onnx, include_alignments=True)


class PiperSpeaking:
    """One Piper voice, loaded once and held for the process."""

    def __init__(
        self,
        model_name: str,
        cache: Path,
        *,
        load_voice: Callable[[Path], PiperVoice] = load_piper_voice,
        load_events: Callable[[PiperVoice], EspeakEvents | None] = load_espeak_events,
    ) -> None:
        """Remember the catalogue name, where ONNX files are kept and how to load."""
        self.model_name = model_name
        self.sample_rate = 0
        self.ready = False
        self.gives_word_marks = False
        self._cache = cache
        self._load_voice = load_voice
        self._load_events = load_events
        self._voice: PiperVoice | None = None
        self._events: EspeakEvents | None = None

    def load(self) -> None:
        """Load the exact local Piper artifacts and keep them resident."""
        onnx, metadata = piper_voice_paths(self._cache, self.model_name)
        if not onnx.is_file() or not metadata.is_file():
            raise FileNotFoundError
        voice = self._load_voice(onnx)
        self._voice = voice
        self._events = self._load_events(voice)
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


_RESPONSE_SECONDS = 120.0
"""How long the server may stay silent in the middle of a response."""
_PCM_READ_BYTES = 64 * 1024
_SAMPLE_RATE_HEADER = "x-audio-sample-rate"
_SPEECH_PATH = "/v1/audio/speech"


class QwenSpeaking:
    """One Qwen voice, spoken through the vLLM-Omni server its process owns."""

    def __init__(self, process: ProviderProcess, speaker: QwenSpeaker) -> None:
        """Remember the server's owner and the voice every request names."""
        self.model_name = QWEN_MODEL_ID
        self.sample_rate = QWEN_SAMPLE_RATE
        self._process = process
        self._speaker = speaker

    @property
    def ready(self) -> bool:
        """Whether the server is up, healthy and not failed."""
        return self._process.ready

    def load(self) -> None:
        """Start the server and return once it answers its health check."""
        self._process.load()

    def set_fatal_callback(self, callback: Callable[[], None]) -> None:
        """Give the server's owner the runtime's lifecycle failure seam."""
        self._process.set_fatal_callback(callback)

    def close(self) -> None:
        """Stop the server and everything it started."""
        self._process.close()

    def pcm_chunks(
        self, text: str, language: str, instructions: str | None = None
    ) -> Generator[bytes]:
        """Stream the server's PCM, each part as it arrives, in whole samples.

        Closing the generator closes the connection. A refusal, a broken
        connection, audio that ends inside a sample or a sample rate that is not
        Qwen's fails the synthesis and the server with it.
        """
        language_name = _language_name(language)
        request: dict[str, object] = {
            "input": text,
            "voice": self._speaker.value,
            "language": language_name,
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
        }
        if instructions is not None:
            request["instructions"] = instructions
        try:
            connection = self._process.connect(_RESPONSE_SECONDS)
            try:
                yield from self._answer(connection, request)
            finally:
                connection.close()
        except OSError, http.client.HTTPException:
            self._fail(reason="connection")
            raise ProviderSynthesisFailedError from None

    def _answer(
        self, connection: http.client.HTTPConnection, request: dict[str, object]
    ) -> Iterator[bytes]:
        connection.request(
            "POST",
            _SPEECH_PATH,
            json.dumps(request),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != HTTPStatus.OK:
            self._fail(status=response.status)
            raise ProviderSynthesisFailedError
        rate = response.getheader(_SAMPLE_RATE_HEADER)
        if rate is not None and rate.strip() != str(self.sample_rate):
            self._fail(reason="sample_rate")
            raise ProviderSynthesisFailedError
        carried = b""
        while arrived := response.read1(_PCM_READ_BYTES):
            available = carried + arrived
            whole = len(available) - len(available) % BYTES_PER_SAMPLE
            carried = available[whole:]
            if whole:
                yield available[:whole]
        if carried:
            self._fail(reason="partial_sample")
            raise ProviderSynthesisFailedError

    def _fail(self, **fields: object) -> None:
        """Log why the synthesis failed, without the server's answer, and stop it."""
        logger.warning("synthesis.failed", extra=fields)
        self._process.mark_failed()


def _language_name(language: str) -> str:
    normalized = language.strip().lower().replace("_", "-")
    name = QWEN_LANGUAGE_NAMES.get((normalized or "de").split("-", 1)[0])
    if name is None:
        raise UnsupportedLanguageError
    return name


LocalSpeaking = PiperSpeaking | QwenSpeaking


def speaking_for_voice(settings: Settings, voice: CatalogueEntry) -> LocalSpeaking:
    """Construct the local engine of one catalogue voice."""
    match voice:
        case PiperEntry():
            return PiperSpeaking(voice.id.value, settings.voice_cache)
        case QwenEntry():
            return QwenSpeaking(
                ProviderProcess(
                    ProviderLaunch(
                        executable=provider_entrypoint(
                            settings.provider_root, ProviderId.QWEN
                        ),
                        arguments=("--cache", str(settings.huggingface_cache)),
                    )
                ),
                voice.pin.speaker,
            )
        case unreachable:
            assert_never(unreachable)
