"""The voice catalogue: the typed records of the packaged file, and its loader."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from speech_provider_contract import QWEN_SAMPLE_RATE, QwenSpeaker

from speech_client.wire import Language, SpeakingModel, VoiceId

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from importlib.resources.abc import Traversable

_SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"
_REVISION_PATTERN: Final = r"^[0-9a-f]{40}$"
PACKAGED_CATALOGUE: Final = files("speech_service") / "catalogue.toml"
"""The catalogue the service's package ships."""


class CatalogueError(ValueError):
    """The catalogue cannot be read, or an entry of it is wrong."""


@dataclass(frozen=True, slots=True)
class _EngineFacts:
    wire_model: SpeakingModel
    takes_instructions: bool
    sample_rate_before_load: int
    loads_at_start: bool


class Engine(StrEnum):
    """The engines a voice is spoken by, and what the service knows of each."""

    PIPER = "piper"
    QWEN3_TTS_CUSTOMVOICE = "qwen3-tts-customvoice"

    @property
    def wire_model(self) -> SpeakingModel:
        """The wire's `model` for this engine, until the wire drops `model`."""
        return _FACTS[self].wire_model

    @property
    def takes_instructions(self) -> bool:
        """Whether a voice of this engine speaks in the tone `instructions` name."""
        return _FACTS[self].takes_instructions

    @property
    def sample_rate_before_load(self) -> int:
        """The rate a voice reports before it loads; 0 where the model sets it."""
        return _FACTS[self].sample_rate_before_load

    @property
    def loads_at_start(self) -> bool:
        """Whether the start loads a voice of this engine, not its first request."""
        return _FACTS[self].loads_at_start


_FACTS: Final = {
    Engine.PIPER: _EngineFacts(
        wire_model=SpeakingModel.PIPER,
        takes_instructions=False,
        sample_rate_before_load=0,
        loads_at_start=True,
    ),
    Engine.QWEN3_TTS_CUSTOMVOICE: _EngineFacts(
        wire_model=SpeakingModel.QWEN,
        takes_instructions=True,
        sample_rate_before_load=QWEN_SAMPLE_RATE,
        loads_at_start=False,
    ),
}


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class PinnedFile(_Record):
    """A file of a voice: its path in the repository and its SHA-256."""

    path: Annotated[str, Field(min_length=1)]
    sha256: Annotated[str, Field(pattern=_SHA256_PATTERN)]


class PiperPin(_Record):
    """The repository and revision a Piper voice is fetched at, and its two files."""

    repository: Annotated[str, Field(min_length=1)]
    revision: Annotated[str, Field(pattern=_REVISION_PATTERN)]
    model: PinnedFile
    metadata: PinnedFile


class QwenCustomVoicePin(_Record):
    """The speaker of Qwen's custom-voice model that a voice speaks with."""

    speaker: QwenSpeaker


class _Voice(_Record):
    id: VoiceId
    name: Annotated[str, Field(min_length=1)]
    languages: Annotated[tuple[Language, ...], Field(min_length=1)]


class PiperEntry(_Voice):
    """A Piper voice: its files are fetched at the pin and loaded in the service."""

    engine: Literal[Engine.PIPER]
    pin: PiperPin


class QwenEntry(_Voice):
    """A voice of Qwen's custom-voice model, spoken by the provider process."""

    engine: Literal[Engine.QWEN3_TTS_CUSTOMVOICE]
    pin: QwenCustomVoicePin


CatalogueEntry = PiperEntry | QwenEntry
Catalogue = dict[VoiceId, CatalogueEntry]


class _CatalogueFile(_Record):
    voices: dict[str, Annotated[CatalogueEntry, Field(discriminator="engine")]]


def load_catalogue(path: Traversable) -> Catalogue:
    """Read the catalogue at `path` (a `Path` or the packaged file), in `VoiceId` order.

    Raises `CatalogueError` naming the file and, for an entry, the field.
    """
    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        message = f"{path} cannot be read as a catalogue: {error}"
        raise CatalogueError(message) from None
    try:
        file = _CatalogueFile.model_validate(_with_ids(path, raw))
    except ValidationError as error:
        problems = "; ".join(_describe(problem) for problem in error.errors())
        message = f"{path} is not a valid catalogue: {problems}"
        raise CatalogueError(message) from None
    entries = {entry.id: entry for entry in file.voices.values()}
    return {voice: entries[voice] for voice in VoiceId if voice in entries}


def offered_voices(
    catalogue: Catalogue,
    requested: Collection[VoiceId] | None,
    carried: Collection[Engine],
) -> tuple[VoiceId, ...]:
    """The voices to offer, in catalogue order: `requested`, else those of `carried`.

    Raises `CatalogueError` naming a requested voice the catalogue does not hold.
    """
    if requested is None:
        return tuple(
            voice for voice, entry in catalogue.items() if entry.engine in carried
        )
    absent = sorted(voice.value for voice in requested if voice not in catalogue)
    if absent:
        message = f"SPEECH_VOICES names {', '.join(absent)}, which the catalogue lacks"
        raise CatalogueError(message)
    return tuple(voice for voice in catalogue if voice in requested)


def _with_ids(path: Traversable, raw: Mapping[str, object]) -> Mapping[str, object]:
    voices = raw.get("voices")
    if not isinstance(voices, dict):
        return raw
    for key, entry in voices.items():
        if isinstance(entry, dict) and "id" in entry:
            message = (
                f"{path} is not a valid catalogue: voices.{key}.id: "
                "the id is the table key and is not a field"
            )
            raise CatalogueError(message)
    return {
        **raw,
        "voices": {
            key: {**entry, "id": key} if isinstance(entry, dict) else entry
            for key, entry in voices.items()
        },
    }


def _describe(problem: Mapping[str, object]) -> str:
    location = ".".join(str(part) for part in problem["loc"] if part not in set(Engine))
    return f"{location}: {problem['msg']}"
