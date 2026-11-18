"""The catalogue file the repository ships, loaded for scripts and tests."""

from pathlib import Path
from typing import TYPE_CHECKING, Final

from speech_service.catalogue import Engine, load_catalogue

if TYPE_CHECKING:
    from speech_client.wire import VoiceId

CATALOGUE_FILE: Final = Path(__file__).resolve().parents[1] / "deploy/catalogue.toml"
CATALOGUE: Final = load_catalogue(CATALOGUE_FILE)
PIPER_VOICES: Final[tuple[VoiceId, ...]] = tuple(
    voice for voice, entry in CATALOGUE.items() if entry.engine is Engine.PIPER
)
"""The shipped catalogue's Piper voices, in catalogue order."""
