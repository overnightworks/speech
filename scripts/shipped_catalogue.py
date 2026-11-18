"""The catalogue file the repository ships, loaded for scripts and tests."""

from pathlib import Path
from typing import TYPE_CHECKING, Final

from speech_service.catalogue import load_catalogue, offered_voices

if TYPE_CHECKING:
    from speech_client.wire import VoiceId

CATALOGUE_FILE: Final = Path(__file__).resolve().parents[1] / "deploy/catalogue.toml"
CATALOGUE: Final = load_catalogue(CATALOGUE_FILE)
PIPER_VOICES: Final[tuple[VoiceId, ...]] = offered_voices(CATALOGUE, None)
"""The voices the service offers by default, in catalogue order."""
