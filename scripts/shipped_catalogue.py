"""The catalogue file the repository ships, loaded for scripts and tests."""

from typing import TYPE_CHECKING, Final

from speech_service.catalogue import (
    PACKAGED_CATALOGUE,
    Engine,
    load_catalogue,
    offered_voices,
)

if TYPE_CHECKING:
    from speech_client.wire import Language, VoiceId

CATALOGUE: Final = load_catalogue(PACKAGED_CATALOGUE)
PIPER_VOICES: Final[tuple[VoiceId, ...]] = offered_voices(
    CATALOGUE, None, {Engine.PIPER}
)
"""The voices the service offers without a provider root, in catalogue order."""
ON_DEMAND_VOICES: Final[tuple[VoiceId, ...]] = tuple(
    voice for voice, entry in CATALOGUE.items() if not entry.engine.loads_at_start
)
"""The voices the start does not load, in catalogue order."""


def first_language(voice: VoiceId) -> Language:
    """The first language the catalogue names for `voice`."""
    return CATALOGUE[voice].languages[0]
