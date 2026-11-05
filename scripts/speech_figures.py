"""How fast a voice speaks: the figures the stage and image proofs hold it to."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from speech_runtime_service.voices import RESIDENT_VOICE, VoiceId


@dataclass(frozen=True)
class VoiceFigures:
    """How a voice spoke the proofs' sentence: load, first audio, speed, card memory.

    `real_time_factor` is audio seconds over wall seconds, so below 1 is slower
    than real time; `peak_card_mib` is the card's peak above its use before the
    voice's first request.
    """

    load_seconds: float
    first_audio_seconds: float
    real_time_factor: float
    peak_card_mib: int


# Qwen on this host's RTX 3090, loaded and driven in process before the image
# carried it, medians of five runs: the P0 plan's G0 measurement.
QWEN_G0: Final = VoiceFigures(
    load_seconds=6.9,
    first_audio_seconds=8.44,
    real_time_factor=0.78,
    peak_card_mib=3714,
)
# Qwen does not stream, so its first byte comes after the whole sentence, and
# the first request also waits for the load. G0 measured neither the HTTP hop
# nor the container, so the limit is half as much again as G0's load and first
# audio together.
FIRST_BYTE_LIMIT_S: Final = {
    RESIDENT_VOICE: 0.5,
    VoiceId.QWEN: 1.5 * (QWEN_G0.load_seconds + QWEN_G0.first_audio_seconds),
}
