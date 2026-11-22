"""The scripts' one reader of a WAV body: its chunks, its PCM and its level.

Standard library only: `contract_client` is an outside caller of the wire and
may not import `speech_service`.
"""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator

WAV_HEADER_BYTES: Final = 44
SILENCE_RMS: Final = 0.01
# The full scale of int16 samples; the service's `pcm.INT16_SCALE` is the same
# number, which this module may not import.
_INT16_SCALE: Final = 32768.0
_SAMPLE_BYTES: Final = 2
_CHUNK_HEADER_BYTES: Final = 8


def riff_chunks(wav: bytes) -> Iterator[tuple[bytes, int, int]]:
    """The id, start and end of each RIFF chunk after the header, pad byte included."""
    position = 12
    while position + _CHUNK_HEADER_BYTES <= len(wav):
        size = int.from_bytes(wav[position + 4 : position + 8], "little")
        end = position + _CHUNK_HEADER_BYTES + size + size % 2
        yield wav[position : position + 4], position, end
        position = end


def pcm_from_wav(data: bytes) -> tuple[int, bytes]:
    """Sample rate and PCM payload of a 16-bit mono WAV, ignoring a streaming size."""
    if len(data) < WAV_HEADER_BYTES or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        message = "not a WAV body"
        raise ValueError(message)
    for chunk_id, start, _ in riff_chunks(data):
        if chunk_id == b"data":
            rate = int.from_bytes(data[24:28], "little")
            return rate, data[start + _CHUNK_HEADER_BYTES :]
    message = "WAV has no data chunk"
    raise ValueError(message)


def rms(pcm: bytes) -> float:
    """Root-mean-square of 16-bit samples, scaled to 0..1.

    An odd last byte is dropped.
    """
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % _SAMPLE_BYTES])
    if not samples:
        return 0.0
    total = sum(sample * sample for sample in samples)
    return (total / len(samples)) ** 0.5 / _INT16_SCALE
