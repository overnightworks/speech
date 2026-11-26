"""The scripts' one reader of a WAV body: its chunks, its PCM and its level.

Standard library and `speech_client` only: `contract_client` is an outside
caller of the wire and may not import `speech_service`.
"""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Final

from speech_client.realtime_wire import PcmFormat
from speech_client.wire import ResponseFormat

if TYPE_CHECKING:
    from collections.abc import Iterator

WAV_HEADER_BYTES: Final = 44
SILENCE_RMS: Final = 0.01
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
    if len(data) < WAV_HEADER_BYTES or not ResponseFormat.WAV.begins_like_container(
        data
    ):
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
    samples.frombytes(pcm[: len(pcm) - len(pcm) % PcmFormat.sample_bytes])
    if not samples:
        return 0.0
    total = sum(sample * sample for sample in samples)
    return (total / len(samples)) ** 0.5 / PcmFormat.full_scale
