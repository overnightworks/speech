"""16-bit PCM helpers for the wire: a streaming WAV header, level and resampling."""

from __future__ import annotations

import math
import struct
from typing import Final

import av
import numpy as np

BITS_PER_SAMPLE: Final = 16
BYTES_PER_SAMPLE: Final = BITS_PER_SAMPLE // 8
CHANNELS: Final = 1
PCM_FORMAT: Final = 1
WAV_HEADER_BYTES: Final = 44
# Streaming WAV: the length is not known when the first byte leaves.
STREAMING_DATA_SIZE: Final = 0xFFFFFFFF
_INT16_SCALE: Final = 32768.0
# The IPTC term id for synthetic audio; never fetched (AI Act Art. 50(2)).
AI_MARK: Final = (
    "DigitalSourceType=http://cv.iptc.org/newscodes/"  # NOSONAR(S5332) id, not fetched
    "digitalsourcetype/trainedAlgorithmicMedia"
)


def wav_header(sample_rate: int, data_bytes: int = STREAMING_DATA_SIZE) -> bytes:
    """A PCM WAV header for 16-bit mono at `sample_rate`, with the AI mark.

    The mark is a RIFF INFO comment between `fmt ` and `data`; `fmt ` stays at
    offset 12.
    """
    byte_rate = sample_rate * CHANNELS * BYTES_PER_SAMPLE
    block_align = CHANNELS * BYTES_PER_SAMPLE
    mark = _ai_mark_chunk()
    riff_size = (
        STREAMING_DATA_SIZE
        if data_bytes == STREAMING_DATA_SIZE
        else (36 + len(mark) + data_bytes) & 0xFFFFFFFF
    )
    format_chunk = struct.pack(
        "<4sI4s4sIHHIIHH",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        PCM_FORMAT,
        CHANNELS,
        sample_rate,
        byte_rate,
        block_align,
        BITS_PER_SAMPLE,
    )
    return format_chunk + mark + struct.pack("<4sI", b"data", data_bytes & 0xFFFFFFFF)


def _ai_mark_chunk() -> bytes:
    comment = AI_MARK.encode("ascii") + b"\0"
    comment += b"\0" * (len(comment) % 2)
    comment_chunk = struct.pack("<4sI", b"ICMT", len(AI_MARK) + 1) + comment
    return (
        struct.pack("<4sI4s", b"LIST", 4 + len(comment_chunk), b"INFO") + comment_chunk
    )


def pcm_from_wav(data: bytes) -> tuple[int, bytes]:
    """Sample rate and PCM payload of a 16-bit mono WAV, ignoring a streaming size."""
    if len(data) < WAV_HEADER_BYTES or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        message = "not a WAV body"
        raise ValueError(message)
    rate = int.from_bytes(data[24:28], "little")
    position = 12
    while position + 8 <= len(data):
        chunk_id = data[position : position + 4]
        chunk_size = int.from_bytes(data[position + 4 : position + 8], "little")
        payload_at = position + 8
        if chunk_id == b"data":
            return rate, data[payload_at:]
        if chunk_size == STREAMING_DATA_SIZE:
            message = "WAV chunk before data has no length"
            raise ValueError(message)
        position = payload_at + chunk_size
    message = "WAV has no data chunk"
    raise ValueError(message)


def _scaled_samples(pcm: bytes) -> np.ndarray:
    """The whole samples of `pcm` as float64 in -1..1; an odd last byte is dropped."""
    count = len(pcm) // BYTES_PER_SAMPLE
    return (
        np.frombuffer(pcm, dtype="<i2", count=count).astype(np.float64) / _INT16_SCALE
    )


def rms(pcm: bytes) -> float:
    """Root-mean-square of 16-bit samples, scaled to 0..1."""
    samples = _scaled_samples(pcm)
    if not samples.size:
        return 0.0
    return math.sqrt(float(np.mean(samples**2)))


LIVE_INPUT_RATE: Final = 24_000
LIVE_OUTPUT_RATE: Final = 16_000


class Resampler:
    """24 kHz to 16 kHz mono 16-bit PCM through the filter a recording passes.

    The filter keeps its state between frames, so any split of the input into
    frames gives the same output as the whole.
    """

    def __init__(self) -> None:
        """Start before the first sample."""
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=LIVE_OUTPUT_RATE
        )

    def feed(self, pcm: bytes) -> bytes:
        """The output samples that the input so far fully determines."""
        if not pcm:
            return b""
        samples = np.frombuffer(pcm, dtype="<i2").reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = LIVE_INPUT_RATE
        return b"".join(
            resampled.to_ndarray().tobytes()
            for resampled in self._resampler.resample(frame)
        )
