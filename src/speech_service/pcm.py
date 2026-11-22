"""16-bit PCM helpers for the wire: a streaming WAV header, level and resampling."""

from __future__ import annotations

import struct
from typing import Final

import av
import numpy as np

from speech_client.realtime_wire import PcmFormat, SessionInput

BYTES_PER_SAMPLE: Final = PcmFormat.sample_bytes
BITS_PER_SAMPLE: Final = 8 * BYTES_PER_SAMPLE
CHANNELS: Final = PcmFormat.channels
AV_SAMPLE_FORMAT: Final = "s16"
AV_LAYOUT: Final = "mono"
PCM_FORMAT: Final = 1
# Streaming WAV: the length is not known when the first byte leaves.
STREAMING_DATA_SIZE: Final = 0xFFFFFFFF
INT16_SCALE: Final = 32768.0
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


LIVE_INPUT_RATE: Final = SessionInput().format.rate
# The rate Whisper, Silero and the sound model take.
HEARING_SAMPLE_RATE: Final = 16_000


class Resampler:
    """24 kHz to 16 kHz mono 16-bit PCM through the filter a recording passes.

    The filter keeps its state between frames, so any split of the input into
    frames gives the same output as the whole.
    """

    def __init__(self) -> None:
        """Start before the first sample."""
        self._resampler = av.AudioResampler(
            format=AV_SAMPLE_FORMAT, layout=AV_LAYOUT, rate=HEARING_SAMPLE_RATE
        )

    def feed(self, pcm: bytes) -> bytes:
        """The output samples that the input so far fully determines."""
        if not pcm:
            return b""
        samples = np.frombuffer(pcm, dtype="<i2").reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(
            samples, format=AV_SAMPLE_FORMAT, layout=AV_LAYOUT
        )
        frame.sample_rate = LIVE_INPUT_RATE
        return b"".join(
            resampled.to_ndarray().tobytes()
            for resampled in self._resampler.resample(frame)
        )
