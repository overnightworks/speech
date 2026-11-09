"""Ogg Opus from the voice's 16-bit mono PCM, encoded as it is spoken."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import av
import numpy as np

from speech_service.pcm import AI_MARK

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from av.audio.frame import AudioFrame
    from av.audio.stream import AudioStream
    from av.container import OutputContainer

_OPUS_RATE: Final = 48_000
# The muxer holds pages for about 2 s unless told otherwise.
_PAGE_DURATION_MICROSECONDS: Final = "100000"
_PCM_FORMAT: Final = "s16"
_MONO: Final = "mono"


class _Sink:
    """The muxer's output: a write-only buffer, so the muxer cannot seek."""

    def __init__(self) -> None:
        self._written = bytearray()

    def write(self, data: bytes) -> int:
        self._written += data
        return len(data)

    def take(self) -> bytes:
        """What was written since the last call."""
        taken = bytes(self._written)
        self._written.clear()
        return taken


def ogg_opus_chunks(pcm_chunks: Iterable[bytes], sample_rate: int) -> Iterator[bytes]:
    """One Ogg Opus stream from `pcm_chunks` at `sample_rate`, mono at 48 kHz.

    Pages leave as the voice speaks. The end page is written only once every
    chunk was encoded; a failure or a departure closes the container and
    leaves what the close writes unread, so nothing follows the audio already sent.
    """
    sink = _Sink()
    container = av.open(
        sink, "w", format="ogg", options={"page_duration": _PAGE_DURATION_MICROSECONDS}
    )
    complete = False
    try:
        stream = container.add_stream("libopus", rate=_OPUS_RATE, layout=_MONO)
        stream.metadata["comment"] = AI_MARK
        resampler = av.AudioResampler(format=_PCM_FORMAT, layout=_MONO, rate=_OPUS_RATE)
        for pcm in pcm_chunks:
            _encode(container, stream, resampler.resample(_frame(pcm, sample_rate)))
            if pages := sink.take():
                yield pages
        _encode(container, stream, [*resampler.resample(None), None])
        complete = True
        container.close()
        yield sink.take()
    finally:
        if not complete:
            # The close writes the end page into the sink; nothing reads it again.
            container.close()


def _frame(pcm: bytes, sample_rate: int) -> AudioFrame:
    samples = np.frombuffer(pcm, dtype="<i2").reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(samples, format=_PCM_FORMAT, layout=_MONO)
    frame.sample_rate = sample_rate
    return frame


def _encode(
    container: OutputContainer,
    stream: AudioStream,
    frames: Iterable[AudioFrame | None],
) -> None:
    for frame in frames:
        for packet in stream.encode(frame):
            container.mux(packet)
