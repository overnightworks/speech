"""Ogg Opus from the voice's 16-bit mono PCM, encoded as it is spoken."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import av
import numpy as np

from speech_client.wire import SentenceMarks
from speech_service.pcm import AI_MARK, AV_LAYOUT, AV_SAMPLE_FORMAT

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from av.audio.frame import AudioFrame
    from av.audio.stream import AudioStream
    from av.container import OutputContainer

_OPUS_RATE: Final = 48_000
# The muxer holds pages for about 2 s unless told otherwise.
_PAGE_DURATION_MICROSECONDS: Final = "100000"
# The muxer writes a page only once the next packet would overfill it, so up to two
# pages of audio are unsent; at 40 ms that stays under the 100 ms a sentence's
# marks may lead its audio.
_MARKED_PAGE_DURATION_MICROSECONDS: Final = "40000"


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


def ogg_opus_chunks(
    pcm_chunks: Iterable[bytes | SentenceMarks],
    sample_rate: int,
    *,
    marked: bool = False,
) -> Iterator[bytes | SentenceMarks]:
    """One Ogg Opus stream from `pcm_chunks` at `sample_rate`, mono at 48 kHz.

    A `SentenceMarks` among the chunks passes through where it stands, so it follows
    the pages of the audio before it. With `marked` the pages are short enough
    that less than 100 ms of that audio is still held back when it passes.
    Pages leave as the voice speaks. The end page is written only once every
    chunk was encoded; a failure or a departure closes the container and
    leaves what the close writes unread, so nothing follows the audio already sent.
    """
    sink = _Sink()
    page_duration = (
        _MARKED_PAGE_DURATION_MICROSECONDS if marked else _PAGE_DURATION_MICROSECONDS
    )
    container = av.open(
        sink, "w", format="ogg", options={"page_duration": page_duration}
    )
    complete = False
    try:
        stream = container.add_stream("libopus", rate=_OPUS_RATE, layout=AV_LAYOUT)
        stream.metadata["comment"] = AI_MARK
        resampler = av.AudioResampler(
            format=AV_SAMPLE_FORMAT, layout=AV_LAYOUT, rate=_OPUS_RATE
        )
        for pcm in pcm_chunks:
            if isinstance(pcm, SentenceMarks):
                yield pcm
                continue
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
    frame = av.AudioFrame.from_ndarray(
        samples, format=AV_SAMPLE_FORMAT, layout=AV_LAYOUT
    )
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
