"""The Ogg Opus stream of a spoken answer: how much audio follows a sentence's marks (WIRE-157)."""

from __future__ import annotations

import struct
from typing import Final

import pytest

from speech_client.wire import SentenceMarks
from speech_service.ogg_opus import ogg_opus_chunks
from tests.support.fakes_support import sine_pcm

_RATE: Final = 22_050
_OPUS_RATE: Final = 48_000
_OGG_HEADER_BYTES: Final = 27
_OGG_GRANULE_OFFSET: Final = 6
_OGG_SEGMENT_COUNT_OFFSET: Final = 26
_PRE_SKIP_OFFSET_IN_HEAD: Final = 10
_MAXIMUM_SECONDS_AFTER_MARKS: Final = 0.1
_SENTENCES: Final = 3


def _decodable_seconds(ogg: bytes) -> float:
    """The audio the complete pages of `ogg` carry, by the last granule position."""
    head = ogg.find(b"OpusHead")
    (pre_skip,) = struct.unpack_from("<H", ogg, head + _PRE_SKIP_OFFSET_IN_HEAD)
    granule = 0
    offset = 0
    while offset + _OGG_HEADER_BYTES <= len(ogg):
        segments = ogg[offset + _OGG_SEGMENT_COUNT_OFFSET]
        table = offset + _OGG_HEADER_BYTES
        length = _OGG_HEADER_BYTES + segments + sum(ogg[table : table + segments])
        if offset + length > len(ogg):
            break
        (position,) = struct.unpack_from("<q", ogg, offset + _OGG_GRANULE_OFFSET)
        granule = max(granule, position)
        offset += length
    return max(0, granule - pre_skip) / _OPUS_RATE


@pytest.mark.parametrize("sentence_seconds", [0.3, 1.0])
def test_at_most_a_tenth_of_a_second_of_a_sentence_follows_its_marks(
    sentence_seconds: float,
) -> None:
    marks = SentenceMarks(words=(), sounds=None)
    sentence = sine_pcm(sentence_seconds, _RATE)
    chunks = [piece for _ in range(_SENTENCES) for piece in (sentence, marks)]

    sent = b""
    shortfalls = []
    for item in ogg_opus_chunks(chunks, _RATE):
        if isinstance(item, SentenceMarks):
            spoken = (len(shortfalls) + 1) * sentence_seconds
            shortfalls.append(spoken - _decodable_seconds(sent))
        else:
            sent += item

    assert len(shortfalls) == _SENTENCES
    assert max(shortfalls) <= _MAXIMUM_SECONDS_AFTER_MARKS
