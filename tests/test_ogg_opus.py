"""How much audio of a sentence follows its marks in the Ogg Opus stream (WIRE-157)."""

from __future__ import annotations

import struct
from itertools import pairwise
from typing import TYPE_CHECKING, Final

import pytest

from speech_client.wire import SentenceMarks
from speech_service.ogg_opus import ogg_opus_chunks
from tests.support.fakes_support import sine_pcm

if TYPE_CHECKING:
    from collections.abc import Iterator

_OPUS_RATE: Final = 48_000
_OGG_HEADER_BYTES: Final = 27
_OGG_GRANULE_OFFSET: Final = 6
_OGG_SEGMENT_COUNT_OFFSET: Final = 26
_PRE_SKIP_OFFSET_IN_HEAD: Final = 10
_MAXIMUM_SECONDS_AFTER_MARKS: Final = 0.1


def _granule_positions(ogg: bytes) -> Iterator[int]:
    """The granule position of each complete page of `ogg`."""
    offset = 0
    while offset + _OGG_HEADER_BYTES <= len(ogg):
        segments = ogg[offset + _OGG_SEGMENT_COUNT_OFFSET]
        table = offset + _OGG_HEADER_BYTES
        length = _OGG_HEADER_BYTES + segments + sum(ogg[table : table + segments])
        if offset + length > len(ogg):
            return
        (position,) = struct.unpack_from("<q", ogg, offset + _OGG_GRANULE_OFFSET)
        yield position
        offset += length


def _decodable_seconds(ogg: bytes) -> float:
    """The audio the complete pages of `ogg` carry, by the last granule position."""
    head = ogg.find(b"OpusHead")
    (pre_skip,) = struct.unpack_from("<H", ogg, head + _PRE_SKIP_OFFSET_IN_HEAD)
    granule = max(_granule_positions(ogg), default=0)
    return max(0, granule - pre_skip) / _OPUS_RATE


@pytest.mark.parametrize("sentence_seconds", [0.3, 0.31, 1.0])
def test_at_most_a_tenth_of_a_second_of_a_sentence_follows_its_marks(
    sentence_seconds: float,
) -> None:
    rate = 22_050
    sentences = 3
    marks = SentenceMarks(words=(), sounds=None)
    sentence = sine_pcm(sentence_seconds, rate)
    chunks = [piece for _ in range(sentences) for piece in (sentence, marks)]

    sent = b""
    shortfalls = []
    for item in ogg_opus_chunks(chunks, rate, marked=True):
        if isinstance(item, SentenceMarks):
            spoken = (len(shortfalls) + 1) * sentence_seconds
            shortfalls.append(spoken - _decodable_seconds(sent))
        else:
            sent += item

    assert len(shortfalls) == sentences
    assert max(shortfalls) <= _MAXIMUM_SECONDS_AFTER_MARKS


def test_the_default_answer_keeps_pages_as_long_as_the_limit_after_marks() -> None:
    rate = 22_050
    ogg = b"".join(
        item
        for item in ogg_opus_chunks([sine_pcm(1.0, rate)], rate)
        if isinstance(item, bytes)
    )

    positions = list(_granule_positions(ogg))
    spans = [later - earlier for earlier, later in pairwise(positions[1:-1])]
    assert spans
    assert min(spans) >= _MAXIMUM_SECONDS_AFTER_MARKS * _OPUS_RATE
