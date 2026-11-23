"""The scripts' WAV reader gives the rate, the PCM and the level of a body."""

from __future__ import annotations

import struct

import pytest

from speech_service.pcm import STREAMING_DATA_SIZE, wav_header
from wav_audio import pcm_from_wav, rms

RATE = 24000
PCM = b"\x01\x02\x03\x04"


def _chunk(chunk_id: bytes, size: int, payload: bytes) -> bytes:
    return struct.pack("<4sI", chunk_id, size) + payload


def _after_header_before_data(*chunks: bytes) -> bytes:
    header = wav_header(RATE)
    return header[:-8] + b"".join(chunks) + header[-8:]


def test_a_marked_service_header_gives_the_rate_and_the_pcm() -> None:
    assert pcm_from_wav(wav_header(RATE) + PCM) == (RATE, PCM)


def test_an_odd_sized_chunk_before_data_is_skipped_with_its_pad_byte() -> None:
    wav = _after_header_before_data(_chunk(b"junk", 3, b"abc\0")) + PCM

    assert pcm_from_wav(wav) == (RATE, PCM)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"x" * 100, id="not a wav"),
        pytest.param(wav_header(RATE)[:20], id="shorter than the header"),
        pytest.param(wav_header(RATE)[:-8] + b"\0" * 8, id="no data chunk"),
        pytest.param(
            _after_header_before_data(_chunk(b"junk", STREAMING_DATA_SIZE, b"")) + PCM,
            id="streaming size before data",
        ),
    ],
)
def test_a_body_without_pcm_is_refused(body: bytes) -> None:
    with pytest.raises(ValueError, match=r"WAV"):
        pcm_from_wav(body)


def test_the_level_of_nothing_is_zero() -> None:
    assert rms(b"") == 0.0


def test_an_odd_last_byte_does_not_change_the_level() -> None:
    assert rms(b"\x00\x40\x00\x40\x7f") == rms(b"\x00\x40\x00\x40")


def test_a_square_wave_at_a_quarter_of_full_scale_has_that_level() -> None:
    quarter = struct.pack("<hh", 8192, -8192) * 4

    assert rms(quarter) == pytest.approx(0.25)
