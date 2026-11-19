"""The 24 kHz to 16 kHz resampler keeps its state across frame boundaries."""

import io
import itertools
import math
import struct
import wave

import pytest

from speech_service.pcm import Resampler
from speech_service.recording import decode_recording
from tests.support.fakes_support import sine_pcm

INPUT_RATE = 24_000
OUTPUT_RATE = 16_000


def _samples(pcm: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(pcm) // 2}h", pcm))


def _in_frames(pcm: bytes, sizes: list[int]) -> bytes:
    resampler = Resampler()
    out = b""
    position = 0
    for size in sizes:
        out += resampler.feed(pcm[position : position + size])
        position += size
    return out + resampler.feed(pcm[position:])


def test_a_tone_split_across_frames_is_resampled_as_if_it_came_in_one() -> None:
    tone = sine_pcm(0.5, INPUT_RATE)

    whole = Resampler().feed(tone)

    assert _in_frames(tone, [2, 6, 1000, 2]) == whole


@pytest.mark.parametrize("split", [2, 4, 6, 14, 962, 2400])
def test_any_split_of_the_input_gives_the_same_output(split: int) -> None:
    tone = sine_pcm(0.2, INPUT_RATE)

    assert _in_frames(tone, [split]) == Resampler().feed(tone)


def test_an_empty_chunk_gives_no_output_and_keeps_the_stream_going() -> None:
    resampler = Resampler()
    tone = sine_pcm(0.2, INPUT_RATE)

    assert resampler.feed(b"") == b""
    assert resampler.feed(tone) == Resampler().feed(tone)


def test_two_thirds_as_many_samples_leave_as_came_in_less_the_filter_delay() -> None:
    out = Resampler().feed(sine_pcm(1.0, INPUT_RATE))

    assert OUTPUT_RATE - FILTER_DELAY_SAMPLES <= len(out) // 2 <= OUTPUT_RATE


def test_a_constant_level_stays_that_level() -> None:
    level = 1234
    out = Resampler().feed(struct.pack("<300h", *[level] * 300))

    assert set(_samples(out)) == {level}


def test_a_tone_keeps_its_pitch() -> None:
    out = _samples(Resampler().feed(sine_pcm(1.0, INPUT_RATE, freq=440.0)))

    rising = sum(1 for a, b in itertools.pairwise(out) if a < 0 <= b)
    assert abs(rising - 440) <= 1


ODD_CHUNKS = [7, 13, 1001, 3, 4999]
EDGE_SAMPLES = 400
FILTER_DELAY_SAMPLES = 32
NO_LIMIT_SECONDS = 60.0


def _as_recording(pcm: bytes) -> bytes:
    """`pcm` at the live input rate as a WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(INPUT_RATE)
        wav.writeframes(pcm)
    return buffer.getvalue()


def _live(pcm: bytes) -> bytes:
    resampler = Resampler()
    out = b""
    position = 0
    for size in itertools.cycle(ODD_CHUNKS):
        if position >= len(pcm):
            return out
        out += resampler.feed(pcm[position : position + 2 * size])
        position += 2 * size
    return out


def _level(pcm: bytes) -> float:
    samples = _samples(pcm)[EDGE_SAMPLES:-EDGE_SAMPLES]
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


@pytest.mark.parametrize("freq", [9_000.0, 11_000.0])
def test_a_tone_above_the_output_band_is_no_louder_than_in_a_recording(
    freq: float,
) -> None:
    tone = sine_pcm(1.0, INPUT_RATE, freq=freq)

    recorded = decode_recording(_as_recording(tone), NO_LIMIT_SECONDS)

    assert _level(_live(tone)) <= _level(recorded) + 1


def test_odd_chunks_lose_no_sample_against_the_whole() -> None:
    tone = sine_pcm(1.0, INPUT_RATE)

    assert len(_live(tone)) == len(Resampler().feed(tone))
