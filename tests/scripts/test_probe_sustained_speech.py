"""The sustained-speech probe reads pinned archives and prints its table."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import struct
import sys
import tarfile
import wave
import zipfile
from argparse import Namespace
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

import sustained_figures as figures
from speech_service.voice_activity import WINDOW_SAMPLES
from sustained_figures import DelayCorpus, Sound
from tests.support.script_support import script_module

if TYPE_CHECKING:
    from pathlib import Path

PROBE = script_module("probe_sustained_speech")
NATIVE_RATE = 16_000
WIRE_WINDOW_SAMPLES = WINDOW_SAMPLES * 3 // 2
LOUD = 8000
VOICED_LEVEL = 1000
MEETING = "M1"
NITE = 'xmlns:nite="http://nite.sourceforge.net/"'
CLIPS_PER_SOUND = 2
SMALL_MINIMUMS = {
    Sound.BACKCHANNEL: 2,
    Sound.LAUGHTER: 2,
    Sound.COUGH: 2,
    Sound.THROAT_CLEARING: 2,
    DelayCorpus.AMI: 1,
    DelayCorpus.FLEURS: 1,
}


def fake_probabilities(pcm: bytes) -> list[float]:
    """A detector that rates a 24 kHz window voiced when its mean level is high."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    whole = len(samples) // WIRE_WINDOW_SAMPLES * WIRE_WINDOW_SAMPLES
    levels = np.abs(samples[:whole]).reshape(-1, WIRE_WINDOW_SAMPLES).mean(axis=1)
    return [0.9 if level >= VOICED_LEVEL else 0.0 for level in levels]


def tone(seconds: float, loud: list[tuple[float, float]]) -> np.ndarray:
    """Silence of `seconds` with loud stretches (start, end), at the native rate."""
    samples = np.zeros(round(seconds * NATIVE_RATE), dtype="<i2")
    for start, end in loud:
        samples[round(start * NATIVE_RATE) : round(end * NATIVE_RATE)] = LOUD
    return samples


def pcm_wav(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(NATIVE_RATE)
        writer.writeframes(samples.tobytes())
    return buffer.getvalue()


def float_wav(samples: np.ndarray) -> bytes:
    payload = (samples.astype("<f4") / 32768).tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(payload),
        b"WAVE",
        b"fmt ",
        16,
        3,
        1,
        NATIVE_RATE,
        NATIVE_RATE * 4,
        4,
        32,
        b"data",
        len(payload),
    )
    return header + payload


def word(number: int, start: float, end: float, text: str = "mhm") -> str:
    return (
        f'<w nite:id="{MEETING}.A.words{number}" starttime="{start}"'
        f' endtime="{end}">{text}</w>'
    )


def punctuation(number: int, at: float) -> str:
    return (
        f'<w nite:id="{MEETING}.A.words{number}" starttime="{at}" endtime="{at}"'
        ' punc="true">.</w>'
    )


def disfluency_marker(number: int, at: float) -> str:
    return (
        f'<disfmarker nite:id="{MEETING}.A.words{number}" starttime="{at}"'
        f' endtime="{at}"/>'
    )


def vocal_sound(number: int, start: float, end: float) -> str:
    return (
        f'<vocalsound nite:id="{MEETING}.A.words{number}" starttime="{start}"'
        f' endtime="{end}" type="laugh"/>'
    )


def act(kind: str, first: int, last: int) -> str:
    return (
        f'<dact nite:id="act{first}"><nite:pointer role="da-aspect"'
        f' href="da-types.xml#id({kind})"/><nite:child'
        f' href="{MEETING}.A.words.xml#id({MEETING}.A.words{first})'
        f'..id({MEETING}.A.words{last})"/></dact>'
    )


def dialogue_acts() -> str:
    return "".join(
        [
            act("ami_da_4", 0, 1),
            act("ami_da_1", 3, 4),
            act("ami_da_4", 6, 7),
            act("ami_da_1", 8, 8),
            act("ami_da_2", 9, 10),
        ]
    )


def annotations() -> bytes:
    """A wearer who says "mhm" alone at 30 s and 50 s and speaks on from 10 s, 45 s.

    The speech at 55 s is a stall (a minor act), the other speech a task act.
    """
    words = "".join(
        [
            word(0, 10.0, 10.5, "hello"),
            word(1, 10.5, 11.75, "there"),
            vocal_sound(2, 20.0, 20.25),
            word(3, 30.0, 30.5),
            punctuation(4, 30.5),
            disfluency_marker(5, 40.0),
            word(6, 45.0, 45.5, "well"),
            word(7, 45.5, 46.75, "then"),
            word(8, 50.0, 50.5),
            word(9, 55.0, 55.5, "so"),
            word(10, 55.5, 56.75, "yes"),
        ]
    )

    files = {
        "corpusResources/meetings.xml": (
            f'<nite:root {NITE}><meeting observation="{MEETING}">'
            '<speaker channel="0" nxt_agent="A"/>'
            '<speaker channel="1" nxt_agent="B"/></meeting></nite:root>'
        ),
        "ontologies/da-types.xml": (
            f'<da-type {NITE} nite:id="root">'
            '<da-type name="minor"><da-type nite:id="ami_da_1" name="bck"/>'
            '<da-type nite:id="ami_da_2" name="stl"/></da-type>'
            '<da-type name="task"><da-type nite:id="ami_da_4" name="inf"/></da-type>'
            "</da-type>"
        ),
        f"words/{MEETING}.A.words.xml": f"<nite:root {NITE}>{words}</nite:root>",
        f"dialogueActs/{MEETING}.A.dialog-act.xml": (
            f"<nite:root {NITE}>{dialogue_acts()}</nite:root>"
        ),
        f"segments/{MEETING}.B.segments.xml": (
            f'<nite:root {NITE}><segment transcriber_start="30.25"'
            ' transcriber_end="31.0"/></nite:root>'
        ),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def headset() -> bytes:
    return pcm_wav(
        tone(
            60.0,
            [
                (10.0, 11.75),
                (30.0, 30.5),
                (43.0, 43.25),
                (45.0, 46.5),
                (50.0, 50.5),
                (55.0, 56.5),
            ],
        )
    )


def vocalsound_shard() -> bytes:
    """Two clips each of three sounds, a third throat clearing too long, a sniff."""
    clips = {
        "m1_0_laughter": tone(2.5, [(0.7, 0.9)]),
        "m2_0_laughter": tone(2.5, [(0.7, 1.0)]),
        "m1_0_cough": tone(2.5, [(0.7, 0.9)]),
        "m2_0_cough": tone(2.5, [(0.7, 1.0)]),
        "m1_0_throatclearing": tone(2.5, [(0.7, 0.9)]),
        "m2_0_throatclearing": tone(2.5, [(0.7, 1.0)]),
        "m3_0_throatclearing": tone(2.5, [(0.7, 2.4)]),
        "m3_0_sniff": tone(2.5, [(0.7, 0.9)]),
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for stem, samples in clips.items():
            label = stem.rpartition("_")[2]
            for name, content in (
                (f"{stem}.json", json.dumps({"label": label}).encode()),
                (f"{stem}.wav", pcm_wav(samples)),
            ):
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


def fleurs_split() -> bytes:
    """German clips: one after a second of silence, one at once, one silent."""
    clips = {
        "dev/1.wav": tone(5.0, [(1.0, 4.0)]),
        "dev/2.wav": tone(5.0, [(0.1, 3.0)]),
        "dev/3.wav": tone(5.0, []),
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        directory = tarfile.TarInfo("dev")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for name, samples in clips.items():
            content = float_wav(samples)
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return gzip.compress(buffer.getvalue())


def pinned_in(cache: Path, name: str, content: bytes) -> object:
    (cache / name).write_bytes(content)
    return PROBE.Pinned(name, "https://example.invalid/" + name, sha256_of(content))


def sha256_of(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


@pytest.fixture
def sources(cache: Path) -> object:
    """The probe's sources, replaced by tiny archives that are already cached."""
    headset_audio = pinned_in(cache, f"{MEETING}.Headset-0.wav", headset())
    return PROBE.Sources(
        ami_annotations=pinned_in(cache, "annotations.zip", annotations()),
        ami_headsets=(PROBE.AmiHeadset(MEETING, 0, headset_audio),),
        ami_extra_headsets=(),
        vocalsound_shards=(pinned_in(cache, "shard.tar", vocalsound_shard()),),
        vocalsound_extra_shards=(),
        fleurs=pinned_in(cache, "fleurs.tar.gz", fleurs_split()),
        fleurs_extra=pinned_in(cache, "fleurs_test.tar.gz", fleurs_split()),
    )


def run(
    cache: Path, sources: object, capsys: pytest.CaptureFixture[str]
) -> tuple[dict[str, object], int | None]:
    """The probe's printed table and its exit code."""
    exit_code = None
    try:
        PROBE.probe(
            Namespace(cache=cache),
            sources=sources,
            probabilities=fake_probabilities,
        )
    except SystemExit as stop:
        exit_code = stop.code
    return json.loads(capsys.readouterr().out), exit_code


def test_the_probe_prints_a_row_for_every_cell_with_the_counts_of_what_it_read(
    monkeypatch: pytest.MonkeyPatch,
    cache: Path,
    sources: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(figures, "MINIMUM_KEPT", SMALL_MINIMUMS)

    table, exit_code = run(cache, sources, capsys)

    assert exit_code is None
    assert table["missed"] == []
    assert len(table["cells"]) == len(figures.GRID)
    counts = table["counts"]
    assert counts[Sound.BACKCHANNEL]["kept"] == CLIPS_PER_SOUND
    assert counts[DelayCorpus.AMI]["kept"] == 1
    assert counts[DelayCorpus.AMI]["dropped"] == {"not isolated": 1, "minor act": 1}
    assert counts[Sound.LAUGHTER]["kept"] == CLIPS_PER_SOUND
    assert counts[Sound.THROAT_CLEARING]["dropped"] == {"too long": 1}
    assert counts[DelayCorpus.FLEURS]["dropped"] == {"no lead-in": 1}
    assert counts[DelayCorpus.FLEURS]["kept"] == 1
    assert "sniff" not in json.dumps(counts)
    ruled = next(
        cell
        for cell in table["cells"]
        if (cell["duration_windows"], cell["restart_run"]) == (13, 5)
    )
    assert set(ruled["false_holds"]) == {
        "backchannel",
        "backchannel pair",
        "laughter",
        "laughter pair",
        "cough",
        "cough pair",
        "throatclearing",
        "throatclearing pair",
    }
    assert ruled["backchannel_by_label"].keys() == {"alone", "overlapped"}
    assert ruled["delay"]["ami english"]["count"] == 1
    assert ruled["delay"]["fleurs german"]["count"] == 1


def test_a_delay_row_counts_the_windows_of_the_duration_from_the_onset(
    monkeypatch: pytest.MonkeyPatch,
    cache: Path,
    sources: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(figures, "MINIMUM_KEPT", SMALL_MINIMUMS)

    table, _ = run(cache, sources, capsys)

    by_duration = {
        cell["duration_windows"]: cell["delay"]["ami english"]["median"]
        for cell in table["cells"]
        if cell["restart_run"] == figures.RULED_CELL.restart_run
    }
    for duration, delay in by_duration.items():
        assert delay == pytest.approx(
            duration * figures.WINDOW_SECONDS, abs=figures.WINDOW_SECONDS
        )


def test_a_minimum_that_is_missed_ends_the_run_with_exit_3_naming_the_group(
    cache: Path, sources: object, capsys: pytest.CaptureFixture[str]
) -> None:
    table, exit_code = run(cache, sources, capsys)

    assert exit_code == PROBE.MISSED_EXIT
    assert Sound.BACKCHANNEL in table["missed"]
    assert DelayCorpus.FLEURS in table["missed"]


def test_an_extra_file_is_read_only_while_a_minimum_is_missed(
    monkeypatch: pytest.MonkeyPatch,
    cache: Path,
    sources: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(figures, "MINIMUM_KEPT", SMALL_MINIMUMS)
    extra = PROBE.Pinned("extra.tar", "https://example.invalid", None)
    with_extra = replace(sources, vocalsound_extra_shards=(extra,))

    table, exit_code = run(cache, with_extra, capsys)

    assert exit_code is None
    assert {pinned["name"] for pinned in table["corpora"][1]["files"]} == {"shard.tar"}


def test_an_extra_file_without_a_checksum_is_refused_when_a_minimum_needs_it(
    cache: Path, sources: object
) -> None:
    extra = PROBE.Pinned("extra.tar", "https://example.invalid", None)
    with_extra = replace(sources, vocalsound_extra_shards=(extra,))

    with pytest.raises(SystemExit, match="no pinned sha256"):
        PROBE.collect(cache, with_extra, fake_probabilities)


def test_a_cached_file_that_is_not_the_pin_is_refused(cache: Path) -> None:
    pinned = PROBE.Pinned("file.bin", "https://example.invalid", sha256_of(b"pinned"))
    (cache / "file.bin").write_bytes(b"tampered")

    with pytest.raises(SystemExit, match="is not the pinned"):
        PROBE.fetch(cache, pinned)


def test_an_annotation_that_declares_an_entity_is_refused() -> None:
    bomb = b'<!DOCTYPE x [<!ENTITY a "aaaa">]><x>&a;</x>'

    with pytest.raises(SystemExit, match="DOCTYPE or an ENTITY"):
        PROBE.parse_xml(bomb)


@pytest.mark.parametrize(
    ("wav", "expected"),
    [
        pytest.param(pcm_wav(tone(0.1, [(0.0, 0.1)])), LOUD, id="pcm16"),
        pytest.param(float_wav(tone(0.1, [(0.0, 0.1)])), LOUD, id="float32"),
    ],
)
def test_a_wav_of_either_supported_format_decodes_to_pcm16(
    wav: bytes, expected: int
) -> None:
    rate, samples = PROBE.decode_wav(wav)

    assert rate == NATIVE_RATE
    assert samples.dtype == np.dtype("<i2")
    assert set(samples.tolist()) == {expected}


def test_a_wav_with_two_channels_is_refused() -> None:
    stereo = bytearray(pcm_wav(tone(0.1, [])))
    stereo[22:24] = struct.pack("<H", 2)
    wav = bytes(stereo)

    with pytest.raises(ValueError, match="2 channels"):
        PROBE.decode_wav(wav)


def test_a_file_that_is_not_cached_is_downloaded_into_the_cache_once(
    cache: Path, tmp_path: Path
) -> None:
    origin = tmp_path / "origin.bin"
    origin.write_bytes(b"corpus")
    pinned = PROBE.Pinned("file.bin", origin.as_uri(), sha256_of(b"corpus"))

    first = PROBE.fetch(cache, pinned)
    origin.unlink()
    second = PROBE.fetch(cache, pinned)

    assert first == second == b"corpus"
    assert [path.name for path in cache.iterdir()] == ["file.bin"]


def test_a_wav_of_a_format_the_probe_does_not_read_is_refused() -> None:
    ulaw = bytearray(pcm_wav(tone(0.1, [])))
    ulaw[20:22] = struct.pack("<H", 7)
    wav = bytes(ulaw)

    with pytest.raises(ValueError, match="format 7"):
        PROBE.decode_wav(wav)


def test_a_tar_member_outside_its_directory_is_refused() -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    archive_bytes = buffer.getvalue()

    with pytest.raises(tarfile.FilterError):
        list(PROBE.tar_members(archive_bytes, "r:"))


def test_the_command_line_names_the_cache_and_the_probe_runs_over_it(
    monkeypatch: pytest.MonkeyPatch, cache: Path
) -> None:
    ran = []
    monkeypatch.setattr(sys, "argv", ["probe_sustained_speech", "--cache", str(cache)])
    monkeypatch.setattr(PROBE, "probe", ran.append)

    PROBE.main()

    assert [arguments.cache for arguments in ran] == [cache]


def test_the_command_line_refuses_to_run_without_a_cache() -> None:
    with pytest.raises(SystemExit):
        PROBE.parse_arguments([])


def test_the_command_line_refuses_a_cache_that_is_not_a_directory(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    arguments = ["--cache", str(missing)]

    with pytest.raises(SystemExit):
        PROBE.parse_arguments(arguments)

    assert not missing.exists()
