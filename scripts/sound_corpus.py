"""The public and the operator's recordings the probe listens to.

Sound clips come from FSD50K and VocalSound, speech from FLEURS, each fetched by
pinned revision. The operator's recordings come from a directory he names.
"""

from __future__ import annotations

import csv
import functools
import io
import itertools
import json
import tarfile
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import soundfile
from huggingface_hub import HfFileSystem

from sound_taggers import SAMPLE_RATE, Samples, hub_file, read_audio
from speech_client.wire import Language

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

LONGEST_CLIP_SECONDS: Final = 10.0
SILENCE_SECONDS: Final = 2.0
SILENCE_NOISE: Final = 1e-4


class Sound(StrEnum):
    """The sounds the service would name."""

    LAUGHTER = "laughter"
    COUGH = "cough"
    THROAT_CLEARING = "throat clearing"
    SIGH = "sigh"
    SCREAM = "scream"
    CLAP = "clap"

    @property
    def audioset_labels(self) -> tuple[str, ...]:
        """The AudioSet labels that stand for the sound."""
        return {
            Sound.LAUGHTER: ("Laughter", "Giggle", "Snicker", "Belly laugh"),
            Sound.COUGH: ("Cough",),
            Sound.THROAT_CLEARING: ("Throat clearing",),
            Sound.SIGH: ("Sigh",),
            Sound.SCREAM: ("Screaming",),
            Sound.CLAP: ("Clapping", "Applause"),
        }[self]


SPEECH_REPOSITORY: Final = "google/fleurs"
SPEECH_REVISION: Final = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FSD50K_REPOSITORY: Final = "Fhrozen/FSD50k"
FSD50K_REVISION: Final = "ccf1acaff12f3f4a4c10052dddafbb6d22152c9b"
FSD50K_SOUNDS: Final = frozenset(Sound) - {Sound.THROAT_CLEARING}
FSD50K_LICENCES: Final = {
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
    "http://creativecommons.org/licenses/by-nc/3.0/": "CC-BY-NC-3.0",
    "http://creativecommons.org/licenses/sampling+/1.0/": "Sampling+-1.0",
}
VOCALSOUND_REPOSITORY: Final = "MahiA/VocalSound"
VOCALSOUND_REVISION: Final = "6b52ccd27a845b647b0fc15c273fff81b794d20c"
VOCALSOUND_SOUNDS: Final = frozenset(Sound) - {Sound.SCREAM, Sound.CLAP}
# The original dataset's licence; the card of the hub copy says MIT.
VOCALSOUND_LICENCE: Final = "CC-BY-SA-4.0"
OPERATOR_DIRECTORIES: Final = ("maep", "echo")


@dataclass(frozen=True, slots=True)
class Clip:
    """One public recording of a sound, 16 kHz mono, under its licence."""

    sound: Sound
    source: str
    licence: str
    samples: Samples


def _strided(names: Sequence[str], count: int) -> list[str]:
    """`count` names spread evenly over the sorted `names`, the same every run."""
    ordered = sorted(names)
    return ordered[:: max(1, len(ordered) // count)][:count]


def _fsd50k_clips(sound: Sound, count: int) -> list[Clip]:
    fetch = functools.partial(
        hub_file, FSD50K_REPOSITORY, revision=FSD50K_REVISION, kind="dataset"
    )
    with fetch("labels/eval.csv").open(encoding="utf-8", newline="") as source:
        names = [
            row["fname"]
            for row in csv.DictReader(source)
            if sound.audioset_labels[0] in row["labels"].split(",")
        ]
    information = json.loads(
        fetch("metadata/eval_clips_info_FSD50K.json").read_text("utf-8")
    )
    clips = []
    for name in _strided(names, count * 2):
        path = fetch(f"clips/eval/{name}.wav")
        if soundfile.info(path).duration <= LONGEST_CLIP_SECONDS:
            licence = FSD50K_LICENCES[information[name]["license"]]
            clips.append(Clip(sound, "FSD50K", licence, read_audio(path)))
    return clips[:count]


def _vocalsound_clips(
    bundle: zipfile.ZipFile, sound: Sound, count: int, cache: Path
) -> list[Clip]:
    with bundle.open("VocalSound/test.csv") as table:
        rows = csv.DictReader(io.TextIOWrapper(table, encoding="utf-8"))
        names = [r["path"] for r in rows if r["classname"] == sound.audioset_labels[0]]
    clips = []
    for name in _strided(names, count):
        path = cache / "vocalsound" / Path(name).name
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bundle.read(f"VocalSound/{name}"))
        clips.append(Clip(sound, "VocalSound", VOCALSOUND_LICENCE, read_audio(path)))
    return clips


def load_sound_clips(count: int, cache: Path) -> list[Clip]:
    """`count` public clips of each sound, shared over the sources that have it."""
    archive = HfFileSystem().open(
        f"datasets/{VOCALSOUND_REPOSITORY}/VocalSound.zip", revision=VOCALSOUND_REVISION
    )
    clips: list[Clip] = []
    with zipfile.ZipFile(archive) as bundle:
        for sound in Sound:
            both = sound in FSD50K_SOUNDS and sound in VOCALSOUND_SOUNDS
            share = -(-count // 2) if both else count
            if sound in FSD50K_SOUNDS:
                clips += _fsd50k_clips(sound, share)
            if sound in VOCALSOUND_SOUNDS:
                clips += _vocalsound_clips(bundle, sound, share, cache)
    return clips


@dataclass(frozen=True, slots=True)
class Sentence:
    """One read sentence of the public speech and its language."""

    samples: Samples
    transcript: str
    language: Language


def _sentences(corpus: str, language: Language) -> Iterator[Sentence]:
    fetch = functools.partial(
        hub_file,
        revision=SPEECH_REVISION,
        kind="dataset",
        repository=SPEECH_REPOSITORY,
    )
    table = fetch(name=f"data/{corpus}/dev.tsv").read_text("utf-8").splitlines()
    transcripts = {
        row[1]: row[2].strip() for row in (line.split("\t") for line in table)
    }
    with tarfile.open(fetch(name=f"data/{corpus}/audio/dev.tar.gz")) as bundle:
        for member in bundle:
            name = Path(member.name).name
            if member.isfile() and name in transcripts:
                samples = read_audio(bundle.extractfile(member).read())
                yield Sentence(samples, transcripts[name], language)


def speech_sentences() -> Iterator[Sentence]:
    """German and English read sentences, alternating, until both run out."""
    pairs = itertools.zip_longest(
        _sentences("de_de", Language.DE), _sentences("en_us", Language.EN)
    )
    return (sentence for pair in pairs for sentence in pair if sentence is not None)


def speech_with_silence(start_minutes: float, minutes: float) -> list[Samples]:
    """Segments of a sentence and the silence after it, `minutes` of them.

    The first `start_minutes` of the corpus are skipped, so that runs of a corpus
    too long for one run add up to the whole.
    """
    noise = np.random.default_rng(0)
    seconds, segments = 0.0, []
    for sentence in speech_sentences():
        silence = noise.normal(0.0, SILENCE_NOISE, int(SILENCE_SECONDS * SAMPLE_RATE))
        segment = np.concatenate([sentence.samples, silence]).astype(np.float32)
        seconds += len(segment) / SAMPLE_RATE
        if seconds > start_minutes * 60:
            segments.append(segment)
        if seconds >= (start_minutes + minutes) * 60:
            break
    return segments


@dataclass(frozen=True, slots=True)
class Corpus:
    """What a run listens to; the operator's recordings only when he gave them."""

    clips: Sequence[Clip]
    segments: Sequence[Samples]
    sentences: Sequence[Sentence]
    maep: Sequence[Samples] = ()
    echo: Sequence[Samples] = ()


def operator_recordings(directory: Path) -> tuple[list[Samples], list[Samples]]:
    """The recordings of both directories; a missing or empty one refuses the run."""
    found = []
    for name in OPERATOR_DIRECTORIES:
        files = sorted(p for p in (directory / name).glob("*") if p.is_file())
        if not files:
            message = (
                f"{directory / name} holds no recording: --operator-recordings needs"
                " audio files in maep (about 20 'määp') and in echo (the agent's"
                " voice through the loudspeaker)"
            )
            raise SystemExit(message)
        found.append([read_audio(path) for path in files])
    return found[0], found[1]
