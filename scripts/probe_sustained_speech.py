"""Measure the sustained-speech duration and pause bound on real recordings.

The probe runs offline on the CPU: no service, no card. It fetches pinned corpora
into `--cache` (each file at its sha256, verified before it is read; archives are
read in memory and never extracted), takes the speech probabilities of every
excerpt once from the shipped Silero, and replays them through the real
`HearingSession` for every cell of the grid in `sustained_figures`. It prints one
JSON: the corpora with their licences, revisions, checksums and counts, and for
each cell the false holds by sound and by pair, and the delay rows. Exit 3 when a
minimum of kept excerpts is missed after every pinned extra.

The measured delay is the service's count only: transport and the stop of the
playback come on top of the card's half second. Re-run it when the shipped
`silero_vad_v6.onnx` or `START_THRESHOLD` / `END_THRESHOLD` change.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import struct
import sys
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final
from xml.etree import ElementTree as ET

import numpy as np

import sustained_figures as figures
from build_sound_model import download_pinned
from hearing_figures import speech_probabilities, to_wire_rate
from speech_service.voice_activity import WINDOW_SECONDS, is_voiced
from sustained_figures import (
    Backchannel,
    DelayCorpus,
    DelayTrial,
    Fate,
    HeardClip,
    HoldTrial,
    Segment,
    Sound,
    Tally,
    Timed,
)
from wav_audio import pcm_from_wav

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from typing import TextIO

MISSED_EXIT: Final = 3
NITE_ID: Final = "{http://nite.sourceforge.net/}id"
_FLOAT_WAV_FORMAT: Final = 3
_PCM_WAV_FORMAT: Final = 1
_FIXED_WAV_FIELDS: Final = (("<HH", 20), ("<H", 34))
_FLOAT_BITS: Final = 32
_PCM_BITS: Final = 16
_VOCALSOUND_LABELS: Final = (Sound.LAUGHTER, Sound.COUGH, Sound.THROAT_CLEARING)

AMI_ANNOTATIONS_URL: Final = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
)
AMI_AUDIO_URL: Final = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
    "{meeting}/audio/{meeting}.Headset-{channel}.wav"
)
VOCALSOUND_URL: Final = "https://zenodo.org/api/records/14650192/files/{name}/content"
# The shards of the record, each at the sha256 of its first fetch (cross-checked
# against the record's md5); a shard not listed is named but never read.
VOCALSOUND_SHA256: Final = {
    "wds-audio-test-000000.tar": (
        "403a4bfdb265fc4bb3d9658c15af951e0feb37706e6e1a227d1b3545b483e03e"
    ),
    "wds-audio-test-000001.tar": (
        "7e48be006c2cce156a05dc2c939912d3b01802033d10cf5b14d0e05f8e5a0e58"
    ),
    "wds-audio-test-000002.tar": (
        "43242a7dfb04fef8458fa5e5040b41cdb89f8966ea08156e29867496d653bf41"
    ),
    "wds-audio-test-000003.tar": (
        "7b47e62cc45439490a0e92a537aa5817ddb45ade0a84a5a4533ef1495d87eb58"
    ),
    "wds-audio-test-000004.tar": (
        "6c01d25cc973a324428b414d34869e7900d030e82898bc9fe0466bc0ca1f50ac"
    ),
    "wds-audio-test-000005.tar": (
        "1fb85fc7e8e078d2b13b4f0173435362214368ddf384f304a0534f144268e14f"
    ),
    "wds-audio-valid-000000.tar": (
        "af07ebc9036b86e4112f9df824fe51971e42183ac118c00cabad88cff924b5c0"
    ),
    "wds-audio-valid-000001.tar": (
        "1c18cff6b4e788d147838d328e6618253e3789ab3c2c3300114697b3432c8f63"
    ),
    "wds-audio-valid-000002.tar": (
        "01e3b5ce0125d445e32007311b2bb88978737571d833f1ec5a2b6aae0c737aa3"
    ),
    "wds-audio-valid-000003.tar": (
        "835702bdf7cff55a59a1322e25115de5f6ec45a7c7cb173759448e0c6b99051c"
    ),
    "wds-audio-valid-000004.tar": (
        "33a85886fdaef81f61cf33d8505b26c8a57518f57758698bc98da6e20ebb02ae"
    ),
    "wds-audio-valid-000005.tar": (
        "5f36d3fc8d2129043619186c9aa2da5bbd1f9bf80a7b88a81248672cc1927c81"
    ),
}
VOCALSOUND_SPLIT_SIZE: Final = 20
FLEURS_URL: Final = (
    "https://huggingface.co/datasets/google/fleurs/resolve/"
    "70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/de_de/audio/{name}"
)


class Corpus(StrEnum):
    """The corpora the probe reads."""

    AMI = "ami"
    VOCALSOUND = "vocalsound"
    FLEURS = "fleurs"


@dataclass(frozen=True)
class Pinned:
    """One file of a corpus: where it comes from and the sha256 it is read at.

    A file without a sha256 is named but never read: its checksum is written here
    at its first fetch, before a measured run reads it.
    """

    name: str
    url: str
    sha256: str | None


@dataclass(frozen=True)
class AmiHeadset:
    """One wearer's microphone of one meeting."""

    meeting: str
    channel: int
    audio: Pinned


def ami_headset(meeting: str, channel: int, sha256: str | None) -> AmiHeadset:
    """The headset of `channel` in `meeting`, read at `sha256`."""
    name = f"{meeting}.Headset-{channel}.wav"
    url = AMI_AUDIO_URL.format(meeting=meeting, channel=channel)
    return AmiHeadset(meeting, channel, Pinned(name, url, sha256))


def vocalsound_shard(split: str, number: int) -> Pinned:
    """A shard of the VocalSound WebDataset repack, at its checksum once fetched."""
    name = f"wds-audio-{split}-{number:06d}.tar"
    return Pinned(name, VOCALSOUND_URL.format(name=name), VOCALSOUND_SHA256.get(name))


def fleurs_split(split: str, sha256: str) -> Pinned:
    """A German FLEURS split of the pinned revision."""
    name = f"{split}.tar.gz"
    return Pinned(f"fleurs_de_{name}", FLEURS_URL.format(name=name), sha256)


@dataclass(frozen=True)
class Sources:
    """Every file the probe may read, the extras after the files it starts with."""

    ami_annotations: Pinned
    ami_headsets: tuple[AmiHeadset, ...]
    ami_extra_headsets: tuple[AmiHeadset, ...]
    vocalsound_shards: tuple[Pinned, ...]
    vocalsound_extra_shards: tuple[Pinned, ...]
    fleurs: Pinned
    fleurs_extra: Pinned


# The extras are the next headsets by isolated backchannels per megabyte over the
# annotations zip, in that order; each gets its sha256 at its first fetch.
SOURCES: Final = Sources(
    ami_annotations=Pinned(
        "ami_public_manual_1.6.2.zip",
        AMI_ANNOTATIONS_URL,
        "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d",
    ),
    ami_headsets=(
        ami_headset(
            "ES2002d",
            2,
            "131d3ecbfa4ab828564614ebe7ac052ab39aaf1bac6b7c59fb8c24a5382dd4ae",
        ),
        ami_headset(
            "IS1007c",
            1,
            "e86f40887e1a1101ae89ec2a650256e5acfe13621be3fb502c5446e7c6257476",
        ),
        ami_headset(
            "ES2002b",
            1,
            "bb79fbf1892d03ae7a81e9e6f72d22feb37513c149fd59a0ddf2d216a262c1fb",
        ),
        ami_headset(
            "IS1000b",
            0,
            "f611bfb49aa5f0df573709f97ace209cbee40ac4f052adb0f168e2fa1cb99cf9",
        ),
    ),
    ami_extra_headsets=(
        ami_headset(
            "TS3012d",
            2,
            "1b07184b89ce18ab7e1207199acf004bf9dcce3e56dd8e72118bf8ab6bfe1740",
        ),
        ami_headset(
            "IS1000c",
            0,
            "77321365da59d08398f078698f6ff19eb84430c510079a8b40d73995ba863afb",
        ),
        ami_headset(
            "ES2014d",
            3,
            "f8e44c96f3a1aee3b3f8a13ebfbf6b0beb463dced9cf36386e970acd1552b636",
        ),
        ami_headset("ES2002b", 3, None),
        ami_headset("ES2006d", 2, None),
    ),
    vocalsound_shards=tuple(vocalsound_shard("test", number) for number in range(4)),
    vocalsound_extra_shards=(
        *(vocalsound_shard("test", number) for number in range(4, 6)),
        *(vocalsound_shard("valid", number) for number in range(VOCALSOUND_SPLIT_SIZE)),
        *(vocalsound_shard("train", number) for number in range(VOCALSOUND_SPLIT_SIZE)),
    ),
    fleurs=fleurs_split(
        "dev", "72cac13bbddbddf674d8102a9e7b90a3b1db5d777ed92e88eccc5ae69cb8fc9d"
    ),
    fleurs_extra=fleurs_split(
        "test", "e86b42dfcdef749926cd92135045f87c25966c09e50d01c401adb04ee7d8628f"
    ),
)

CORPORA: Final = {
    Corpus.AMI: {
        "corpus": "AMI Meeting Corpus",
        "language": "English",
        "licence": "CC BY 4.0",
        "licence_source": (
            "LICENCE.txt of the annotations zip: released under the Creative Commons"
            " Attribution 4.0 International Public License"
        ),
        "revision": "annotations 1.6.2 (10.04.2017), headset audio of the mirror",
        "used_for": "backchannels and onsets (English)",
        "gaps": [
            "meetings with a role-played design task, not free conversation",
            "its laughs and coughs are not used: VocalSound owns them",
        ],
    },
    Corpus.VOCALSOUND: {
        "corpus": "VocalSound",
        "language": "none (non-speech sounds)",
        "licence": (
            "CC BY-SA 4.0 (the authors, README of github.com/YuanGongND/vocalsound);"
            " the Zenodo repack, record 14650192, is CC BY 4.0"
        ),
        "revision": (
            "Zenodo record 14650192, test, validation and train shards;"
            " md5 values are the record's"
        ),
        "used_for": "laughter, cough and throat clearing",
        "gaps": [
            "recorded on request by crowdsourcing, not in conversation",
            "the shards' speaker metadata is never read, only the class label",
            "a table of counts is not adapted material, so ShareAlike does not bind it",
        ],
    },
    Corpus.FLEURS: {
        "corpus": "FLEURS German",
        "language": "German",
        "licence": "CC BY 4.0",
        "revision": "google/fleurs at 70bb2e84b976b7e960aa89f1c648e09c59f894dd",
        "used_for": "onsets (German)",
        "gaps": [
            "read speech: no open German conversational corpus",
            (
                "no open German backchannel corpus was found, so the false-hold half"
                " is English and language-neutral sounds"
            ),
        ],
    },
}
DELAY_NOTE: Final = (
    "The delay is the service's count only: transport and the stop of the playback"
    " come on top of the card's half second."
)


def fetch(cache: Path, pinned: Pinned) -> bytes:
    """The file's bytes, from the cache or downloaded into it, refused unless pinned."""
    if pinned.sha256 is None:
        message = (
            f"{pinned.name} has no pinned sha256: fetch it once, and write its"
            " checksum into the script before a measured run reads it"
        )
        raise SystemExit(message)
    target = cache / pinned.name
    if target.exists():
        content = target.read_bytes()
        if hashlib.sha256(content).hexdigest() != pinned.sha256:
            message = f"{target} is not the pinned {pinned.sha256}"
            raise SystemExit(message)
        return content
    content = download_pinned(pinned.url, pinned.sha256)
    cache.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    partial.write_bytes(content)
    partial.rename(target)
    return content


def parse_xml(data: bytes) -> ET.Element:
    """The root of an annotation file, refused with a DOCTYPE or ENTITY declaration."""
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        message = "an annotation file declares a DOCTYPE or an ENTITY"
        raise SystemExit(message)
    return ET.fromstring(data)  # noqa: S314 - no declaration reaches the parser


def decode_wav(data: bytes) -> tuple[int, np.ndarray]:
    """Sample rate and the samples of a mono WAV of 16-bit PCM or 32-bit float."""
    rate, payload = pcm_from_wav(data)
    (tag, channels), (bits,) = (
        struct.unpack_from(layout, data, offset) for layout, offset in _FIXED_WAV_FIELDS
    )
    if channels != 1:
        message = f"a WAV with {channels} channels: the probe reads one"
        raise ValueError(message)
    if tag == _FLOAT_WAV_FORMAT and bits == _FLOAT_BITS:
        floats = np.frombuffer(payload[: len(payload) // 4 * 4], dtype="<f4")
        return rate, np.clip(np.round(floats * 32768), -32768, 32767).astype("<i2")
    if tag == _PCM_WAV_FORMAT and bits == _PCM_BITS:
        return rate, np.frombuffer(payload[: len(payload) // 2 * 2], dtype="<i2")
    message = f"a WAV of format {tag} and {bits} bits: the probe reads PCM16 or float32"
    raise ValueError(message)


@dataclass
class Collection:
    """What the corpora gave: trials, pairable clips, counts and the files read."""

    probabilities: Callable[[bytes], list[float]]
    tally: Tally = field(default_factory=Tally)
    hold_trials: list[HoldTrial] = field(default_factory=list)
    delay_trials: list[DelayTrial] = field(default_factory=list)
    pairable: dict[Sound, list[tuple[str, HeardClip]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    read: dict[Corpus, list[Pinned]] = field(default_factory=lambda: defaultdict(list))

    def hear(self, samples: np.ndarray, rate: int) -> HeardClip:
        """A clip as the live socket hears it: 24 kHz, then Silero's windows."""
        wire = to_wire_rate(samples.tobytes(), rate)
        return HeardClip(wire, tuple(self.probabilities(wire)))

    def add_sound(
        self,
        sound: Sound,
        clip: HeardClip,
        fate: Fate,
        key: str,
        *,
        overlapped: bool = False,
    ) -> None:
        """Count one excerpt of `sound`; a kept one is a trial, a heard one pairs."""
        self.tally.add(sound, fate)
        if fate.kept:
            self.hold_trials.append(
                HoldTrial(
                    sound,
                    paired=False,
                    overlapped=overlapped,
                    probabilities=figures.closed(clip.probabilities),
                )
            )
        if fate is Fate.HEARD:
            self.pairable[sound].append((key, clip))

    def add_onset(
        self,
        corpus: DelayCorpus,
        clip: HeardClip,
        fate: Fate,
        onset_seconds: float,
        *,
        overlapped: bool = False,
    ) -> None:
        """Count one onset of `corpus`; a kept one is heard or never heard."""
        self.tally.add(corpus, fate)
        if fate.kept:
            self.delay_trials.append(
                DelayTrial(corpus, overlapped, clip.probabilities, onset_seconds)
            )

    def add_pairs(self) -> None:
        """Join each sound's heard clips two by two; each pair is a trial."""
        for sound, keyed in self.pairable.items():
            clips = [clip for _, clip in sorted(keyed, key=lambda entry: entry[0])]
            for index, (first, second) in enumerate(figures.two_by_two(clips)):
                joined = figures.join_pair(
                    first, second, figures.pair_gap_windows(index)
                )
                self.hold_trials.append(
                    HoldTrial(
                        sound,
                        paired=True,
                        overlapped=False,
                        probabilities=figures.closed(self.probabilities(joined)),
                    )
                )


def excerpt(
    samples: np.ndarray, rate: int, start: float, end: float
) -> tuple[np.ndarray, float]:
    """The samples between `start` and `end` seconds, cut at the recording's ends.

    Never padded; the second is where the excerpt starts.
    """
    first = max(round(start * rate), 0)
    return samples[first : min(round(end * rate), len(samples))], first / rate


@dataclass(frozen=True)
class SpeakerAnnotation:
    """A wearer's timed words and vocal sounds, and where their backchannels lie."""

    timed: list[Timed]
    backchannels: list[range]


def _timed(element: ET.Element) -> Timed | None:
    start, end = element.get("starttime"), element.get("endtime")
    if start is None or end is None or element.get("punc") == "true":
        return None
    if element.tag == "w":
        return Timed(float(start), float(end), spoken=True)
    if element.tag == "vocalsound":
        return Timed(float(start), float(end), spoken=False)
    return None


def ami_annotation(
    archive: zipfile.ZipFile, meeting: str, speaker: str
) -> SpeakerAnnotation:
    """The speaker's timed items and the dialogue acts of type `bck` over them.

    Each act is the range of items from its first to its last child.
    """
    timed: list[Timed] = []
    before: dict[str, int] = {}
    after: dict[str, int] = {}
    for element in parse_xml(archive.read(f"words/{meeting}.{speaker}.words.xml")):
        identifier = element.get(NITE_ID, "")
        before[identifier] = len(timed)
        if (item := _timed(element)) is not None:
            timed.append(item)
        after[identifier] = len(timed)
    kinds = parse_xml(archive.read("ontologies/da-types.xml"))
    backchannel_type = next(
        kind.get(NITE_ID) for kind in kinds.iter("da-type") if kind.get("name") == "bck"
    )
    acts = parse_xml(archive.read(f"dialogueActs/{meeting}.{speaker}.dialog-act.xml"))
    ranges = []
    for act in acts:
        types = [p.get("href", "") for p in act if p.tag.endswith("pointer")]
        if not any(href.endswith(f"#id({backchannel_type})") for href in types):
            continue
        child = next(c for c in act if c.tag.endswith("child")).get("href", "")
        ids = re.findall(r"id\(([^)]+)\)", child.partition("#")[2])
        ranges.append(range(before[ids[0]], after[ids[-1]]))
    return SpeakerAnnotation(timed, ranges)


def ami_segments(archive: zipfile.ZipFile, meeting: str, speaker: str) -> list[Segment]:
    """The stretches in which the speaker talks."""
    root = parse_xml(archive.read(f"segments/{meeting}.{speaker}.segments.xml"))
    return [
        Segment(float(s.get("transcriber_start")), float(s.get("transcriber_end")))
        for s in root
        if s.get("transcriber_start") and s.get("transcriber_end")
    ]


def ami_speakers(archive: zipfile.ZipFile, meeting: str) -> dict[int, str]:
    """The speaker letter of each headset channel of the meeting."""
    meetings = parse_xml(archive.read("corpusResources/meetings.xml"))
    return {
        int(speaker.get("channel")): speaker.get("nxt_agent")
        for observed in meetings
        if observed.get("observation") == meeting
        for speaker in observed
    }


def collect_ami(collection: Collection, cache: Path, sources: Sources) -> None:
    """Backchannels and onsets of each headset; extra headsets only while missed."""
    collection.read[Corpus.AMI].append(sources.ami_annotations)
    archive = zipfile.ZipFile(io.BytesIO(fetch(cache, sources.ami_annotations)))
    for headset in _until_met(
        sources.ami_headsets,
        sources.ami_extra_headsets,
        collection,
        [Sound.BACKCHANNEL, DelayCorpus.AMI],
    ):
        collection.read[Corpus.AMI].append(headset.audio)
        rate, samples = decode_wav(fetch(cache, headset.audio))
        speakers = ami_speakers(archive, headset.meeting)
        wearer = speakers[headset.channel]
        others = [
            segment
            for letter in speakers.values()
            if letter != wearer
            for segment in ami_segments(archive, headset.meeting, letter)
        ]
        annotation = ami_annotation(archive, headset.meeting, wearer)
        for span in figures.backchannel_spans(
            annotation.timed, annotation.backchannels, others
        ):
            _take_backchannel(
                collection, samples, rate, span, f"{headset.meeting}.{wearer}"
            )
        for onset in figures.speech_onsets(annotation.timed, others):
            _take_ami_onset(collection, samples, rate, onset)


def _take_ami_onset(
    collection: Collection, samples: np.ndarray, rate: int, onset: figures.Onset
) -> None:
    clip_samples, begins = excerpt(
        samples,
        rate,
        onset.at - figures.EXCERPT_BEFORE_SECONDS,
        onset.at + figures.ONSET_AFTER_SECONDS,
    )
    clip = collection.hear(clip_samples, rate)
    at = onset.at - begins
    fate = figures.fate_of(
        clip.probabilities,
        inside=(at - figures.ISOLATION_TOLERANCE_SECONDS, math.inf),
    )
    collection.add_onset(DelayCorpus.AMI, clip, fate, at, overlapped=onset.overlapped)


def _take_backchannel(
    collection: Collection,
    samples: np.ndarray,
    rate: int,
    span: Backchannel,
    wearer: str,
) -> None:
    clip_samples, begins = excerpt(
        samples,
        rate,
        span.start - figures.EXCERPT_BEFORE_SECONDS,
        span.end + figures.BACKCHANNEL_AFTER_SECONDS,
    )
    clip = collection.hear(clip_samples, rate)
    tolerance = figures.ISOLATION_TOLERANCE_SECONDS
    fate = figures.fate_of(
        clip.probabilities,
        inside=(span.start - begins - tolerance, span.end - begins + tolerance),
    )
    collection.add_sound(
        Sound.BACKCHANNEL,
        clip,
        fate,
        f"{wearer}.{span.start:012.3f}",
        overlapped=span.overlapped,
    )


def _until_met[T](
    first: Sequence[T],
    extra: Sequence[T],
    collection: Collection,
    groups: Sequence[str],
) -> Iterator[T]:
    """The first files, then extras one by one while a group is below its minimum."""
    yield from first
    for item in extra:
        if not collection.tally.missed(groups):
            return
        yield item


def tar_members(content: bytes, mode: str) -> Iterator[tuple[str, bytes]]:
    """The regular files of a tar held in memory, never written out."""
    with tarfile.open(fileobj=io.BytesIO(content), mode=mode) as archive:
        for member in archive:
            if member.isfile():
                tarfile.data_filter(member, "")
                yield member.name, archive.extractfile(member).read()


def collect_vocalsound(collection: Collection, cache: Path, sources: Sources) -> None:
    """Whole clips of laughter, cough and throat clearing; extra shards while missed."""
    for shard in _until_met(
        sources.vocalsound_shards,
        sources.vocalsound_extra_shards,
        collection,
        _VOCALSOUND_LABELS,
    ):
        collection.read[Corpus.VOCALSOUND].append(shard)
        files = dict(tar_members(fetch(cache, shard), "r:"))
        for name in sorted(files):
            stem, _, suffix = name.rpartition(".")
            if suffix != "json":
                continue
            label = json.loads(files[name]).get("label")
            if label not in _VOCALSOUND_LABELS:
                continue
            rate, samples = decode_wav(files[f"{stem}.wav"])
            clip = collection.hear(samples, rate)
            fate = figures.fate_of(
                clip.probabilities, max_windows=figures.MAX_SOUND_WINDOWS
            )
            collection.add_sound(Sound(label), clip, fate, Path(stem).name)


def collect_fleurs(collection: Collection, cache: Path, sources: Sources) -> None:
    """Each clip's first voiced window is its onset; the test split only if missed.

    A clip that Silero never hears has no onset.
    """
    for split in _until_met(
        [sources.fleurs], [sources.fleurs_extra], collection, [DelayCorpus.FLEURS]
    ):
        collection.read[Corpus.FLEURS].append(split)
        for _, content in tar_members(fetch(cache, split), "r:gz"):
            rate, samples = decode_wav(content)
            voiced = [
                index
                for index, p in enumerate(collection.hear(samples, rate).probabilities)
                if is_voiced(p)
            ]
            if not voiced:
                continue
            onset = voiced[0] * WINDOW_SECONDS
            if onset < figures.LEAD_IN_SECONDS:
                collection.tally.add(DelayCorpus.FLEURS, Fate.NO_LEAD_IN)
                continue
            clip_samples, begins = excerpt(
                samples,
                rate,
                onset - figures.EXCERPT_BEFORE_SECONDS,
                onset + figures.ONSET_AFTER_SECONDS,
            )
            clip = collection.hear(clip_samples, rate)
            collection.add_onset(
                DelayCorpus.FLEURS,
                clip,
                figures.fate_of(clip.probabilities),
                onset - begins,
            )


def collect(
    cache: Path,
    sources: Sources,
    probabilities: Callable[[bytes], list[float]],
) -> Collection:
    """Every trial the corpora give, and what became of each excerpt."""
    collection = Collection(probabilities)
    collect_ami(collection, cache, sources)
    collect_vocalsound(collection, cache, sources)
    collect_fleurs(collection, cache, sources)
    collection.add_pairs()
    return collection


def corpora_record(collection: Collection) -> list[dict[str, object]]:
    """The corpora with their licences, and the checksums of the files read."""
    return [
        {
            **description,
            "files": [
                {"name": pinned.name, "sha256": pinned.sha256}
                for pinned in collection.read[corpus]
            ],
        }
        for corpus, description in CORPORA.items()
    ]


def document(collection: Collection) -> dict[str, object]:
    """The table: counts, then each cell's row, then what the pick rule takes."""
    measured = [
        figures.measure(cell, collection.hold_trials, collection.delay_trials)
        for cell in figures.GRID
    ]
    taken = figures.pick(measured)
    return {
        "missed": collection.tally.missed(figures.MINIMUM_KEPT),
        "corpora": corpora_record(collection),
        "counts": collection.tally.record(),
        "minimums": {str(group): n for group, n in figures.MINIMUM_KEPT.items()},
        "delay_note": DELAY_NOTE,
        "cells": [figures.cell_record(f, collection.hold_trials) for f in measured],
        "pick": {
            "cell": None
            if taken.cell is None
            else {
                "duration_windows": taken.cell.duration_windows,
                "restart_run": taken.cell.restart_run,
            },
            "replaces_ruled": taken.replaces_ruled,
            "within_bound": taken.within_bound,
        },
    }


def parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="where the pinned corpora are kept; never under /tmp or in a repository",
    )
    return parser.parse_args(arguments)


def probe(
    arguments: argparse.Namespace,
    *,
    sources: Sources = SOURCES,
    probabilities: Callable[[bytes], list[float]] = speech_probabilities,
    output: TextIO | None = None,
) -> None:
    """Collect the trials and print the table; exit 3 when a minimum is missed."""
    output = output or sys.stdout
    table = document(collect(arguments.cache, sources, probabilities))
    json.dump(table, output)
    output.write("\n")
    if table["missed"]:
        raise SystemExit(MISSED_EXIT)


def main() -> None:
    """Run the probe over the pinned corpora."""
    probe(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
