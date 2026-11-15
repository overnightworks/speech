"""Measure, on CPU, how a turn-end detector judges German and English sentences.

Each recording is one read sentence, heard through the service's own turn
detection. The detector is asked wherever the service has heard
`DECISION_SILENCE_SECONDS` of silence: after the sentence, where "finished" ends
the turn sooner than the fixed silence does, and at each pause the reader makes
inside the sentence, where "finished" cuts the turn. The report is Markdown on
standard output; models and recordings stay in the Hugging Face cache.

Run it as `uv run scripts/measure_turn_end.py`.
"""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "faster-whisper==1.2.1",
#     "huggingface-hub==2.4.0",
#     "numpy==2.5.4",
#     "onnxruntime==1.29.0",
#     "soundfile==0.13.1",
#     "transformers==5.23.0",
# ]
# ///

# ruff: noqa: T201 - the measurement prints its report as the command line output

from __future__ import annotations

import argparse
import io
import re
import statistics
import sys
import tarfile
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import numpy as np
import onnxruntime
import soundfile
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

# The service's turn detection is read from this checkout: installing the service
# would pull its private dependencies into the measurement.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speech_service.hearing_session import (
    MAX_TURN_SECONDS,
    HearingSession,
    SpeechStart,
    TurnEnd,
)
from speech_service.voice_activity import (
    SAMPLE_RATE,
    SileroDetectors,
    open_shipped_model,
)

FIXED_SILENCE_SECONDS: Final = 0.5
DECISION_SILENCE_SECONDS: Final = 0.2
LONGER_SILENCES_SECONDS: Final = (1.0, 2.0, 3.0)

SPEECH_REPOSITORY: Final = "google/fleurs"
SPEECH_REVISION: Final = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
SPEECH_SPLIT: Final = "dev"
SPEECH_LICENCE: Final = "CC-BY-4.0"
_ONE_SENTENCE: Final = re.compile(r"[^.!?]*[.!?]")
_TRANSCRIPT_COLUMN: Final = 2
_FILE_NAME_COLUMN: Final = 1

SMART_TURN_REPOSITORY: Final = "pipecat-ai/smart-turn-v3"
SMART_TURN_REVISION: Final = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
SMART_TURN_FILE: Final = "smart-turn-v3.2-cpu.onnx"
SMART_TURN_LICENCE: Final = "BSD-2-Clause"
SMART_TURN_SECONDS: Final = 8
SMART_TURN_THRESHOLD: Final = 0.5

NOT_MEASURED: Final = (
    (
        "LiveKit turn detector: its model licence allows use only together with the"
        " LiveKit Agents framework, not standalone."
    ),
    (
        "Kyutai semantic VAD: a head of Kyutai's own speech-to-text model"
        " (stt-1b-en_fr), English and French only."
    ),
)

_BYTES_PER_SAMPLE: Final = 2
_BYTES_PER_MILLISECOND: Final = SAMPLE_RATE // 1000 * _BYTES_PER_SAMPLE
_INT16_PEAK: Final = 32767


class Language(StrEnum):
    """The languages measured, by their name in the speech corpus."""

    GERMAN = "de_de"
    ENGLISH = "en_us"


@dataclass(frozen=True, slots=True)
class Sentence:
    """One read sentence as the service hears it, 16 kHz PCM16 mono.

    `pauses` holds the audio heard up to each pause inside the sentence, `end`
    the audio heard once the sentence is over; both stop where the silence has
    lasted `DECISION_SILENCE_SECONDS`. `cut_by` names the fixed silences that
    end the turn inside the sentence.
    """

    pauses: tuple[bytes, ...]
    end: bytes
    cut_by: frozenset[float]


@dataclass(frozen=True, slots=True)
class Judgement:
    """Whether the detector called the turn finished, and what deciding cost."""

    finished: bool
    cpu_seconds: float
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class SentenceResult:
    """The detector's judgements on one sentence."""

    sentence: Sentence
    at_end: Judgement
    at_pauses: tuple[Judgement, ...]

    @property
    def cut(self) -> bool:
        """Whether the detector ends the turn at a pause inside the sentence."""
        return any(judgement.finished for judgement in self.at_pauses)


def _one_sentence_recordings(language: Language) -> frozenset[str]:
    """The file names whose transcript is one sentence, ended by its only mark."""
    table = hf_hub_download(
        SPEECH_REPOSITORY,
        f"data/{language}/{SPEECH_SPLIT}.tsv",
        repo_type="dataset",
        revision=SPEECH_REVISION,
    )
    rows = (
        line.split("\t")
        for line in Path(table).read_text(encoding="utf-8").splitlines()
    )
    return frozenset(
        row[_FILE_NAME_COLUMN]
        for row in rows
        if _ONE_SENTENCE.fullmatch(row[_TRANSCRIPT_COLUMN].strip())
    )


def _pcm_of(recording: bytes) -> bytes:
    """A 16 kHz mono recording as PCM16; any other format fails loud."""
    samples, rate = soundfile.read(io.BytesIO(recording), dtype="float32")
    if rate != SAMPLE_RATE or samples.ndim != 1:
        message = f"a recording is {rate} Hz with shape {samples.shape}"
        raise ValueError(message)
    return (np.clip(samples, -1, 1) * _INT16_PEAK).round().astype("<i2").tobytes()


def load_recordings(language: Language, limit: int) -> list[bytes]:
    """The first `limit` one-sentence recordings of `language` below the turn cap."""
    wanted = _one_sentence_recordings(language)
    archive_path = hf_hub_download(
        SPEECH_REPOSITORY,
        f"data/{language}/audio/{SPEECH_SPLIT}.tar.gz",
        repo_type="dataset",
        revision=SPEECH_REVISION,
    )
    longest = int(MAX_TURN_SECONDS * SAMPLE_RATE) * _BYTES_PER_SAMPLE
    recordings: list[bytes] = []
    with tarfile.open(archive_path) as archive:
        for member in archive:
            if len(recordings) == limit:
                break
            if not member.isfile() or Path(member.name).name not in wanted:
                continue
            pcm = _pcm_of(archive.extractfile(member).read())
            if len(pcm) < longest:
                recordings.append(pcm)
    return recordings


def _heard_at_silences(
    pcm: bytes, detectors: SileroDetectors, silence_seconds: float
) -> tuple[list[bytes], bytes | None]:
    """What the service has heard wherever `silence_seconds` of silence end a turn.

    Returns the audio up to each such silence that speech follows, and the audio
    up to the one after the last speech, `None` when the recording stops sooner.
    Each stretch starts where the service opened the first turn.
    """
    session = HearingSession(lambda: 0.0, detectors.open(), silence_seconds)
    events = session.add(pcm)
    ends = [event for event in events if isinstance(event, TurnEnd)]
    if not ends:
        return [], None
    first = ends[0].start_ms * _BYTES_PER_MILLISECOND
    heard = [pcm[first : end.end_ms * _BYTES_PER_MILLISECOND] for end in ends]
    if isinstance(events[-1], SpeechStart):
        return heard, None
    return heard[:-1], heard[-1]


def hear_sentence(pcm: bytes, detectors: SileroDetectors) -> Sentence | None:
    """The sentence in `pcm`; `None` when no silence long enough to decide follows."""
    pauses, end = _heard_at_silences(pcm, detectors, DECISION_SILENCE_SECONDS)
    if end is None:
        return None
    fixed_silences = (FIXED_SILENCE_SECONDS, *LONGER_SILENCES_SECONDS)
    cut_by = frozenset(
        silence
        for silence in fixed_silences
        if _heard_at_silences(pcm, detectors, silence)[0]
    )
    return Sentence(tuple(pauses), end, cut_by)


class SmartTurn:
    """Pipecat Smart Turn: judges the last seconds of a turn's audio."""

    def __init__(self) -> None:
        """Fetch the pinned CPU model and open it on one thread, as a service would."""
        self.path = Path(
            hf_hub_download(
                SMART_TURN_REPOSITORY, SMART_TURN_FILE, revision=SMART_TURN_REVISION
            )
        )
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(self.path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._features = WhisperFeatureExtractor(chunk_length=SMART_TURN_SECONDS)
        self.finished(bytes(SAMPLE_RATE * _BYTES_PER_SAMPLE))

    def finished(self, heard: bytes) -> bool:
        """Whether the turn heard so far sounds finished.

        The model takes its last seconds, with silence in front of a shorter turn.
        """
        window = SMART_TURN_SECONDS * SAMPLE_RATE
        samples = np.frombuffer(heard, dtype="<i2").astype(np.float32) / (
            _INT16_PEAK + 1
        )
        samples = np.pad(samples[-window:], (max(window - len(samples), 0), 0))
        features = self._features(
            samples,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=window,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        probability = self._session.run(None, {"input_features": features})[0]
        return float(probability.item()) > SMART_TURN_THRESHOLD


def judge(detector: SmartTurn, heard: bytes) -> Judgement:
    """The detector's judgement of `heard`, timed in CPU and wall seconds."""
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    finished = detector.finished(heard)
    return Judgement(
        finished,
        time.process_time() - cpu_start,
        time.perf_counter() - wall_start,
    )


def judge_sentence(detector: SmartTurn, sentence: Sentence) -> SentenceResult:
    """The detector's judgements after `sentence` and at each pause inside it."""
    return SentenceResult(
        sentence,
        judge(detector, sentence.end),
        tuple(judge(detector, heard) for heard in sentence.pauses),
    )


def _share(part: int, whole: int) -> str:
    return f"{part} of {whole} ({part / whole:.0%})" if whole else "0 of 0"


def _milliseconds(seconds: list[float]) -> str:
    median = statistics.median(seconds) * 1000
    high = statistics.quantiles(seconds, n=20)[-1] * 1000
    return f"{median:.1f} / {high:.1f}"


def report_language(
    language: Language, recordings: int, results: list[SentenceResult]
) -> list[str]:
    """The Markdown lines of one language's figures."""
    total = len(results)
    judgements = [
        judgement
        for result in results
        for judgement in (result.at_end, *result.at_pauses)
    ]
    pauses = [judgement for result in results for judgement in result.at_pauses]
    cut_today = [r for r in results if FIXED_SILENCE_SECONDS in r.sentence.cut_by]
    kept_whole = _share(sum(not r.cut for r in cut_today), len(cut_today))
    longer = ", ".join(
        f"{silence:g} s: {sum(silence in r.sentence.cut_by for r in results)}"
        for silence in LONGER_SILENCES_SECONDS
    )
    return [
        f"### {language.name.title()}",
        "",
        (
            f"- Sentences: {total} of {recordings} recordings; the others stop before"
            f" {DECISION_SILENCE_SECONDS:g} s of closing silence."
        ),
        (
            f"- Pauses of {DECISION_SILENCE_SECONDS:g} s or more inside a sentence:"
            f" {len(pauses)}, in {sum(bool(r.at_pauses) for r in results)} sentences."
        ),
        f"- Turns a longer fixed silence still cuts inside the sentence: {longer}.",
        "",
        (
            "| Rule | Finished turns ended sooner | Turns cut inside the sentence"
            " | CPU ms per decision, median / p95 | Wall ms, median / p95 |"
        ),
        "|---|---|---|---|---|",
        (
            f"| Fixed {FIXED_SILENCE_SECONDS:g} s silence | {_share(0, total)}"
            f" | {_share(len(cut_today), total)} | none | none |"
        ),
        (
            f"| Smart Turn at {DECISION_SILENCE_SECONDS:g} s silence"
            f" | {_share(sum(r.at_end.finished for r in results), total)}"
            f" | {_share(sum(r.cut for r in results), total)}"
            f" | {_milliseconds([j.cpu_seconds for j in judgements])}"
            f" | {_milliseconds([j.wall_seconds for j in judgements])} |"
        ),
        "",
        (
            f"- Pauses Smart Turn calls finished:"
            f" {_share(sum(j.finished for j in pauses), len(pauses))}."
        ),
        (
            f"- Turns the fixed {FIXED_SILENCE_SECONDS:g} s silence cuts that Smart"
            f" Turn keeps whole: {kept_whole}."
        ),
        "",
    ]


def report_setup(detector: SmartTurn) -> list[str]:
    """The Markdown lines naming the data, the detector and what was left out."""
    megabytes = detector.path.stat().st_size / 1_000_000
    return [
        "## Turn end measurement",
        "",
        (
            f"- Speech: `{SPEECH_REPOSITORY}` at `{SPEECH_REVISION[:12]}`, split"
            f" `{SPEECH_SPLIT}`, {SPEECH_LICENCE}: recordings of one read sentence."
        ),
        (
            f"- Detector: `{SMART_TURN_REPOSITORY}` `{SMART_TURN_FILE}` at"
            f" `{SMART_TURN_REVISION[:12]}`, {SMART_TURN_LICENCE}, {megabytes:.1f} MB,"
            f" one CPU thread, finished above {SMART_TURN_THRESHOLD:g}."
        ),
        *(f"- Not measured: {reason}" for reason in NOT_MEASURED),
        "",
    ]


def main() -> None:
    """Measure both languages and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recordings-per-language",
        type=int,
        default=300,
        help="how many one-sentence recordings to hear per language",
    )
    limit = parser.parse_args().recordings_per_language
    detectors = SileroDetectors(open_shipped_model)
    detectors.load()
    detector = SmartTurn()
    lines = report_setup(detector)
    for language in Language:
        recordings = load_recordings(language, limit)
        heard = [hear_sentence(pcm, detectors) for pcm in recordings]
        results = [judge_sentence(detector, sentence) for sentence in heard if sentence]
        lines += report_language(language, len(recordings), results)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
