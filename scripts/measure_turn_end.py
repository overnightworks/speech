"""Measure, on CPU, how turn-end detectors judge German and English sentences.

Each recording is one read sentence, heard through the service's own turn
detection. A detector is asked wherever the service has heard
`DECISION_SILENCE_SECONDS` of silence: after the sentence, where "finished" ends
the turn sooner than the fixed silence does, and at each pause the reader makes
inside the sentence, where "finished" cuts the turn. Smart Turn judges the audio
heard so far, Namo the words the service's own hearing makes of it. The report
is Markdown on standard output; models and recordings stay in the Hugging Face
cache.

Run it as `uv run scripts/measure_turn_end.py`.
"""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "faster-whisper==1.2.1",
#     "huggingface-hub==2.4.0",
#     "numpy==2.5.4",
#     "onnxruntime==1.29.0",
#     "pydantic==2.14.3",
#     "soundfile==0.13.1",
#     "threadpoolctl==3.6.0",
#     "transformers==5.23.0",
# ]
# ///

# ruff: noqa: T201 - the measurement prints its report as the command line output

from __future__ import annotations

import argparse
import io
import math
import re
import statistics
import sys
import tarfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np
import soundfile
from huggingface_hub import constants, hf_hub_download
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer, WhisperFeatureExtractor

# The service's turn detection and hearing are read from this checkout: installing
# the service would pull its private dependencies into the measurement.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/client/src"))

from onnx_session import one_thread_session
from speech_client.wire import Language
from speech_service.hearing import ScoreIdentifier, WhisperHearing
from speech_service.hearing_models import HearingModel, fetch_model, voiced_floor
from speech_service.hearing_session import (
    MAX_TURN_SECONDS,
    HearingSession,
    SpeechStart,
    TurnEnd,
)
from speech_service.voice_activity import SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

FIXED_SILENCE_SECONDS: Final = 0.5
DECISION_SILENCE_SECONDS: Final = 0.2
LONGER_SILENCES_SECONDS: Final = (1.0, 2.0, 3.0)
FINISHED_ABOVE: Final = 0.5

SPEECH_REPOSITORY: Final = "google/fleurs"
SPEECH_REVISION: Final = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
SPEECH_SPLIT: Final = "dev"
SPEECH_LICENCE: Final = "CC-BY-4.0"
_ONE_SENTENCE: Final = re.compile(r"[^.!?]*[.!?]")
_TRANSCRIPT_COLUMN: Final = 2
_FILE_NAME_COLUMN: Final = 1

# The large model the service recommends needs the GPU this measurement leaves alone.
HEARING_MODEL: Final = HearingModel.SMALL

SMART_TURN_SECONDS: Final = 8
NAMO_MAX_TOKENS: Final = 512

NOT_MEASURED: Final = (
    (
        "LiveKit turn detector: its model licence allows use only together with the"
        " LiveKit Agents framework, not standalone or with another framework."
    ),
    (
        "Kyutai semantic VAD: a head of Kyutai's own speech-to-text model"
        " (stt-1b-en_fr), English and French only, so it has no German and would"
        " replace the speech-to-text model."
    ),
)

_BYTES_PER_SAMPLE: Final = 2
_BYTES_PER_MILLISECOND: Final = SAMPLE_RATE // 1000 * _BYTES_PER_SAMPLE
_INT16_PEAK: Final = 32767
_FEWEST_FOR_QUANTILES: Final = 2


@dataclass(frozen=True, slots=True)
class ModelFile:
    """A pinned model file on the Hugging Face Hub, under its licence."""

    repository: str
    name: str
    revision: str
    licence: str

    def fetch(self) -> Path:
        """The file in the Hugging Face cache, downloaded when it is not there."""
        return Path(hf_hub_download(self.repository, self.name, revision=self.revision))


def _smart_turn(name: str) -> ModelFile:
    return ModelFile(
        "pipecat-ai/smart-turn-v3",
        name,
        "f766f81d3cfdf7737ac64aad813d91bbfd56bf93",
        "BSD-2-Clause",
    )


def _namo(language: str, revision: str) -> ModelFile:
    # The repository's quantised file scores every text alike under this
    # onnxruntime on a CPU with AVX2 and no VNNI, the publisher's own examples
    # included; the unquantised file tells them apart.
    return ModelFile(
        f"videosdk-live/Namo-Turn-Detector-v1-{language}",
        "model.onnx",
        revision,
        "Apache-2.0",
    )


SMART_TURN_FILES: Final = {
    "Smart Turn, 8-bit": _smart_turn("smart-turn-v3.2-cpu.onnx"),
    "Smart Turn, 32-bit": _smart_turn("smart-turn-v3.2-gpu.onnx"),
}
NAMO_LABEL: Final = "Namo"


@dataclass(frozen=True, slots=True)
class Measured:
    """One measured language: its corpus name, how the service hears it, its Namo."""

    title: str
    corpus: str
    heard_as: Language
    namo: ModelFile


LANGUAGES: Final = (
    Measured(
        "German",
        "de_de",
        Language.DE,
        _namo("German", "d40a517b7377be104bad6d42efe74ca68cdb6415"),
    ),
    Measured(
        "English",
        "en_us",
        Language.EN,
        _namo("English", "4dcc9713be5071ef43e510f72b1eefcb2f99cb04"),
    ),
)


@dataclass(frozen=True, slots=True)
class Recording:
    """One read sentence: its 16 kHz PCM16 mono audio and its written words."""

    pcm: bytes
    transcript: str


@dataclass(frozen=True, slots=True)
class Sentence:
    """One read sentence as the service hears it, 16 kHz PCM16 mono.

    `pauses` holds the audio heard up to each pause inside the sentence, `end`
    the audio heard once the sentence is over; both stop where the silence has
    lasted `DECISION_SILENCE_SECONDS`. `cut_by` names the fixed silences that
    end the turn inside the sentence.
    """

    transcript: str
    pauses: tuple[bytes, ...]
    end: bytes
    cut_by: frozenset[float]


@dataclass(frozen=True, slots=True)
class Words:
    """What the service's hearing makes of a stretch, and what hearing it cost."""

    text: str
    cpu_seconds: float
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class Judgement:
    """How finished the detector scored the turn, and what deciding cost."""

    score: float
    cpu_seconds: float
    wall_seconds: float

    @property
    def finished(self) -> bool:
        """Whether the score calls the turn finished."""
        return self.score > FINISHED_ABOVE


@dataclass(frozen=True, slots=True)
class SentenceResult:
    """A detector's judgements on one sentence, beside today's fixed silence."""

    cut_today: bool
    at_end: Judgement
    at_pauses: tuple[Judgement, ...]

    @property
    def cut(self) -> bool:
        """Whether the detector ends the turn at a pause inside the sentence."""
        return any(judgement.finished for judgement in self.at_pauses)


@dataclass(frozen=True, slots=True)
class Findings:
    """One detector on one language: its sentences, and its validity control.

    `control` names each case whose truth is known and holds, per sentence,
    whether the detector called it finished. `notes` say what the figures leave
    out about how the detector was fed.
    """

    label: str
    results: tuple[SentenceResult, ...]
    control: Mapping[str, tuple[bool, ...]]
    notes: tuple[str, ...] = ()


class Detector[Heard](Protocol):
    """A turn-end detector over what it is given of a turn: its audio or its words."""

    def finished_score(self, heard: Heard) -> float:
        """How finished the turn is, from 0 to 1."""


def _one_sentence_transcripts(language: Measured) -> dict[str, str]:
    """The transcripts that are one sentence ended by its only mark, by file name."""
    table = hf_hub_download(
        SPEECH_REPOSITORY,
        f"data/{language.corpus}/{SPEECH_SPLIT}.tsv",
        repo_type="dataset",
        revision=SPEECH_REVISION,
    )
    rows = (
        line.split("\t")
        for line in Path(table).read_text(encoding="utf-8").splitlines()
    )
    transcripts = (
        (row[_FILE_NAME_COLUMN], row[_TRANSCRIPT_COLUMN].strip()) for row in rows
    )
    return {
        name: transcript
        for name, transcript in transcripts
        if _ONE_SENTENCE.fullmatch(transcript)
    }


def _pcm_of(recording: bytes) -> bytes:
    """A 16 kHz mono recording as PCM16; any other format fails loud."""
    samples, rate = soundfile.read(io.BytesIO(recording), dtype="float32")
    if rate != SAMPLE_RATE or samples.ndim != 1:
        message = f"a recording is {rate} Hz with shape {samples.shape}"
        raise ValueError(message)
    return (np.clip(samples, -1, 1) * _INT16_PEAK).round().astype("<i2").tobytes()


def load_recordings(language: Measured, limit: int) -> list[Recording]:
    """The first `limit` one-sentence recordings of `language` below the turn cap."""
    transcripts = _one_sentence_transcripts(language)
    archive_path = hf_hub_download(
        SPEECH_REPOSITORY,
        f"data/{language.corpus}/audio/{SPEECH_SPLIT}.tar.gz",
        repo_type="dataset",
        revision=SPEECH_REVISION,
    )
    longest = int(MAX_TURN_SECONDS * SAMPLE_RATE) * _BYTES_PER_SAMPLE
    recordings: list[Recording] = []
    with tarfile.open(archive_path) as archive:
        for member in archive:
            if len(recordings) == limit:
                break
            transcript = transcripts.get(Path(member.name).name)
            if not member.isfile() or transcript is None:
                continue
            pcm = _pcm_of(archive.extractfile(member).read())
            if len(pcm) < longest:
                recordings.append(Recording(pcm, transcript))
    return recordings


def open_hearing() -> WhisperHearing:
    """The service's own hearing on the CPU, its pinned model fetched to the cache."""
    directory = fetch_model(HEARING_MODEL, Path(constants.HF_HUB_CACHE))
    hearing = WhisperHearing(
        HEARING_MODEL, directory, "cpu", ScoreIdentifier(voiced_floor(HEARING_MODEL))
    )
    hearing.load()
    return hearing


def hear_words(hearing: WhisperHearing, language: Language, pcm: bytes) -> Words:
    """The words the service's hearing makes of `pcm`, timed in CPU and wall seconds."""
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    text = hearing.transcribe(pcm, (language,)).text
    return Words(
        text,
        time.process_time() - cpu_start,
        time.perf_counter() - wall_start,
    )


def _heard_at_silences(
    pcm: bytes, hearing: WhisperHearing, silence_seconds: float
) -> tuple[list[bytes], bytes | None]:
    """What the service has heard wherever `silence_seconds` of silence end a turn.

    Returns the audio up to each such silence that speech follows, and the audio
    up to the one after the last speech, `None` when the recording stops sooner.
    Each stretch starts where the service opened the first turn.
    """
    session = HearingSession(lambda: 0.0, hearing.open_detector(), silence_seconds)
    events = session.add(pcm)
    ends = [event for event in events if isinstance(event, TurnEnd)]
    if not ends:
        return [], None
    first = ends[0].start_ms * _BYTES_PER_MILLISECOND
    heard = [pcm[first : end.end_ms * _BYTES_PER_MILLISECOND] for end in ends]
    if isinstance(events[-1], SpeechStart):
        return heard, None
    return heard[:-1], heard[-1]


def hear_sentence(recording: Recording, hearing: WhisperHearing) -> Sentence | None:
    """The sentence recorded; `None` when no silence long enough to decide follows."""
    pauses, end = _heard_at_silences(recording.pcm, hearing, DECISION_SILENCE_SECONDS)
    if end is None:
        return None
    fixed_silences = (FIXED_SILENCE_SECONDS, *LONGER_SILENCES_SECONDS)
    cut_by = frozenset(
        silence
        for silence in fixed_silences
        if _heard_at_silences(recording.pcm, hearing, silence)[0]
    )
    return Sentence(recording.transcript, tuple(pauses), end, cut_by)


class SmartTurn:
    """Pipecat Smart Turn: judges the last seconds of a turn's audio."""

    def __init__(self, file: ModelFile) -> None:
        """Open the pinned model on one thread, as a service would hold it."""
        self._session = one_thread_session(file.fetch())
        self._features = WhisperFeatureExtractor(chunk_length=SMART_TURN_SECONDS)
        self.finished_score(bytes(SAMPLE_RATE * _BYTES_PER_SAMPLE))

    def finished_score(self, heard: bytes) -> float:
        """The model's probability that the turn heard so far is finished.

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
        return float(probability.item())


class Namo:
    """VideoSDK Namo: judges the words of a turn, in one language."""

    def __init__(self, file: ModelFile) -> None:
        """Open the pinned model on one thread, and its repository's tokenizer."""
        self._session = one_thread_session(file.fetch())
        self._tokenizer = AutoTokenizer.from_pretrained(
            file.repository, revision=file.revision
        )
        self.finished_score("")

    def finished_score(self, heard: str) -> float:
        """The model's probability that the words end a turn."""
        tokens = self._tokenizer(
            heard, truncation=True, max_length=NAMO_MAX_TOKENS, return_tensors="np"
        )
        logits = self._session.run(
            None,
            {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
            },
        )[0][0]
        unfinished, finished = (float(logit) for logit in logits)
        return 1 / (1 + math.exp(unfinished - finished))


def judge[Heard](detector: Detector[Heard], heard: Heard) -> Judgement:
    """The detector's judgement of `heard`, timed in CPU and wall seconds."""
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    score = detector.finished_score(heard)
    return Judgement(
        score, time.process_time() - cpu_start, time.perf_counter() - wall_start
    )


def _judge_sentence[Heard](
    detector: Detector[Heard], sentence: Sentence, end: Heard, pauses: Iterable[Heard]
) -> SentenceResult:
    return SentenceResult(
        FIXED_SILENCE_SECONDS in sentence.cut_by,
        judge(detector, end),
        tuple(judge(detector, heard) for heard in pauses),
    )


def _called_finished[Heard](
    detector: Detector[Heard], cases: Iterable[Heard]
) -> tuple[bool, ...]:
    return tuple(detector.finished_score(case) > FINISHED_ABOVE for case in cases)


def _broken_off(sentence: Sentence) -> bytes:
    """The sentence's first half as audio, then the silence a decision waits for."""
    half = len(sentence.end) // 2 // _BYTES_PER_SAMPLE * _BYTES_PER_SAMPLE
    silence = int(DECISION_SILENCE_SECONDS * SAMPLE_RATE) * _BYTES_PER_SAMPLE
    return sentence.end[:half] + bytes(silence)


def measure_audio(
    label: str, detector: Detector[bytes], sentences: Sequence[Sentence]
) -> Findings:
    """A detector of audio on what the service has heard at each silence.

    Its control is the sentence broken off in the middle of its speech.
    """
    results = tuple(
        _judge_sentence(detector, sentence, sentence.end, sentence.pauses)
        for sentence in sentences
    )
    broken = (_broken_off(sentence) for sentence in sentences)
    control = {
        "the whole sentence": tuple(result.at_end.finished for result in results),
        (
            "the sentence broken off at half its audio, then"
            f" {DECISION_SILENCE_SECONDS:g} s of silence"
        ): _called_finished(detector, broken),
    }
    return Findings(label, results, control)


def _first_half(transcript: str) -> str:
    words = transcript.split()
    return " ".join(words[: len(words) // 2])


def _closing_mark(words: str) -> str:
    """How the words end: in an ellipsis, in another mark or in none."""
    ending = words.rstrip()
    if ending.endswith(("...", "…")):
        return "an ellipsis"
    if ending and not ending[-1].isalnum():
        return f"`{ending[-1]}`"
    return "no mark"


def _closing_marks(heard: Iterable[Words]) -> str:
    marks = Counter(_closing_mark(words.text) for words in heard)
    counted = ", ".join(f"{mark} ({count})" for mark, count in marks.most_common())
    return counted or "none heard"


def _hearing_notes(
    at_ends: Sequence[Words], at_pauses: Sequence[Words]
) -> tuple[str, ...]:
    passes = [*at_ends, *at_pauses]
    return (
        (
            "each decision follows a pass of the hearing model, which its time"
            f" leaves out: {_milliseconds([words.cpu_seconds for words in passes])}"
            f" CPU ms and {_milliseconds([words.wall_seconds for words in passes])}"
            " wall ms, median / p95."
        ),
        f"how the words heard up to the sentence end close: {_closing_marks(at_ends)}.",
        f"how the words heard up to a pause close: {_closing_marks(at_pauses)}.",
    )


def measure_text(
    label: str,
    detector: Detector[str],
    sentences: Sequence[Sentence],
    words_in: Callable[[bytes], Words],
) -> Findings:
    """A detector of words on what the service's hearing makes of each silence.

    Its control is the written sentence: whole, without the mark that closes it,
    and cut after half of its words. Every stretch is heard before the first
    decision, so that no hearing thread still runs while a decision is timed.
    """
    heard = [
        (words_in(sentence.end), [words_in(pause) for pause in sentence.pauses])
        for sentence in sentences
    ]
    results = tuple(
        _judge_sentence(detector, sentence, end.text, (pause.text for pause in pauses))
        for sentence, (end, pauses) in zip(sentences, heard, strict=True)
    )
    notes = _hearing_notes(
        [end for end, _ in heard], [pause for _, pauses in heard for pause in pauses]
    )
    written = [sentence.transcript for sentence in sentences]
    control = {
        "the written sentence": _called_finished(detector, written),
        "the written sentence without its closing mark": _called_finished(
            detector, (transcript[:-1] for transcript in written)
        ),
        "the first half of its written words": _called_finished(
            detector, (_first_half(transcript) for transcript in written)
        ),
    }
    return Findings(label, results, control, notes)


def _share(part: int, whole: int) -> str:
    return f"{part} of {whole} ({part / whole:.0%})" if whole else "0 of 0"


def _milliseconds(seconds: Sequence[float]) -> str:
    if len(seconds) < _FEWEST_FOR_QUANTILES:
        return "none"
    median = statistics.median(seconds) * 1000
    high = statistics.quantiles(seconds, n=20)[-1] * 1000
    return f"{median:.1f} / {high:.1f}"


def _quartiles(scores: Sequence[float]) -> str:
    if len(scores) < _FEWEST_FOR_QUANTILES:
        return "none"
    return " / ".join(f"{quartile:.2f}" for quartile in statistics.quantiles(scores))


def _rule_row(findings: Findings) -> str:
    results = findings.results
    pauses = [judgement for result in results for judgement in result.at_pauses]
    judgements = [*(result.at_end for result in results), *pauses]
    cut_today = [result for result in results if result.cut_today]
    return (
        f"| {findings.label} at {DECISION_SILENCE_SECONDS:g} s silence"
        f" | {len(results)}"
        f" | {_share(sum(r.at_end.finished for r in results), len(results))}"
        f" | {_share(sum(r.cut for r in results), len(results))}"
        f" | {_share(sum(j.finished for j in pauses), len(pauses))}"
        f" | {_share(sum(not r.cut for r in cut_today), len(cut_today))}"
        f" | {_milliseconds([j.cpu_seconds for j in judgements])}"
        f" | {_milliseconds([j.wall_seconds for j in judgements])} |"
    )


def _score_row(findings: Findings) -> str:
    at_end = [result.at_end.score for result in findings.results]
    at_pauses = [j.score for r in findings.results for j in r.at_pauses]
    return f"| {findings.label} | {_quartiles(at_end)} | {_quartiles(at_pauses)} |"


def _control_rows(findings: Findings) -> list[str]:
    return [
        f"| {findings.label} | {case} | {_share(sum(finished), len(finished))} |"
        for case, finished in findings.control.items()
    ]


def _heard_lines(recordings: int, sentences: Sequence[Sentence]) -> list[str]:
    longer = ", ".join(
        f"{silence:g} s: {sum(silence in s.cut_by for s in sentences)}"
        for silence in LONGER_SILENCES_SECONDS
    )
    return [
        (
            f"- Sentences: {len(sentences)} of {recordings} recordings; the others"
            f" stop before {DECISION_SILENCE_SECONDS:g} s of closing silence."
        ),
        (
            f"- Pauses of {DECISION_SILENCE_SECONDS:g} s or more inside a sentence:"
            f" {sum(len(s.pauses) for s in sentences)},"
            f" in {sum(bool(s.pauses) for s in sentences)} sentences."
        ),
        f"- Turns a longer fixed silence still cuts inside the sentence: {longer}.",
    ]


def _rule_table(
    sentences: Sequence[Sentence], measured: Sequence[Findings]
) -> list[str]:
    total = len(sentences)
    cut_today = sum(FIXED_SILENCE_SECONDS in s.cut_by for s in sentences)
    sooner = FIXED_SILENCE_SECONDS - DECISION_SILENCE_SECONDS
    return [
        (
            f"| Rule | Sentences | Finished turns ended {sooner:g} s sooner"
            " | Turns cut inside the sentence | Pauses called finished"
            f" | Turns the fixed {FIXED_SILENCE_SECONDS:g} s silence cuts, kept whole"
            " | CPU ms per decision, median / p95 | Wall ms, median / p95 |"
        ),
        "|---|---|---|---|---|---|---|---|",
        (
            f"| Fixed {FIXED_SILENCE_SECONDS:g} s silence | {total}"
            f" | {_share(0, total)} | {_share(cut_today, total)}"
            " | none | none | none | none |"
        ),
        *(_rule_row(findings) for findings in measured),
    ]


def report_language(
    language: Measured,
    recordings: int,
    sentences: Sequence[Sentence],
    measured: Sequence[Findings],
) -> list[str]:
    """The Markdown lines of one language's figures."""
    return [
        f"### {language.title}",
        "",
        *_heard_lines(recordings, sentences),
        "",
        *_rule_table(sentences, measured),
        "",
        (
            "| Detector | Score at the sentence end, quartiles"
            " | Score at a pause, quartiles |"
        ),
        "|---|---|---|",
        *(_score_row(findings) for findings in measured),
        "",
        "| Detector | Validity control | Called finished |",
        "|---|---|---|",
        *(row for findings in measured for row in _control_rows(findings)),
        "",
        *(
            f"- {findings.label}: {note}"
            for findings in measured
            for note in findings.notes
        ),
        "",
    ]


def _described(label: str, file: ModelFile) -> str:
    megabytes = file.fetch().stat().st_size / 1_000_000
    return (
        f"- {label}: `{file.repository}` `{file.name}` at `{file.revision[:12]}`,"
        f" {file.licence}, {megabytes:.1f} MB."
    )


def report_setup() -> list[str]:
    """The Markdown lines naming the data, the detectors and what was left out."""
    return [
        "## Turn end measurement",
        "",
        (
            f"- Speech: `{SPEECH_REPOSITORY}` at `{SPEECH_REVISION[:12]}`, split"
            f" `{SPEECH_SPLIT}`, {SPEECH_LICENCE}: recordings of one read sentence."
        ),
        (
            "- Turns and words: the service's own turn detection and its hearing"
            f" with `{HEARING_MODEL.value}` on the CPU."
        ),
        *(_described(label, file) for label, file in SMART_TURN_FILES.items()),
        *(
            _described(f"{NAMO_LABEL}, {language.title}", language.namo)
            for language in LANGUAGES
        ),
        (
            "- Every decision runs on one CPU thread and calls a turn finished above"
            f" a score of {FINISHED_ABOVE:g}."
        ),
        *(f"- Not measured: {reason}" for reason in NOT_MEASURED),
        "",
    ]


def measure_language(
    language: Measured,
    hearing: WhisperHearing,
    smart_turns: Mapping[str, SmartTurn],
    recordings_limit: int,
    transcribed_limit: int,
) -> list[str]:
    """Hear one language's recordings, ask every detector and report its figures."""
    recordings = load_recordings(language, recordings_limit)
    heard = (hear_sentence(recording, hearing) for recording in recordings)
    sentences = [sentence for sentence in heard if sentence]
    measured = [
        measure_audio(label, detector, sentences)
        for label, detector in smart_turns.items()
    ]
    measured.append(
        measure_text(
            NAMO_LABEL,
            Namo(language.namo),
            sentences[:transcribed_limit],
            lambda pcm: hear_words(hearing, language.heard_as, pcm),
        )
    )
    return report_language(language, len(recordings), sentences, measured)


def positive_count(text: str) -> int:
    """Parse a command line count that must be at least one."""
    count = int(text)
    if count < 1:
        message = f"{text} is not a positive count"
        raise argparse.ArgumentTypeError(message)
    return count


def main() -> None:
    """Measure both languages and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recordings-per-language",
        type=positive_count,
        default=300,
        help="how many one-sentence recordings to hear per language",
    )
    parser.add_argument(
        "--transcribed-per-language",
        type=positive_count,
        default=100,
        help=(
            "how many of their sentences the detector of words judges: each of its"
            " decisions costs one pass of the hearing model"
        ),
    )
    limits = parser.parse_args()
    hearing = open_hearing()
    smart_turns = {label: SmartTurn(file) for label, file in SMART_TURN_FILES.items()}
    lines = report_setup()
    # The feature step's BLAS pool would otherwise spin its threads through the
    # model's run and bill a decision several times its one-thread cost.
    with threadpool_limits(limits=1, user_api="blas"):
        for language in LANGUAGES:
            lines += measure_language(
                language,
                hearing,
                smart_turns,
                limits.recordings_per_language,
                limits.transcribed_per_language,
            )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
