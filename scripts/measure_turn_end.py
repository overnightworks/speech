"""Measure, on CPU, how turn-end detectors judge German and English sentences.

Each recording is one read sentence, heard through the service's own turn
detection. A detector is asked wherever the service has heard a decision
silence: after the sentence, and at each pause the reader makes inside it;
`turn_end_figures` says how its answers read. Smart Turn judges the audio heard
so far; Namo and the general language models judge the words the service's own
hearing makes of it. The report is Markdown on standard output; models and
recordings stay in the Hugging Face cache.

Run it as `uv run scripts/measure_turn_end.py`; `--languages` and `--judges`
cut it into runs that each stay below a time cap.
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
import enum
import io
import math
import re
import sys
import tarfile
import time
from dataclasses import dataclass
from functools import partial
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

import turn_end_figures as figures
from onnx_session import one_thread_session
from speech_client.wire import Language
from speech_service.hearing import CPU_THREADS, ScoreIdentifier, WhisperHearing
from speech_service.hearing_models import HearingModel, fetch_model, voiced_floor
from speech_service.hearing_session import (
    MAX_TURN_SECONDS,
    HearingSession,
    SpeechStart,
    TurnEnd,
)
from speech_service.voice_activity import SAMPLE_RATE
from turn_end_figures import (
    DECISION_SILENCE_SECONDS,
    EARLY_START_SILENCES_SECONDS,
    FINISHED_ABOVE,
    FIXED_SILENCES_SECONDS,
    Findings,
    Judgement,
    Sentence,
    Stretches,
    Words,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

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

# What a language model reads before the words it judges: read sentences, one on
# each line, that end in their closing mark.
READ_SENTENCES: Final = (
    "Der Zug nach Hamburg fährt heute von Gleis sieben ab.",
    "The new bridge was opened by the mayor last week.",
    "Viele Besucher kamen schon am frühen Morgen in das Museum.",
    "Scientists have found water on the planet.",
)
# A token that closes a sentence; one that trails off in an ellipsis does not.
_CLOSING_TOKEN: Final = re.compile(r"\s*[.!?][^\w.…]*")
_TRAILING_MARKS: Final = re.compile(r"\W+$")
_MEMORY_INPUTS: Final = "past_key_values"
_MEMORY_OUTPUTS: Final = "present"

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


def _qwen3(size: str, revision: str) -> ModelFile:
    # The 4-bit file is the one whose load stays below 4 GB of memory at both sizes.
    return ModelFile(
        f"onnx-community/Qwen3-{size}-ONNX",
        "onnxruntime/cpu_and_mobile/cpu-int4-kld-block-128/model.onnx",
        revision,
        f"Apache-2.0 as its base model `Qwen/Qwen3-{size}`",
    )


SMART_TURN_FILES: Final = {
    "Smart Turn, 8-bit": _smart_turn("smart-turn-v3.2-cpu.onnx"),
    "Smart Turn, 32-bit": _smart_turn("smart-turn-v3.2-gpu.onnx"),
}
NAMO_LABEL: Final = "Namo"
LANGUAGE_MODEL_FILES: Final = {
    "Qwen3 0.6B": _qwen3("0.6B", "79346d370dfae34801d26993fda9ae2372900fd0"),
    "Qwen3 1.7B": _qwen3("1.7B", "3231b522c6e703b2f9b3cd8ea27426a1289a02f2"),
}


class Judges(enum.StrEnum):
    """What a detector judges: the audio of a turn, or the words heard in it."""

    AUDIO = "audio"
    WORDS = "words"


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
class LanguageHeard:
    """One language once its audio is no longer needed.

    `lines` report what was heard of it, `in_audio` what the detectors of audio
    found, and `in_words` holds the sentences the judges of words are given.
    """

    language: Measured
    lines: tuple[str, ...]
    in_audio: tuple[Findings, ...]
    in_words: tuple[Sentence[Words], ...]


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


def hear_sentence(
    recording: Recording, hearing: WhisperHearing
) -> Sentence[bytes] | None:
    """The sentence recorded; `None` when no silence long enough to decide follows."""
    heard: dict[float, Stretches[bytes]] = {}
    for silence in EARLY_START_SILENCES_SECONDS:
        pauses, end = _heard_at_silences(recording.pcm, hearing, silence)
        if end is None:
            return None
        heard[silence] = Stretches(tuple(pauses), end)
    cut_by = frozenset(
        silence
        for silence in FIXED_SILENCES_SECONDS
        if _heard_at_silences(recording.pcm, hearing, silence)[0]
    )
    return Sentence(recording.transcript, heard, cut_by)


def in_words(
    sentence: Sentence[bytes], words_in: Callable[[bytes], Words]
) -> Sentence[Words]:
    """The sentence as the words the service's hearing makes of each stretch."""
    heard = {
        silence: Stretches(
            tuple(words_in(pause) for pause in stretches.pauses),
            words_in(stretches.end),
        )
        for silence, stretches in sentence.heard.items()
    }
    return Sentence(sentence.transcript, heard, sentence.cut_by)


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


def _shared_start(tokens: Sequence[int], known: Sequence[int]) -> int:
    """How many tokens two sequences have in common from their start."""
    pairs = zip(tokens, known, strict=False)
    differing = (at for at, (token, other) in enumerate(pairs) if token != other)
    return next(differing, min(len(tokens), len(known)))


class LanguageModel:
    """A general language model that goes on from the words heard so far.

    Its score is the probability it gives a closing mark as the next token:
    whether it would end the sentence here itself. That is what a model this
    small knows; asked for a Yes or No about the sentence, it tells a finished
    one from a broken-off one poorly. It reads the words without the mark the
    hearing closed them with, which is the hearing's guess at the same question.
    """

    def __init__(self, file: ModelFile) -> None:
        """Open the pinned model on one thread, and read `READ_SENTENCES` once."""
        self._session = one_thread_session(file.fetch())
        self._tokenizer = AutoTokenizer.from_pretrained(
            file.repository, revision=file.revision
        )
        self._closing = [
            token
            for text, token in self._tokenizer.get_vocab().items()
            if _CLOSING_TOKEN.fullmatch(
                self._tokenizer.convert_tokens_to_string([text])
            )
        ]
        layers = [
            layer
            for layer in self._session.get_inputs()
            if layer.name.startswith(_MEMORY_INPUTS)
        ]
        self._layers = [layer.name for layer in layers]
        self._before = "".join(f"{sentence}\n" for sentence in READ_SENTENCES)
        self._known = self._tokens(self._before)
        nothing = [
            np.zeros((1, layer.shape[1], 0, layer.shape[3]), dtype=np.float32)
            for layer in layers
        ]
        kept = [name.replace(_MEMORY_INPUTS, _MEMORY_OUTPUTS) for name in self._layers]
        self._memory = self._run(self._known, nothing, kept)

    def _tokens(self, text: str) -> list[int]:
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def _run(
        self, tokens: Sequence[int], memory: Sequence[np.ndarray], outputs: list[str]
    ) -> list[np.ndarray]:
        """Read `tokens` after what `memory` holds of the tokens before them."""
        read = memory[0].shape[2] + len(tokens)
        return self._session.run(
            outputs,
            {
                "input_ids": np.array([tokens], dtype=np.int64),
                "attention_mask": np.ones((1, read), dtype=np.int64),
                **dict(zip(self._layers, memory, strict=True)),
            },
        )

    def finished_score(self, heard: str) -> float:
        """The model's probability that a closing mark follows the words."""
        tokens = self._tokens(self._before + _TRAILING_MARKS.sub("", heard))
        # The model has to read one token at least, also when no word was heard.
        shared = min(_shared_start(tokens, self._known), len(tokens) - 1)
        memory = [layer[:, :, :shared] for layer in self._memory]
        logits = self._run(tokens[shared:], memory, ["logits"])[0][0, -1]
        likelihoods = np.exp(logits.astype(np.float64) - logits.max())
        return float(likelihoods[self._closing].sum() / likelihoods.sum())


def judge[Heard](
    detector: Detector[Heard], heard: Heard, words_seconds: float = 0.0
) -> Judgement:
    """The detector's judgement of `heard`, timed in CPU and wall seconds.

    `words_seconds` is what the hearing took to make the words judged.
    """
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    score = detector.finished_score(heard)
    return Judgement(
        score,
        time.process_time() - cpu_start,
        time.perf_counter() - wall_start,
        words_seconds,
    )


def _called_finished[Heard](
    detector: Detector[Heard], cases: Iterable[Heard]
) -> tuple[bool, ...]:
    return tuple(detector.finished_score(case) > FINISHED_ABOVE for case in cases)


def _broken_off(sentence: Sentence[bytes]) -> bytes:
    """The sentence's first half as audio, then the silence a decision waits for."""
    end = sentence.heard[DECISION_SILENCE_SECONDS].end
    half = len(end) // 2 // _BYTES_PER_SAMPLE * _BYTES_PER_SAMPLE
    silence = int(DECISION_SILENCE_SECONDS * SAMPLE_RATE) * _BYTES_PER_SAMPLE
    return end[:half] + bytes(silence)


def measure_audio(
    label: str, detector: Detector[bytes], sentences: Sequence[Sentence[bytes]]
) -> Findings:
    """A detector of audio on what the service has heard at each silence.

    Its control is the sentence broken off in the middle of its speech.
    """
    results = figures.results_of(sentences, partial(judge, detector))
    at_end = results[DECISION_SILENCE_SECONDS]
    broken = (_broken_off(sentence) for sentence in sentences)
    control = {
        "the whole sentence": tuple(result.at_end.finished for result in at_end),
        (
            "the sentence broken off at half its audio, then"
            f" {DECISION_SILENCE_SECONDS:g} s of silence"
        ): _called_finished(detector, broken),
    }
    return Findings(label, results, control)


def _first_half(transcript: str) -> str:
    words = transcript.split()
    return " ".join(words[: len(words) // 2])


def measure_text(
    label: str, detector: Detector[str], sentences: Sequence[Sentence[Words]]
) -> Findings:
    """A judge of words on what the service's hearing makes of each silence.

    Its control is the written sentence: whole, without the mark that closes it,
    and cut after half of its words. The stretches are heard before the first
    decision, so that no hearing thread still runs while a decision is timed.
    """
    results = figures.results_of(
        sentences, lambda words: judge(detector, words.text, words.wall_seconds)
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
    return Findings(label, results, control)


def measure_words(
    language: Measured, sentences: Sequence[Sentence[Words]]
) -> list[Findings]:
    """Every judge of words on one language, each model open only for its turn."""
    if not sentences:
        return []
    openers = {
        NAMO_LABEL: partial(Namo, language.namo),
        **{
            label: partial(LanguageModel, file)
            for label, file in LANGUAGE_MODEL_FILES.items()
        },
    }
    return [
        measure_text(label, opener(), sentences) for label, opener in openers.items()
    ]


def _described(label: str, file: ModelFile) -> str:
    megabytes = file.fetch().stat().st_size / 1_000_000
    return (
        f"- {label}: `{file.repository}` `{file.name}` at `{file.revision[:12]}`,"
        f" {file.licence}, {megabytes:.1f} MB."
    )


def _detector_lines(
    languages: Sequence[Measured], judges: Sequence[Judges]
) -> list[str]:
    in_audio = [_described(label, file) for label, file in SMART_TURN_FILES.items()]
    in_words = [
        *(
            _described(f"{NAMO_LABEL}, {language.title}", language.namo)
            for language in languages
        ),
        *(_described(label, file) for label, file in LANGUAGE_MODEL_FILES.items()),
    ]
    return [
        *(in_audio if Judges.AUDIO in judges else []),
        *(in_words if Judges.WORDS in judges else []),
    ]


def report_setup(languages: Sequence[Measured], judges: Sequence[Judges]) -> list[str]:
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
        *_detector_lines(languages, judges),
        (
            "- A detector's own decision runs on one CPU thread and calls a turn"
            f" finished above a score of {FINISHED_ABOVE:g}. The hearing pass that"
            f" gives a judge of words its words runs on {CPU_THREADS} threads."
        ),
        *figures.reading_lines(),
        *(f"- Not measured: {reason}" for reason in NOT_MEASURED),
        "",
    ]


def hear_language(
    language: Measured,
    hearing: WhisperHearing,
    smart_turns: Mapping[str, SmartTurn],
    recordings_heard: int,
    sentences_in_words: int,
) -> LanguageHeard:
    """Hear one language's recordings: all that needs their audio or the hearing."""
    recordings = load_recordings(language, recordings_heard)
    heard = (hear_sentence(recording, hearing) for recording in recordings)
    sentences = [sentence for sentence in heard if sentence]
    in_audio = tuple(
        measure_audio(label, detector, sentences)
        for label, detector in smart_turns.items()
    )
    words = tuple(
        in_words(sentence, lambda pcm: hear_words(hearing, language.heard_as, pcm))
        for sentence in sentences[:sentences_in_words]
    )
    lines = figures.heard_lines(len(recordings), sentences)
    if words:
        lines += figures.words_lines(words, CPU_THREADS)
    return LanguageHeard(language, tuple(lines), in_audio, words)


def hear_languages(
    languages: Sequence[Measured],
    judges: Sequence[Judges],
    recordings_heard: int,
    sentences_in_words: int,
) -> list[LanguageHeard]:
    """Hear every language, then close the hearing: a language model needs its room."""
    hearing = open_hearing()
    smart_turns = (
        {label: SmartTurn(file) for label, file in SMART_TURN_FILES.items()}
        if Judges.AUDIO in judges
        else {}
    )
    in_words_each = sentences_in_words if Judges.WORDS in judges else 0
    heard = [
        hear_language(language, hearing, smart_turns, recordings_heard, in_words_each)
        for language in languages
    ]
    hearing.close()
    return heard


def positive_count(text: str) -> int:
    """Parse a command line count that must be at least one."""
    count = int(text)
    if count < 1:
        message = f"{text} is not a positive count"
        raise argparse.ArgumentTypeError(message)
    return count


def _limits() -> argparse.Namespace:
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
            "how many of their sentences the judges of words are given: each of"
            " their decisions costs one pass of the hearing model"
        ),
    )
    titles = [language.title for language in LANGUAGES]
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=titles,
        default=titles,
        help="which languages to measure",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        type=Judges,
        choices=list(Judges),
        default=list(Judges),
        help="which detectors to ask: those of audio, those of words, or both",
    )
    return parser.parse_args()


def main() -> None:
    """Measure the chosen languages with the chosen detectors and print the report."""
    limits = _limits()
    languages = [
        language for language in LANGUAGES if language.title in limits.languages
    ]
    lines = report_setup(languages, limits.judges)
    # The feature step's BLAS pool would otherwise spin its threads through the
    # model's run and bill a decision several times its one-thread cost.
    with threadpool_limits(limits=1, user_api="blas"):
        for heard in hear_languages(
            languages,
            limits.judges,
            limits.recordings_per_language,
            limits.transcribed_per_language,
        ):
            measured = [
                *heard.in_audio,
                *measure_words(heard.language, heard.in_words),
            ]
            lines += figures.report_language(
                heard.language.title, heard.lines, measured
            )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
