"""The figures of the stage proof of speaking: each voice, two voices, hearing beside.

The steps take the speak and hear calls as arguments and start no service;
`prove_speech` hands them the calls that reach the running one. Every figure a
step measures is printed and returned as a record field, so a document can cite it.
"""

# ruff: noqa: T201 - the proof prints its stage lines as the command line output

from __future__ import annotations

import functools
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import hearing_figures as figures
from speech_client.wire import VOICES, Language
from speech_service.pcm import BYTES_PER_SAMPLE, pcm_from_wav, rms

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from speech_client.wire import VoiceId

type Speak = Callable[[str, VoiceId], tuple[bytes, float, float]]

MEASURE_REPETITIONS: Final = 5
SILENCE_RMS: Final = 0.01
MIN_SPEECH_SECONDS: Final = 0.4
MIN_SENTENCE_SECONDS: Final = 1.0
COMPLETE_FRACTION: Final = 0.8
"""The share of its single-voice audio an answer given beside another keeps."""
MAX_WORD_ERROR_RATE: Final = 0.2
"""The most word errors per reference word a heard recording may hold."""


@dataclass(frozen=True)
class Spoken:
    """One finished speak: the WAV, its times and what its audio holds."""

    wav: bytes
    first_audio_seconds: float
    total_seconds: float
    audio_seconds: float
    level: float
    rate: int

    @property
    def real_time_factor(self) -> float:
        """Seconds of audio made per second of the answer."""
        return self.audio_seconds / self.total_seconds if self.total_seconds else 0.0


def spoken_of(answer: tuple[bytes, float, float]) -> Spoken:
    """The figures of a speak's WAV, first audio and total time."""
    wav, first_audio, total = answer
    rate, pcm = pcm_from_wav(wav)
    seconds = (len(pcm) / BYTES_PER_SAMPLE) / rate
    return Spoken(wav, first_audio, total, seconds, rms(pcm), rate)


def language_of(voice: VoiceId) -> Language:
    """The language the catalogue says `voice` speaks first."""
    return VOICES[voice].languages[0]


def sentence_of(voice: VoiceId, sentences: Mapping[Language, str]) -> str:
    """The proof sentence in the language of `voice`."""
    return sentences[language_of(voice)]


def voice_pair(voices: Iterable[VoiceId]) -> tuple[VoiceId, VoiceId]:
    """The first German and the first English voice of `voices`, in their order."""
    offered = tuple(voices)
    german, english = (
        next(voice for voice in offered if language_of(voice) is language)
        for language in (Language.DE, Language.EN)
    )
    return german, english


def prove_one_speak(
    speak: Speak,
    text: str,
    voice: VoiceId,
    *,
    limit: float,
    label: str,
) -> Spoken:
    """Speak once; print the run and fail on silence, a stub or a late first byte."""
    spoken = spoken_of(speak(text, voice))
    print(
        label,
        f"first_byte_s={spoken.first_audio_seconds:.3f}",
        f"total_s={spoken.total_seconds:.3f}",
        f"limit_s={limit:.1f}",
        f"duration_s={spoken.audio_seconds:.3f}",
        f"rtf={spoken.real_time_factor:.2f}",
        f"rms={spoken.level:.4f}",
        f"rate={spoken.rate}",
        f"bytes={len(spoken.wav)}",
    )
    if spoken.level < SILENCE_RMS:
        message = f"{label} produced silence"
        raise SystemExit(message)
    if spoken.audio_seconds < MIN_SENTENCE_SECONDS:
        message = f"{label} produced too little audio to be a sentence"
        raise SystemExit(message)
    if spoken.first_audio_seconds > limit:
        message = (
            f"{label} first byte {spoken.first_audio_seconds:.3f}s"
            f" exceeded the {limit:.1f}s limit"
        )
        raise SystemExit(message)
    return spoken


@dataclass(frozen=True)
class VoiceFigures:
    """The medians of one voice's single runs, and the last run's WAV."""

    wav: bytes
    first_audio_seconds: float
    total_seconds: float
    audio_seconds: float
    real_time_factor: float


def prove_speak_medians(
    speak: Speak,
    text: str,
    voice: VoiceId,
    *,
    limit: float,
    label: str,
) -> VoiceFigures:
    """Speak the same sentence MEASURE_REPETITIONS times; print and return medians."""
    runs = [
        prove_one_speak(speak, text, voice, limit=limit, label=f"{label}_rep{rep}")
        for rep in range(1, MEASURE_REPETITIONS + 1)
    ]
    medians = VoiceFigures(
        runs[-1].wav,
        statistics.median(run.first_audio_seconds for run in runs),
        statistics.median(run.total_seconds for run in runs),
        statistics.median(run.audio_seconds for run in runs),
        statistics.median(run.real_time_factor for run in runs),
    )
    print(
        f"{label}_median",
        f"first_byte_s={medians.first_audio_seconds:.3f}",
        f"rtf={medians.real_time_factor:.2f}",
        f"limit_s={limit:.1f}",
        f"n={MEASURE_REPETITIONS}",
    )
    return medians


def prove_each_voice(
    speak: Speak,
    voices: Iterable[VoiceId],
    sentences: Mapping[Language, str],
    *,
    limit: float,
) -> dict[VoiceId, VoiceFigures]:
    """Speak each voice's sentence in its language; every repetition is judged."""
    return {
        voice: prove_speak_medians(
            speak,
            sentence_of(voice, sentences),
            voice,
            limit=limit,
            label=f"speak_wire_{voice.value}",
        )
        for voice in voices
    }


def speaking_tasks(
    speak: Speak, voices: Iterable[VoiceId], sentences: Mapping[Language, str]
) -> list[Callable[[], tuple[bytes, float, float]]]:
    """One call per voice that speaks its proof sentence."""
    return [
        functools.partial(speak, sentence_of(voice, sentences), voice)
        for voice in voices
    ]


def run_together[T](tasks: Sequence[Callable[[], T]]) -> list[T]:
    """Run each task in its own thread, all starting at one moment; their results.

    The first task that failed ends the call with its own error.
    """
    start = threading.Barrier(len(tasks))

    def work(task: Callable[[], T]) -> T:
        start.wait()
        return task()

    with ThreadPoolExecutor(len(tasks)) as pool:
        return list(pool.map(work, tasks))


def run_during[T, U](
    tasks: Sequence[Callable[[], T]], during: Callable[[], U]
) -> tuple[U, list[T]]:
    """Run `during` while each task repeats in its own thread; both results.

    Every task has begun its first call when `during` starts and finishes the call
    it is in when `during` ends, so each task completes at least once.
    """
    ended = threading.Event()
    begun = threading.Barrier(len(tasks) + 1)

    def repeat(task: Callable[[], T]) -> list[T]:
        begun.wait()
        results = [task()]
        while not ended.is_set():
            results.append(task())
        return results

    with ThreadPoolExecutor(len(tasks)) as pool:
        running = [pool.submit(repeat, task) for task in tasks]
        begun.wait()
        try:
            outcome = during()
        finally:
            ended.set()
        return outcome, [result for run in running for result in run.result()]


def require_complete(label: str, spoken: Spoken, single: VoiceFigures) -> None:
    """Fail unless an answer given beside another is audible and whole."""
    if spoken.level < SILENCE_RMS:
        message = f"{label} produced silence"
        raise SystemExit(message)
    if spoken.audio_seconds < COMPLETE_FRACTION * single.audio_seconds:
        message = (
            f"{label} is {spoken.audio_seconds:.2f}s of audio against"
            f" {single.audio_seconds:.2f}s alone, so it is cut short"
        )
        raise SystemExit(message)


def _record_of_times(
    first_audio: Sequence[float], total: Sequence[float]
) -> dict[str, float]:
    return {
        "first_audio_seconds": statistics.median(first_audio),
        "total_seconds": statistics.median(total),
    }


def prove_two_voices(
    speak: Speak,
    pair: tuple[VoiceId, VoiceId],
    sentences: Mapping[Language, str],
    singles: Mapping[VoiceId, VoiceFigures],
) -> dict[str, object]:
    """Speak both voices at one moment, repeated; report their times and the single's.

    Both answers of every repetition must be audible and whole; the times are
    figures, since a shared CPU swings them by more than an honest limit.
    """
    together: dict[VoiceId, list[Spoken]] = {voice: [] for voice in pair}
    tasks = speaking_tasks(speak, pair, sentences)
    for repetition in range(1, MEASURE_REPETITIONS + 1):
        answers = run_together(tasks)
        for voice, answer in zip(pair, answers, strict=True):
            spoken = spoken_of(answer)
            label = f"speak_pair_{voice.value}_rep{repetition}"
            print(
                label,
                f"first_byte_s={spoken.first_audio_seconds:.3f}",
                f"total_s={spoken.total_seconds:.3f}",
                f"duration_s={spoken.audio_seconds:.3f}",
            )
            require_complete(label, spoken, singles[voice])
            together[voice].append(spoken)
    record: dict[str, object] = {}
    for voice, runs in together.items():
        beside = _record_of_times(
            [run.first_audio_seconds for run in runs],
            [run.total_seconds for run in runs],
        )
        alone = singles[voice]
        print(
            f"speak_pair_{voice.value}_median",
            f"first_byte_s={beside['first_audio_seconds']:.3f}",
            f"single_first_byte_s={alone.first_audio_seconds:.3f}",
            f"total_s={beside['total_seconds']:.3f}",
            f"single_total_s={alone.total_seconds:.3f}",
            f"n={len(runs)}",
        )
        record[voice.value] = {
            "single": _record_of_times(
                [alone.first_audio_seconds], [alone.total_seconds]
            ),
            "together": beside,
        }
    return record


def require_heard(label: str, reference: str, heard: str) -> float:
    """The word error rate of `heard` against `reference`; fail above the limit."""
    rate = figures.word_errors(reference, heard).rate
    if rate > MAX_WORD_ERROR_RATE:
        message = (
            f"{label} heard {rate:.0%} of the words wrong, the limit is"
            f" {MAX_WORD_ERROR_RATE:.0%}"
        )
        raise SystemExit(message)
    return rate


def prove_hearing_beside_speaking(
    hear: Callable[[], tuple[float, str]],
    speaks: Sequence[Callable[[], tuple[bytes, float, float]]],
    reference: str,
) -> dict[str, object]:
    """Hear one recording alone, then while `speaks` repeat; report both latencies.

    Both transcripts must be right and every speak beside the hearing audible;
    the latencies are figures, not judged.
    """
    alone_seconds, alone_text = hear()
    alone_rate = require_heard("hearing alone", reference, alone_text)
    (busy_seconds, busy_text), answers = run_during(speaks, hear)
    busy_rate = require_heard("hearing beside speaking", reference, busy_text)
    for answer in answers:
        if spoken_of(answer).level < SILENCE_RMS:
            message = "a speak beside the hearing produced silence"
            raise SystemExit(message)
    print(
        "hear_alone",
        f"seconds={alone_seconds:.3f}",
        f"word_error_rate={alone_rate:.3f}",
    )
    print(
        "hear_beside_speaking",
        f"seconds={busy_seconds:.3f}",
        f"word_error_rate={busy_rate:.3f}",
        f"speaks={len(answers)}",
    )
    return {
        "alone": {"seconds": alone_seconds, "word_error_rate": alone_rate},
        "beside_speaking": {
            "seconds": busy_seconds,
            "word_error_rate": busy_rate,
            "speaks_completed": len(answers),
        },
    }


def speaking_record(
    device: str,
    singles: Mapping[VoiceId, VoiceFigures],
    two_voices: Mapping[str, object],
    hearing: Mapping[str, object] | None,
) -> dict[str, object]:
    """The figures of one run with the device they were taken on."""
    return {
        "device": device,
        "voices": {
            voice.value: {
                "first_audio_seconds": figure.first_audio_seconds,
                "total_seconds": figure.total_seconds,
                "real_time_factor": figure.real_time_factor,
            }
            for voice, figure in singles.items()
        },
        "two_voices": dict(two_voices),
        "hearing": None if hearing is None else dict(hearing),
    }
