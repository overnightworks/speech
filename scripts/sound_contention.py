"""The detector's hops and the hearing's times while two voices speak."""

from __future__ import annotations

import itertools
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from huggingface_hub import constants

from shipped_catalogue import CATALOGUE
from sound_taggers import (
    HOP_SECONDS,
    SAMPLE_RATE,
    HopCost,
    Samples,
    Tagger,
    timed,
    windows_of,
)
from speech_client.wire import Language, VoiceId
from speech_service.hearing import ScoreIdentifier, WhisperHearing
from speech_service.hearing_models import HearingModel, fetch_model, voiced_floor
from speech_service.piper_voices import fetch_voice_files
from speech_service.speaking import PiperSpeaking

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sound_corpus import Corpus, Sentence

MAX_HEARING_RISE: Final = 0.10
RECORDING_SECONDS: Final = (8.0, 14.0)
HEARING_ROUNDS: Final = 3
PHASE_SECONDS: Final = 40.0
CONTENTION_CLIPS: Final = 40
SPOKEN_SENTENCES: Final = 40
HEARING_MODEL: Final = HearingModel.SMALL
PIPER_VOICES: Final = ((VoiceId.THORSTEN, Language.DE), (VoiceId.NORMAN, Language.EN))
_INT16_PEAK: Final = 32767


def _rise(alone: Sequence[float], with_detector: Sequence[float]) -> float:
    """How much the median hearing time rises with the detector running."""
    return statistics.median(with_detector) / statistics.median(alone) - 1


@dataclass(frozen=True, slots=True)
class Contention:
    """The detector's hops and the hearing's times while two voices speak."""

    hops: tuple[HopCost, ...]
    hearing_alone: tuple[float, ...]
    hearing_with_detector: tuple[float, ...]
    round_rises: tuple[float, ...]

    @property
    def rise(self) -> float:
        """How much the median hearing time rises over all rounds."""
        return _rise(self.hearing_alone, self.hearing_with_detector)

    @property
    def too_noisy(self) -> bool:
        """Whether machine load, not the detector, moves the hearing time.

        The rounds measure the same rise; a spread wider than the threshold being
        judged, or a hearing that gets faster with the detector, cannot be it.
        """
        spread = max(self.round_rises) - min(self.round_rises)
        return spread > MAX_HEARING_RISE or self.rise < 0


def _speak(voice: PiperSpeaking, texts: list[str], stop: threading.Event) -> None:
    """Synthesise `texts` in a loop at the pace a client plays them."""
    started, played = time.perf_counter(), 0.0
    for text in itertools.cycle(texts):
        for chunk in voice.pcm_chunks(text, ""):
            played += len(chunk) / 2 / voice.sample_rate
            if stop.wait(max(0.0, started + played - time.perf_counter())):
                return


def _detect_paced(
    tagger: Tagger,
    windows: Sequence[Samples],
    stop: threading.Event,
    costs: list[HopCost],
) -> None:
    due = time.perf_counter()
    for window in itertools.cycle(windows):
        if stop.is_set():
            return
        costs.append(timed(tagger, window)[1])
        due += HOP_SECONDS
        stop.wait(max(0.0, due - time.perf_counter()))


def _hear_times(
    hearing: WhisperHearing, recording: Sentence, runs: int, seconds: float
) -> list[float]:
    """The times of at least `runs` hearings that together last `seconds`."""
    pcm = (recording.samples * _INT16_PEAK).round().astype("<i2").tobytes()
    times, begun = [], time.perf_counter()
    while len(times) < runs or time.perf_counter() - begun < seconds:
        started = time.perf_counter()
        hearing.transcribe(pcm, (recording.language,))
        times.append(time.perf_counter() - started)
    return times


def measure_contention(
    tagger: Tagger, corpus: Corpus, cache: Path, runs: int
) -> Contention:
    """The hearing's time with and without the detector, two voices speaking.

    The two conditions alternate `HEARING_ROUNDS` times, each for `PHASE_SECONDS`
    and at least `runs` hearings, so that a drifting machine load falls on both.
    The detector hops at the service's pace over windows of the public clips; the
    voices speak at the pace of a playing client.
    """
    hearing = WhisperHearing(
        HEARING_MODEL,
        fetch_model(HEARING_MODEL, Path(constants.HF_HUB_CACHE)),
        "cpu",
        ScoreIdentifier(voiced_floor(HEARING_MODEL)),
    )
    hearing.load()
    stop, speakers = threading.Event(), []
    spoken = corpus.sentences[: 2 * SPOKEN_SENTENCES]
    for voice_id, language in PIPER_VOICES:
        fetch_voice_files(CATALOGUE[voice_id], cache / "voices")
        voice = PiperSpeaking(voice_id.value, cache / "voices")
        voice.load()
        texts = [s.transcript for s in spoken if s.language is language]
        speakers.append(
            threading.Thread(target=_speak, args=(voice, texts, stop), daemon=True)
        )
    low, high = RECORDING_SECONDS
    recording = next(
        s for s in corpus.sentences if low <= len(s.samples) / SAMPLE_RATE <= high
    )
    spread = corpus.clips[:: max(1, len(corpus.clips) // CONTENTION_CLIPS)]
    windows = [
        w for clip in spread for w in windows_of(clip.samples, tagger.window_samples)
    ]
    alone, with_detector, hops, round_rises = [], [], [], []
    for speaker in speakers:
        speaker.start()
    _hear_times(hearing, recording, 1, 0.0)
    for _ in range(HEARING_ROUNDS):
        round_alone = _hear_times(hearing, recording, runs, PHASE_SECONDS)
        paused = threading.Event()
        detector = threading.Thread(
            target=_detect_paced, args=(tagger, windows, paused, hops), daemon=True
        )
        detector.start()
        round_with = _hear_times(hearing, recording, runs, PHASE_SECONDS)
        paused.set()
        detector.join()
        alone += round_alone
        with_detector += round_with
        round_rises.append(_rise(round_alone, round_with))
    stop.set()
    for speaker in speakers:
        speaker.join()
    return Contention(
        tuple(hops), tuple(alone), tuple(with_detector), tuple(round_rises)
    )
