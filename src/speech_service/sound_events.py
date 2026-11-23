"""Sound events: the AudioSet tagger frame_mn10 over 16 kHz audio and what it names.

One ONNX session is shared by every live session and runs on its own worker thread,
apart from hearing's decode worker; each session that asks opens a stream that
carries its own audio tail and the sounds it has begun.
"""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from speech_client.realtime_wire import SoundName
from speech_service.config import setting_variable
from speech_service.pcm import HEARING_SAMPLE_RATE, INT16_SCALE
from speech_service.voice_activity import one_thread_session

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    import numpy.typing as npt
    from onnxruntime import InferenceSession

type Scores = npt.NDArray[np.float32]

WINDOW_SAMPLES: Final = 2 * HEARING_SAMPLE_RATE
HOP_SAMPLES: Final = HEARING_SAMPLE_RATE // 2
# The sounds' measured hit-rates are those at this threshold.
THRESHOLD: Final = 0.1
QUIET_HOPS_TO_END: Final = 2
# The model answers scores shaped [1, labels, frames].
_SCORE_AXES: Final = 3
MODEL_FILE: Final = "frame_mn10.onnx"
LABELS_FILE: Final = "labels.json"
MODEL_FILES: Final = (MODEL_FILE, f"{MODEL_FILE}.data", LABELS_FILE, "LICENSE")
BUILD_COMMAND: Final = (
    "uv run --locked --no-build --script scripts/build_sound_model.py"
)
DIRECTORY_SETTING: Final = setting_variable("sound_model_directory")
SWITCH: Final = setting_variable("sound_events")
_INPUT: Final = "waveform"
_KEPT_SAMPLES: Final = WINDOW_SAMPLES + HOP_SAMPLES
# The model's labels are AudioSet's; Applause is its own class there, not a child of
# Clapping, so folding it into Clapping is this service's mapping. The order is the
# order events leave in when one window holds two sounds.
_LABELS_OF: Final[Mapping[SoundName, tuple[str, ...]]] = {
    SoundName.LAUGHTER: ("Laughter", "Giggle", "Snicker", "Belly laugh"),
    SoundName.COUGH: ("Cough",),
    SoundName.CLAPPING: ("Clapping", "Applause"),
    SoundName.SCREAMING: ("Screaming",),
}


class SoundModelError(RuntimeError):
    """The sound model is not the one the stream feeds."""


class SoundModelMissingError(SoundModelError):
    """The sound model's directory lacks a file of the build."""


def check_model_directory(directory: Path) -> None:
    """Raise `SoundModelMissingError` unless `directory` holds the build's files."""
    if all((directory / name).is_file() for name in MODEL_FILES):
        return
    message = (
        f"{DIRECTORY_SETTING}={directory} lacks the sound model; "
        f"build it with: {BUILD_COMMAND} {directory}, "
        f"or start without sounds with {SWITCH}=false"
    )
    raise SoundModelMissingError(message)


@dataclass(frozen=True, slots=True)
class SoundWindow:
    """Two seconds of audio to score, and where they lie on the session clock."""

    samples: Scores
    audio_start_ms: int
    audio_end_ms: int


@dataclass(frozen=True, slots=True)
class DetectedSound:
    """A sound that began in a window, and its highest score there."""

    name: SoundName
    score: float
    audio_start_ms: int
    audio_end_ms: int


class SoundStream:
    """One session's audio, hop by hop, and the sounds it has begun.

    A hop is due every half second from two seconds on. One window is in flight at a
    time; hops that fall due meanwhile, or together, are skipped but for the latest.
    A sound ends after two scored hops in a row below the threshold; a skipped hop
    counts neither way.
    """

    def __init__(self, columns: Mapping[SoundName, tuple[int, ...]]) -> None:
        """Start before the first sample; `columns` are each sound's score columns."""
        self._columns = columns
        self._tail = np.zeros(0, dtype="<i2")
        self._total = 0
        self._next_hop = 0
        self._in_flight = False
        self._begun = dict.fromkeys(columns, False)
        self._quiet_hops = dict.fromkeys(columns, 0)

    def add(self, pcm: bytes) -> None:
        """Take the session's next 16 kHz mono 16-bit audio."""
        samples = np.frombuffer(pcm, dtype="<i2")
        self._total += len(samples)
        self._tail = np.concatenate([self._tail, samples])[-_KEPT_SAMPLES:]

    def next_window(self) -> SoundWindow | None:
        """The latest due hop's window, or `None` while one flies or none is due."""
        latest = (self._total - WINDOW_SAMPLES) // HOP_SAMPLES
        if self._in_flight or latest < self._next_hop:
            return None
        self._next_hop = latest + 1
        self._in_flight = True
        start = latest * HOP_SAMPLES
        offset = start - (self._total - len(self._tail))
        window = self._tail[offset : offset + WINDOW_SAMPLES]
        return SoundWindow(
            (window.astype(np.float32) / INT16_SCALE)[None],
            start * 1000 // HEARING_SAMPLE_RATE,
            (start + WINDOW_SAMPLES) * 1000 // HEARING_SAMPLE_RATE,
        )

    def scored(self, window: SoundWindow, scores: Scores) -> list[DetectedSound]:
        """The sounds that begin in `window`, scored `[1, labels, frames]`."""
        self._in_flight = False
        began: list[DetectedSound] = []
        for name, columns in self._columns.items():
            score = float(scores[0, columns, :].max())
            if score < THRESHOLD:
                self._quiet_hops[name] += 1
                if self._quiet_hops[name] >= QUIET_HOPS_TO_END:
                    self._begun[name] = False
                continue
            self._quiet_hops[name] = 0
            if not self._begun[name]:
                self._begun[name] = True
                began.append(
                    DetectedSound(
                        name, score, window.audio_start_ms, window.audio_end_ms
                    )
                )
        return began


class SoundTaggers:
    """The shared sound model, its worker thread and one stream per asking session."""

    def __init__(
        self,
        directory: Path,
        open_model: Callable[[Path], InferenceSession] = one_thread_session,
    ) -> None:
        """Start unloaded; `open_model` opens the session over the model file."""
        self.directory = directory
        self._open_model = open_model
        self._session: InferenceSession | None = None
        self._columns: Mapping[SoundName, tuple[int, ...]] = {}
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sounds")

    def load(self) -> None:
        """Open the model and check its shapes and that it names every label mapped.

        Raises `SoundModelError` when it is not the model the stream feeds.
        """
        session = self._open_model(self.directory / MODEL_FILE)
        inputs = [(entry.name, list(entry.shape)) for entry in session.get_inputs()]
        if inputs != [(_INPUT, [1, WINDOW_SAMPLES])]:
            message = f"the sound model has inputs {inputs}"
            raise SoundModelError(message)
        labels = json.loads((self.directory / LABELS_FILE).read_text(encoding="utf-8"))
        columns = {
            name: tuple(_column(labels, label) for label in names)
            for name, names in _LABELS_OF.items()
        }
        silence = np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
        (answer,) = session.run(None, {_INPUT: silence})
        if answer.ndim != _SCORE_AXES or answer.shape[:2] != (1, len(labels)):
            message = f"the sound model answers scores of shape {answer.shape}"
            raise SoundModelError(message)
        self._columns = columns
        self._session = session

    def open_stream(self) -> SoundStream:
        """A stream at the start of one session's audio."""
        return SoundStream(self._columns)

    def submit(self, window: SoundWindow) -> Future[Scores]:
        """Score `window` on the worker; `RuntimeError` when closed or not loaded."""
        session = self._session
        if session is None:
            message = "the sound model is not loaded"
            raise RuntimeError(message)
        return self._worker.submit(_scored, session, window)

    def close(self) -> None:
        """Take no new window, wait for the running one, then drop the session."""
        self._worker.shutdown(wait=True, cancel_futures=True)
        self._session = None


def _column(labels: list[str], label: str) -> int:
    if label not in labels:
        message = f"the sound model's labels lack {label!r}"
        raise SoundModelError(message)
    return labels.index(label)


def _scored(session: InferenceSession, window: SoundWindow) -> Scores:
    (scores,) = session.run(None, {_INPUT: window.samples})
    return scores
