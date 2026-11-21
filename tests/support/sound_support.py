"""A sound model at the port: a model directory a test builds, and scripted scores."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from speech_service.sound_events import (
    LABELS_FILE,
    MODEL_FILES,
    WINDOW_SAMPLES,
    SoundTaggers,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

WAIT_SECONDS = 5.0
FRAMES = 3
LABELS = (
    "Speech",
    "Laughter",
    "Giggle",
    "Snicker",
    "Belly laugh",
    "Cough",
    "Clapping",
    "Applause",
    "Screaming",
    "Throat clearing",
    "Sigh",
)


def write_model_directory(
    directory: Path, labels: Sequence[str] = LABELS, *, skip: str | None = None
) -> Path:
    """The files of a built sound model in `directory`, except `skip`."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        if name != skip:
            (directory / name).write_text("model")
    if skip != LABELS_FILE:
        (directory / LABELS_FILE).write_text(json.dumps(list(labels)))
    return directory


def scores_of(
    levels: Mapping[str, float], labels: Sequence[str] = LABELS
) -> np.ndarray:
    """Scores for one window: every label quiet but the `levels`, in its last frame."""
    scores = np.zeros((1, len(labels), FRAMES), dtype=np.float32)
    for name, level in levels.items():
        scores[0, labels.index(name), -1] = level
    return scores


class FakeSoundModel:
    """An ONNX session that answers the scripted scores of each window, in order.

    A window with no scripted answer is quiet. A run can be held, or made to fail.
    """

    def __init__(
        self,
        *,
        input_shape: Sequence[object] = (1, WINDOW_SAMPLES),
        answer_labels: Sequence[str] = LABELS,
    ) -> None:
        """Answer with `answer_labels`' columns."""
        self._input_shape = list(input_shape)
        self._labels = answer_labels
        self._script: list[Mapping[str, float]] = []
        self.windows: list[np.ndarray] = []
        self.failure: Exception | None = None
        self.thread_names: list[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def hold_runs(self) -> None:
        """Make every later run wait for `release`, announced by `entered`."""
        self.entered.clear()
        self.release.clear()

    def script(self, *answers: Mapping[str, float]) -> None:
        """Answer the next windows with these levels."""
        self._script.extend(answers)

    def get_inputs(self) -> list[SimpleNamespace]:
        """The one input, `waveform`."""
        return [SimpleNamespace(name="waveform", shape=self._input_shape)]

    def run(
        self, _outputs: object, feeds: Mapping[str, np.ndarray]
    ) -> list[np.ndarray]:
        """Record the window, wait for the release, answer or fail."""
        self.thread_names.append(threading.current_thread().name)
        self.entered.set()
        if not self.release.wait(WAIT_SECONDS):
            message = "the run was never released"
            raise TimeoutError(message)
        if self.failure is not None:
            raise self.failure
        self.windows.append(feeds["waveform"])
        levels = self._script.pop(0) if self._script else {}
        return [scores_of(levels, self._labels)]


def opener_of(model: FakeSoundModel) -> Callable[[Path], FakeSoundModel]:
    """A model opener that returns `model` whatever the path."""
    return lambda _path: model


def loaded_taggers(
    directory: Path, model: FakeSoundModel | None = None
) -> tuple[SoundTaggers, FakeSoundModel]:
    """Sound taggers over a written model directory and `model`, loaded."""
    model = model or FakeSoundModel()
    write_model_directory(directory)
    taggers = SoundTaggers(directory, opener_of(model))
    taggers.load()
    model.windows.clear()
    return taggers, model
