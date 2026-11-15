"""The sound taggers the probe measures, each as ONNX on one thread.

A candidate fetches its pinned files, opens them, scores a window of 16 kHz audio
for every AudioSet label and says whether its ONNX output matches its reference.
Files come from the Hugging Face cache, or from `cache` for what is not on the Hub.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import shutil
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np
import soundfile
import soxr
from huggingface_hub import hf_hub_download

from onnx_session import one_thread_session
from speech_service.piper_voices import sha256_of

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy.typing as npt

type Samples = npt.NDArray[np.float32]

SAMPLE_RATE: Final = 16_000
HOP_SAMPLES: Final = SAMPLE_RATE // 2
HOP_SECONDS: Final = HOP_SAMPLES / SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class Check:
    """Whether a candidate's ONNX output matches its reference."""

    passed: bool
    detail: str


class Tagger(Protocol):
    """A candidate: its files, then a loaded model that scores a window of audio."""

    title: str
    licence: str
    window_samples: int
    files: tuple[Path, ...]
    labels: tuple[str, ...]

    def load(self) -> None:
        """Open the model on one thread."""

    def scores(self, window: Samples) -> npt.NDArray[np.float32]:
        """The score from 0 to 1 of every label for `window`."""

    def check(self) -> Check:
        """Compare the model with its reference."""


@dataclass(frozen=True, slots=True)
class HopCost:
    """What one hop cost the thread that ran it, in seconds."""

    cpu: float
    wall: float


def windows_of(samples: Samples, size: int) -> list[Samples]:
    """The windows a recording is scored in: one per hop, a short one padded."""
    padded = np.pad(samples, (0, max(0, size - len(samples))))
    last = len(padded) - size + 1
    return [padded[start : start + size] for start in range(0, last, HOP_SAMPLES)]


def timed(tagger: Tagger, window: Samples) -> tuple[npt.NDArray[np.float32], HopCost]:
    """The scores of `window` and what they cost this thread."""
    cpu, wall = time.thread_time(), time.perf_counter()
    scores = tagger.scores(window)
    return scores, HopCost(time.thread_time() - cpu, time.perf_counter() - wall)


def _labels_of(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as source:
        return tuple(row["display_name"] for row in csv.DictReader(source))


def hub_file(repository: str, name: str, revision: str, kind: str = "model") -> Path:
    """The pinned file of a Hub repository, in the Hugging Face cache."""
    return Path(hf_hub_download(repository, name, repo_type=kind, revision=revision))


def fetch_pinned(cache: Path, url: str, name: str, sha256: str | None = None) -> Path:
    """The file at the https `url` in `cache`, downloaded once.

    A file with a `sha256` is held to it; a URL that names a commit needs none.
    """
    target = cache / name
    if target.is_file() and (sha256 is None or sha256_of(target) == sha256):
        return target
    cache.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as out:  # noqa: S310 - pinned https
        shutil.copyfileobj(response, out)
    if sha256 is not None and sha256_of(part) != sha256:
        part.unlink()
        message = f"{url} is not the pinned {sha256}"
        raise RuntimeError(message)
    part.replace(target)
    return target


def unpack(archive: Path, directory: Path) -> Path:
    """`directory` beside `archive`, unpacked from it once."""
    if not directory.is_dir():
        with tarfile.open(archive) as bundle:
            bundle.extractall(directory.parent, filter="data")
    return directory


class Yamnet:
    """YAMNet through an ONNX conversion, held to Google's float TFLite export."""

    title = "YAMNet"
    licence = "Apache-2.0"
    window_samples = 15600
    tolerance = 1e-3
    _onnx = ("audiomagic/yamnet-onnx", "f25b741c2f0bdc6d7e6db24b5fddda23347dbafd")
    _reference = ("thelou1s/yamnet", "156527da8fb34f53a3798d6011810fa975738062")

    def __init__(self, cache: Path, windows: Sequence[Samples]) -> None:
        """Fetch the ONNX and its label map; `windows` are the real ones to check on."""
        del cache
        repository, revision = self._onnx
        self.files = (
            hub_file(repository, "yamnet.onnx", revision),
            hub_file(repository, "yamnet_class_map.csv", revision),
        )
        self.labels = _labels_of(self.files[1])
        self._windows = windows

    def load(self) -> None:
        """Open the ONNX."""
        self._session = one_thread_session(self.files[0])

    def scores(self, window: Samples) -> npt.NDArray[np.float32]:
        """The scores of the one 0.975 s patch."""
        return self._session.run(None, {"waveform": window})[0][0]

    def check(self) -> Check:
        """The ONNX scores against the TFLite scores of the same windows."""
        from ai_edge_litert.interpreter import Interpreter  # noqa: PLC0415

        repository, revision = self._reference
        name = "lite-model_yamnet_tflite_1.tflite"
        reference = Interpreter(
            model_path=str(hub_file(repository, name, revision)), num_threads=1
        )
        slot = reference.get_input_details()[0]["index"]
        reference.resize_tensor_input(slot, [self.window_samples])
        reference.allocate_tensors()
        worst = 0.0
        for window in self._windows:
            window = window[: self.window_samples]  # noqa: PLW2901 - the patch the reference takes
            reference.set_tensor(slot, window)
            reference.invoke()
            expected = reference.get_tensor(reference.get_output_details()[0]["index"])
            worst = max(worst, float(np.abs(self.scores(window) - expected[0]).max()))
        return Check(
            worst <= self.tolerance,
            f"largest score difference to Google's TFLite export {worst:.1e}"
            f" (limit {self.tolerance:g})",
        )


class Zipformer:
    """The sherpa-onnx zipformer small audio tagger, held to its documentation.

    Only the 8-bit file is measured: the publisher's numbers come from it, so the
    32-bit file has no reference to be held to.
    """

    licence = "Apache-2.0"
    window_samples = 2 * SAMPLE_RATE
    _directory = "sherpa-onnx-zipformer-small-audio-tagging-2024-04-15"
    _url = (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        f"audio-tagging-models/{_directory}.tar.bz2"
    )
    _sha256 = "07e2fafcdcbc461f2816188d9b0bbafced12584030cf67d5652e549ef256a2c6"
    _model_name = "model.int8.onnx"
    tolerance = 0.02
    # The first result the publisher documents for its test wave N (sherpa-onnx,
    # audio tagging, pretrained models), produced with the 8-bit file.
    documented = {  # noqa: RUF012 - read only
        1: ("Animal", 0.947886),
        2: ("Whistling", 0.804928),
        3: ("Music", 0.79673),
        4: ("Laughter", 0.929239),
        5: ("Finger snapping", 0.690543),
        6: ("Baby cry, infant cry", 0.912273),
        7: ("Smoke detector, smoke alarm", 0.781478),
        8: ("Siren", 0.877108),
        10: ("Stream", 0.247785),
        11: ("Meow", 0.814944),
        12: ("Animal", 0.688237),
        13: ("Oink", 0.888416),
    }

    title = "sherpa-onnx zipformer small, 8-bit"

    def __init__(self, cache: Path) -> None:
        """Unpack the pinned release."""
        archive = fetch_pinned(
            cache, self._url, f"{self._directory}.tar.bz2", self._sha256
        )
        directory = unpack(archive, cache / self._directory)
        self.files = (
            directory / self._model_name,
            directory / "class_labels_indices.csv",
        )
        self.labels = _labels_of(self.files[1])

    def load(self) -> None:
        """Open the model through sherpa-onnx's own runtime, asking for all labels."""
        import sherpa_onnx  # noqa: PLC0415 - only this candidate needs it

        model, labels = self.files
        zipformer = sherpa_onnx.OfflineZipformerAudioTaggingModelConfig(
            model=str(model)
        )
        self._tagger = sherpa_onnx.AudioTagging(
            sherpa_onnx.AudioTaggingConfig(
                model=sherpa_onnx.AudioTaggingModelConfig(
                    zipformer=zipformer, num_threads=1, provider="cpu"
                ),
                labels=str(labels),
                top_k=len(self.labels),
            )
        )

    def scores(self, window: Samples) -> npt.NDArray[np.float32]:
        """The window's scores."""
        stream = self._tagger.create_stream()
        stream.accept_waveform(SAMPLE_RATE, window)
        scores = np.zeros(len(self.labels), dtype=np.float32)
        for event in self._tagger.compute(stream):
            scores[event.index] = event.prob
        return scores

    def check(self) -> Check:
        """The first result on each test wave of the release, against the documented."""
        wrong, worst = [], 0.0
        for number, (name, probability) in self.documented.items():
            wave = self.files[0].parent / "test_wavs" / f"{number}.wav"
            scores = self.scores(read_audio(wave))
            best = int(scores.argmax())
            worst = max(worst, abs(float(scores[best]) - probability))
            if self.labels[best] != name:
                wrong.append(number)
        return Check(
            not wrong and worst <= self.tolerance,
            (
                f"the first result is wrong on test waves {wrong}"
                if wrong
                else "the first result is right on every test wave"
            )
            + f"; largest probability difference to the publisher's documented"
            f" value {worst:.3f} (limit {self.tolerance:g})",
        )


class FrameMn:
    """PretrainedSED frame_mn10, converted to ONNX here, held to its PyTorch model."""

    title = "PretrainedSED frame_mn10"
    licence = "MIT (repository; the checkpoint states no other)"
    window_samples = 2 * SAMPLE_RATE
    tolerance = 1e-4
    _commit = "1aa47e482f7e89904cba2338999345025d8b4e36"
    _checkpoint = "frame_mn10_strong_1"

    def __init__(self, cache: Path, windows: Sequence[Samples]) -> None:
        """Unpack the pinned source, fetch the checkpoint and convert it when absent."""
        archive = fetch_pinned(
            cache,
            f"https://codeload.github.com/fschmid56/PretrainedSED/tar.gz/{self._commit}",
            f"PretrainedSED-{self._commit}.tar.gz",
        )
        self._source = unpack(archive, cache / f"PretrainedSED-{self._commit}")
        self._weights = fetch_pinned(
            cache,
            "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/"
            f"{self._checkpoint}.pt",
            f"{self._checkpoint}.pt",
            "4a0fe320d5369987b772394c51881fb20a602967e6842f72f3e5c8181065ece7",
        )
        onnx = cache / "frame_mn10.onnx"
        if not onnx.is_file():
            self._convert(onnx)
        self.files = (onnx, Path(f"{onnx}.data"))
        self.labels = self._label_names()
        self._windows = windows

    def _label_names(self) -> tuple[str, ...]:
        spec = importlib.util.spec_from_file_location(
            "audioset_classes", self._source / "data_util" / "audioset_classes.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return tuple(module.as_strong_train_classes)

    def _model(self):  # noqa: ANN202 - torch stays lazy
        """The repository's model with its checkpoint, mel front end included."""
        sys.path.insert(0, str(self._source))
        import models.prediction_wrapper as wrapper  # noqa: PLC0415
        from models.frame_mn.Frame_MN_wrapper import FrameMNWrapper  # noqa: PLC0415

        # The repository reads its checkpoint folder from a module constant, and
        # prints its network and its checkpoint where it builds them.
        wrapper.RESOURCES_FOLDER = str(self._weights.parent)
        with contextlib.redirect_stdout(sys.stderr):
            frame_mn = FrameMNWrapper(1.0)
            embed_dim = frame_mn.state_dict()["frame_mn.features.16.1.bias"].shape[0]
            return wrapper.PredictionsWrapper(
                frame_mn, checkpoint=self._checkpoint, embed_dim=embed_dim
            ).eval()

    def _convert(self, onnx: Path) -> None:
        import torch  # noqa: PLC0415 - only the conversion and its check need it

        model = self._model()

        class Whole(torch.nn.Module):
            def forward(self, window):  # noqa: ANN001, ANN202
                return torch.sigmoid(model(model.mel_forward(window))[0])

        # The export prints its progress; the report owns standard output.
        with contextlib.redirect_stdout(sys.stderr):
            torch.onnx.export(
                Whole().eval(),
                (torch.zeros(1, self.window_samples),),
                str(onnx),
                input_names=["waveform"],
                output_names=["scores"],
                opset_version=18,
                dynamo=True,
            )

    def load(self) -> None:
        """Open the ONNX."""
        self._session = one_thread_session(self.files[0])

    def scores(self, window: Samples) -> npt.NDArray[np.float32]:
        """Per label the highest score over the frames of the window."""
        return self._session.run(None, {"waveform": window[None]})[0][0].max(axis=1)

    def check(self) -> Check:
        """The ONNX scores against the PyTorch model's on the same windows."""
        import torch  # noqa: PLC0415

        model = self._model()
        worst = 0.0
        for window in self._windows:
            with torch.no_grad():
                frames = torch.sigmoid(
                    model(model.mel_forward(torch.from_numpy(window[None])))[0]
                )
            expected = frames[0].numpy().max(axis=1)
            worst = max(worst, float(np.abs(self.scores(window) - expected).max()))
        return Check(
            worst <= self.tolerance,
            f"largest score difference to the PyTorch model {worst:.1e}"
            f" (limit {self.tolerance:g})",
        )


CANDIDATES: Final[Mapping[str, Callable[[Path, Sequence[Samples]], Tagger]]] = {
    "yamnet": Yamnet,
    "zipformer-small-int8": lambda cache, _: Zipformer(cache),
    "frame-mn10": FrameMn,
}


def read_audio(source: Path | bytes) -> Samples:
    """A recording as 16 kHz mono float samples; an unreadable one fails loud."""
    samples, rate = soundfile.read(
        io.BytesIO(source) if isinstance(source, bytes) else source,
        dtype="float32",
        always_2d=True,
    )
    mono = samples.mean(axis=1)
    if rate == SAMPLE_RATE:
        return mono
    return soxr.resample(mono, rate, SAMPLE_RATE).astype(np.float32)
