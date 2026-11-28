"""Build the frame_mn10 sound model as ONNX, held to its PyTorch model.

PretrainedSED's frame_mn10 is the AudioSet sound tagger the service runs beside
its speech models. This script fetches its source files and its checkpoint, each
at a pinned sha256, converts the model to ONNX and writes the directory the
service reads: `frame_mn10.onnx`, `frame_mn10.onnx.data`, `labels.json` (the
upstream strong-class names, in output order) and the upstream `LICENSE`.

Nothing is written to the destination unless the ONNX output matches PyTorch's
within `TOLERANCE` on seeded noise and on every window of the committed
reference clip; the directory then appears whole, by one rename.

Run it as `uv run --locked --no-build --script <this file> <destination>` beside
its lock, `build_sound_model.py.lock`, which pins and hashes every package;
`--locked` refuses a header that drifted from the lock. The destination must not
exist.
"""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "coverage==7.17.0",
#     "numpy==2.5.4",
#     "onnx==1.23.2",
#     "onnxruntime==1.29.0",
#     "onnxscript==0.7.2",
#     "torch==2.11.0",
#     "torchaudio==2.11.0",
#     "torchvision==0.26.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchaudio = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
# ///

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

# onnxruntime reads the switch at import: set after it, its telemetry still connects.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import onnxruntime

if TYPE_CHECKING:
    import numpy.typing as npt

type Samples = npt.NDArray[np.float32]

SOURCE_URL: Final = (
    "https://raw.githubusercontent.com/fschmid56/PretrainedSED/{commit}/{path}"
)
SOURCE_COMMIT: Final = "1aa47e482f7e89904cba2338999345025d8b4e36"
# The import closure of PredictionsWrapper and FrameMNWrapper, and the licence.
SOURCE_SHA256: Final = {
    "LICENSE": "399acc6d903cfd65a6cd5baba6a99af3ee534d79161b07854c1a6ffeab3dc88b",
    "config.py": "10e4012076a9b884d1222c9b199b8bbd7a916775d46d1c2425c264d12cebe683",
    "data_util/audioset_classes.py": (
        "771adb63768b349171a1668b4b0c2f932f6db1b21e10450c7e9136c0397e076f"
    ),
    "models/frame_mn/Frame_MN_wrapper.py": (
        "c9f9be29be6cb5dfb94b03a0cb95a22c35519aaefabf913fdebc0e3fc2f40d77"
    ),
    "models/frame_mn/block_types.py": (
        "1dfa5636aadb209eeaa7e412d5fc422bfefe230b4b30f49a14fe11776eab825c"
    ),
    "models/frame_mn/model.py": (
        "6c316eab1744c3a6d45537532bd4dfc91718f7132f1de42e79a922c8c5b48077"
    ),
    "models/frame_mn/utils.py": (
        "b952d15279cf2fb21658e6916ce23e00e3338187564b4ad9c635a10dd261aac5"
    ),
    "models/frame_passt/preprocess.py": (
        "1ac324f812566924ab52879077186dcaffdc8a4062ee186153be48e99b658db0"
    ),
    "models/prediction_wrapper.py": (
        "53f37f7870a85e64242d07dd087ffb4c72a8a8413d43a593ab46bba85b3983eb"
    ),
    "models/seq_models.py": (
        "d7ac2ab967ce65e3dad7dd3cf717163a107e8cc0b57bdab0d775a72f075ca667"
    ),
    "models/transformer_wrapper.py": (
        "8ca7284475e47dc00a5f476bd8d937400e35c95ae3164b8fd9111505c4a4dee8"
    ),
}
CHECKPOINT: Final = "frame_mn10_strong_1"
CHECKPOINT_URL: Final = (
    "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/"
    f"{CHECKPOINT}.pt"
)
CHECKPOINT_SHA256: Final = (
    "4a0fe320d5369987b772394c51881fb20a602967e6842f72f3e5c8181065ece7"
)

SAMPLE_RATE: Final = 16_000
WINDOW_SAMPLES: Final = 2 * SAMPLE_RATE
HOP_SAMPLES: Final = SAMPLE_RATE // 2
TOLERANCE: Final = 1e-4
NOISE_SEED: Final = 20260521
NOISE_WINDOWS: Final = 4
NOISE_SCALE: Final = 0.1
REFERENCE_CLIP: Final = Path(__file__).parent / "sound_reference" / "laughter.wav"
MODEL_NAME: Final = "frame_mn10.onnx"


def download_pinned(url: str, sha256: str) -> bytes:
    """The bytes at the https `url`, refused unless they hash to `sha256`."""
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - pinned https
        content = response.read()
    if hashlib.sha256(content).hexdigest() != sha256:
        message = f"{url} is not the pinned {sha256}"
        raise RuntimeError(message)
    return content


def fetch_source(source: Path) -> None:
    """Every pinned source file and the checkpoint, under `source`."""
    for path, sha256 in SOURCE_SHA256.items():
        url = SOURCE_URL.format(commit=SOURCE_COMMIT, path=path)
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(download_pinned(url, sha256))
    resources = source / "resources"
    resources.mkdir()
    (resources / f"{CHECKPOINT}.pt").write_bytes(
        download_pinned(CHECKPOINT_URL, CHECKPOINT_SHA256)
    )


def class_names(source: Path) -> list[str]:
    """PretrainedSED's strong class names, in the model's output order."""
    spec = importlib.util.spec_from_file_location(
        "audioset_classes", source / "data_util" / "audioset_classes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.as_strong_train_classes)


def pytorch_model(source: Path):  # noqa: ANN201 - torch stays lazy
    """The repository's model with its checkpoint, mel front end included."""
    sys.path.insert(0, str(source))
    import models.prediction_wrapper as wrapper  # noqa: PLC0415 - the source path is added just above
    from models.frame_mn.Frame_MN_wrapper import FrameMNWrapper  # noqa: PLC0415 - the source path is added just above

    # The repository reads its checkpoint folder from a module constant, and
    # prints its network and its checkpoint where it builds them.
    wrapper.RESOURCES_FOLDER = str(source / "resources")
    with contextlib.redirect_stdout(sys.stderr):
        frame_mn = FrameMNWrapper(1.0)
        embed_dim = frame_mn.state_dict()["frame_mn.features.16.1.bias"].shape[0]
        return wrapper.PredictionsWrapper(
            frame_mn, checkpoint=CHECKPOINT, embed_dim=embed_dim
        ).eval()


def convert(model, onnx: Path) -> None:  # noqa: ANN001 - torch stays lazy
    """Export the model's scores for one window to `onnx`."""
    import torch  # noqa: PLC0415 - only the conversion and its check need it

    class Scores(torch.nn.Module):
        def forward(self, window):  # noqa: ANN001, ANN202 - torch stays lazy
            return torch.sigmoid(model(model.mel_forward(window))[0])

    # The export prints its progress; the build's own failures are the output.
    with contextlib.redirect_stdout(sys.stderr):
        torch.onnx.export(
            Scores().eval(),
            (torch.zeros(1, WINDOW_SAMPLES),),
            str(onnx),
            input_names=["waveform"],
            output_names=["scores"],
            opset_version=18,
            dynamo=True,
        )


def reference_windows() -> list[Samples]:
    """Seeded noise windows, then every window of the reference clip."""
    generator = np.random.default_rng(NOISE_SEED)
    noise = generator.standard_normal((NOISE_WINDOWS, WINDOW_SAMPLES)) * NOISE_SCALE
    with wave.open(str(REFERENCE_CLIP), "rb") as clip:
        if (clip.getframerate(), clip.getnchannels(), clip.getsampwidth()) != (
            SAMPLE_RATE,
            1,
            2,
        ):
            message = f"{REFERENCE_CLIP} is not 16 kHz mono PCM16"
            raise RuntimeError(message)
        samples = np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2")
    audio = samples.astype(np.float32) / 32768
    starts = range(0, len(audio) - WINDOW_SAMPLES + 1, HOP_SAMPLES)
    return [*noise.astype(np.float32), *(audio[s : s + WINDOW_SAMPLES] for s in starts)]


def score_difference(found: Samples, expected: Samples, labels: int) -> float:
    """The largest gap between two scores; a malformed pair is refused."""
    if found.shape != expected.shape or found.shape[1] != labels:
        message = f"the ONNX scores are {found.shape}, PyTorch's {expected.shape}"
        raise RuntimeError(message)
    difference = np.abs(found - expected)
    if not np.isfinite(difference).all():
        message = "the ONNX or PyTorch scores hold a non-finite value"
        raise RuntimeError(message)
    return float(difference.max())


def worst_difference(model, onnx: Path, windows: list[Samples], labels: int) -> float:  # noqa: ANN001 - torch stays lazy
    """The largest score difference between the ONNX and the PyTorch model."""
    import torch  # noqa: PLC0415 - torch stays lazy

    options = onnxruntime.SessionOptions()
    options.log_severity_level = 4
    session = onnxruntime.InferenceSession(
        str(onnx), sess_options=options, providers=["CPUExecutionProvider"]
    )
    worst = 0.0
    for window in windows:
        (found,) = session.run(None, {"waveform": window[None]})
        with torch.no_grad():
            expected = torch.sigmoid(
                model(model.mel_forward(torch.from_numpy(window[None])))[0]
            ).numpy()
        worst = max(worst, score_difference(found, expected, labels))
    return worst


def build(destination: Path) -> None:
    """Write the model directory to `destination`, whole or not at all."""
    if destination.exists():
        message = f"{destination} exists"
        raise RuntimeError(message)
    windows = reference_windows()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The staging directory is a sibling, so its rename is one step on one volume.
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}-"
    ) as staging:
        source = Path(staging) / "source"
        output = Path(staging) / "model"
        output.mkdir()
        fetch_source(source)
        labels = class_names(source)
        model = pytorch_model(source)
        convert(model, output / MODEL_NAME)
        worst = worst_difference(model, output / MODEL_NAME, windows, len(labels))
        if worst > TOLERANCE:
            message = (
                f"the ONNX differs from PyTorch by {worst:.1e} (limit {TOLERANCE:g})"
            )
            raise RuntimeError(message)
        (output / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
        (output / "LICENSE").write_bytes((source / "LICENSE").read_bytes())
        output.rename(destination)


if __name__ == "__main__":
    if len(sys.argv) != 2:  # noqa: PLR2004 - the one destination argument
        sys.exit(f"usage: {Path(sys.argv[0]).name} <destination>")
    build(Path(sys.argv[1]).resolve())
