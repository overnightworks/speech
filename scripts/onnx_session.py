"""The ONNX session the measuring scripts open: CPU, one thread."""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnxruntime

if TYPE_CHECKING:
    from pathlib import Path


def one_thread_session(path: Path) -> onnxruntime.InferenceSession:
    """An ONNX session over `path` on the CPU with one thread."""
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
