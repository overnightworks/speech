"""Hold the card's memory until stopped: the taken card of the hearing stage proof.

Allocates on the first CUDA device in large steps, then small ones, until the
card refuses, prints `held <mebibytes>`, and waits for TERM. The driver frees
the memory when the process ends, so nothing stays held after a stop or a kill.
"""

# ruff: noqa: T201 - the proof reads the held size from this command's output

from __future__ import annotations

import threading

import ctranslate2
import numpy as np

from speech_runtime_service.cuda_libs import prepare_cuda_libraries

MIB = 1024 * 1024
STEPS_MIB = (512, 16)


def hold_card(blocks: list[ctranslate2.StorageView]) -> int:
    """Take card memory into `blocks` until an allocation fails; the mebibytes held."""
    total_mib = 0
    for step_mib in STEPS_MIB:
        block = np.zeros(step_mib * MIB, dtype=np.int8)
        while True:
            try:
                blocks.append(
                    ctranslate2.StorageView.from_array(block).to_device(
                        ctranslate2.Device.cuda
                    )
                )
            except RuntimeError:
                break
            total_mib += step_mib
    return total_mib


def main() -> None:
    """Hold the card and say how much, then wait to be told to stop."""
    prepare_cuda_libraries()
    blocks: list[ctranslate2.StorageView] = []
    print("held", hold_card(blocks), flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
