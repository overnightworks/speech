"""Write the browser package's copy of the hop: its frame format and close codes.

`speech_client.wire` owns the frame format and `speech_client.relay` the hop's
frame size and close codes; this generator renders them into one TypeScript
module. `--check` writes nothing and fails when the committed module differs
from the rendering or is missing, so CI catches drift.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from speech_client.relay import HOP_FRAME_BYTES, HOP_FRAME_SAMPLES, HopClose
from speech_client.wire import PcmFormat, SessionInput

if TYPE_CHECKING:
    from collections.abc import Sequence

REPOSITORY = Path(__file__).resolve().parent.parent
OUTPUT = Path("packages/browser/src/hop.ts")
CHECK_FLAG = "--check"
COMMAND = "uv run --locked python scripts/generate_browser_wire.py"
INDENT = "  "


def _constant(name: str, value: int) -> str:
    return f"export const {name} = {value};\n"


def _close_codes() -> str:
    members = "".join(f"{INDENT}{code.name}: {code.value},\n" for code in HopClose)
    return (
        f"export const HopClose = {{\n{members}}} as const;\n\n"
        "export type HopClose = (typeof HopClose)[keyof typeof HopClose];\n"
    )


def render() -> str:
    """The module's whole text, read from the Python owners."""
    constants = "".join(
        _constant(name, value)
        for name, value in (
            ("SAMPLE_RATE", SessionInput().format.rate),
            ("SAMPLE_BYTES", PcmFormat.sample_bytes),
            ("CHANNELS", PcmFormat.channels),
            ("HOP_FRAME_SAMPLES", HOP_FRAME_SAMPLES),
            ("HOP_FRAME_BYTES", HOP_FRAME_BYTES),
        )
    )
    header = (
        "// Generated from speech_client.wire and speech_client.relay"
        " by scripts/generate_browser_wire.py. Do not edit.\n"
        f"// Rewrite it with: {COMMAND}\n"
    )
    return f"{header}\n{constants}\n{_close_codes()}"


def main(arguments: Sequence[str], root: Path = REPOSITORY) -> int:
    """Write the module under `root`, or with `--check` only compare it."""
    path = root / OUTPUT
    content = render()
    if CHECK_FLAG in arguments:
        if not path.is_file() or path.read_text() != content:
            message = f"generate_browser_wire: {path} out of date; run {COMMAND}"
            raise SystemExit(message)
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
