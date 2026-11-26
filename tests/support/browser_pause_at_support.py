"""What the browser cells of the player's hold at a position share.

The cells throw `pauseAt` of the player (`packages/browser`) on a stepped stage and read
what the page saw: the call's result, how the sound it held ended, and the audio the
context rendered.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tests.support.browser_player_support import (
    audible_seconds,
    last_ramp_end,
    quantum_seconds,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.support.browser_player_support import SteppedStage

# A position well inside the long sound and past the first chunk of its streamed form.
HOLD_AT = 2.0


def held_at(position: float) -> dict[str, object]:
    return {"kind": "held", "playedSeconds": position}


def silent_from(seen: dict[str, object], at: float) -> bool:
    """Whether every quantum that starts from `at` of the clock was silent."""
    return audible_seconds(seen, at, math.inf) == 0


async def mute(stage: SteppedStage) -> None:
    await stage.set_muted(muted=True)


async def mute_during_the_fade(stage: SteppedStage) -> None:
    thrown = await stage.snapshot()
    two_quanta = 2 * quantum_seconds(thrown)
    await stage.step_for(last_ramp_end(thrown, 0) - two_quanta - thrown["contextNow"])
    await mute(stage)


# The switch the app throws while a call waits, and how the sound it ends reads.
SWITCHES: dict[str, tuple[Callable[[SteppedStage], object], str]] = {
    "stop": (lambda stage: stage.act("stop"), "stopped"),
    "mute": (mute, "muted"),
    "mute_during_the_fade": (mute_during_the_fade, "muted"),
    "close": (lambda stage: stage.act("close"), "stopped"),
}
