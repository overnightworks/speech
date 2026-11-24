"""The player stage counts the quanta the output never rendered."""

from __future__ import annotations

import pytest

from tests.support.browser_player_support import RENDER_QUANTUM_FRAMES, dropped_seconds

SAMPLE_RATE = 48_000
QUANTUM = RENDER_QUANTUM_FRAMES / SAMPLE_RATE


def recording(*quanta: int) -> dict[str, object]:
    """A recording whose ear reported a quantum at each given index of the clock."""
    return {
        "sampleRate": SAMPLE_RATE,
        "heardPeaks": [[index * QUANTUM, 0.5] for index in quanta],
    }


@pytest.mark.parametrize(
    ("quanta", "since", "until", "dropped_quanta"),
    [
        pytest.param(range(10), 0, 9, 0, id="no_jump"),
        pytest.param(
            [*range(5), *range(9, 12)], 0, 11, 4, id="a_jump_of_several_quanta"
        ),
        pytest.param(
            [*range(5), 5, 5, *range(7, 10)], 0, 9, 0, id="a_duplicate_nets_a_jump"
        ),
        pytest.param(
            [*range(5), 5, 5, *range(8, 10)],
            0,
            9,
            1,
            id="a_duplicate_nets_part_of_a_jump",
        ),
        pytest.param(range(10), 20, 30, 0, id="no_quanta_in_the_window"),
        pytest.param(
            [*range(5), *range(8, 10)],
            0,
            4,
            0,
            id="a_jump_after_the_window_is_not_counted",
        ),
        pytest.param(
            [*range(3), *range(5, 8)], 2, 5, 2, id="both_edges_belong_to_the_window"
        ),
        pytest.param(
            [0, 1, 3, 4],
            1,
            3,
            1,
            id="a_jump_inside_a_window_narrower_than_the_recording",
        ),
    ],
)
def test_dropped_seconds_counts_elapsed_quanta_less_reported_ones(
    quanta: list[int], since: int, until: int, dropped_quanta: int
) -> None:
    seen = recording(*quanta)

    dropped = dropped_seconds(seen, since * QUANTUM, until * QUANTUM)

    assert dropped == pytest.approx(dropped_quanta * QUANTUM)
