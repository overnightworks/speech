"""The GPU-voice cells' proof that the service stayed up through a failure."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

import image_gpu_voice
from image_stack import REPOSITORY, Compose
from tests.support.docker_support import DockerOnPath

if TYPE_CHECKING:
    from pathlib import Path

STARTED = ("container", "2026-11-06T10:00:00Z", 0)


def inspected(started_at: str, restarts: int) -> str:
    return json.dumps([{"State": {"StartedAt": started_at}, "RestartCount": restarts}])


@pytest.mark.parametrize(
    ("after", "held"),
    [
        pytest.param(STARTED, True, id="the same container"),
        pytest.param(("recreated", STARTED[1], 0), False, id="another container"),
        pytest.param(
            ("container", "2026-11-06T10:05:00Z", 0), False, id="started again"
        ),
        pytest.param(("container", STARTED[1], 1), False, id="restarted by Compose"),
    ],
)
def test_only_the_container_that_started_and_never_restarted_has_stayed_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after: tuple[str, str, int],
    *,
    held: bool,
) -> None:
    container, started_at, restarts = after
    DockerOnPath(
        tmp_path,
        monkeypatch,
        {
            "ps --quiet": container,
            f"inspect {container}": inspected(started_at, restarts),
        },
    )
    stack = Compose((REPOSITORY / "compose.yaml",), {})

    if held:
        image_gpu_voice.require_not_restarted(stack, STARTED)
    else:
        with pytest.raises(SystemExit, match="restarted"):
            image_gpu_voice.require_not_restarted(stack, STARTED)
