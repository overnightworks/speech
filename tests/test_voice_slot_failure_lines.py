"""A failed voice load is logged at ERROR with its voice and error class."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from shipped_catalogue import CATALOGUE
from speech_client.wire import VoiceId
from speech_service.clock import LoopClock
from speech_service.service import RuntimeDependencies, SpeakingEngine
from speech_service.voice_slot import VoiceSlot
from tests.support.fakes_support import GATE_TIMEOUT_SECONDS, GatedEngine

if TYPE_CHECKING:
    from collections.abc import Callable

_EVENT = "voice_load.failed"


class _FailingLoad(GatedEngine):
    def load(self) -> None:
        message = "the weights are unreadable"
        raise OSError(message)


def _slot() -> VoiceSlot:
    def build(_voice: VoiceId) -> SpeakingEngine:
        return _FailingLoad()

    return VoiceSlot(
        CATALOGUE[VoiceId.THORSTEN],
        RuntimeDependencies(
            catalogue=CATALOGUE,
            engine_factory=build,
            missing_artifacts=lambda _voice: (),
        ),
        LoopClock(),
    )


def _load_directly(slot: VoiceSlot) -> None:
    with pytest.raises(RuntimeError, match="failed to load"):
        slot.load()


def _load_shared(slot: VoiceSlot) -> None:
    job = slot.shared_load()
    assert job is not None
    assert job.result(GATE_TIMEOUT_SECONDS) is None


@pytest.mark.parametrize(
    ("drive", "error_types"),
    [(_load_directly, [None]), (_load_shared, [None, "RuntimeError"])],
    ids=["direct-load-names-the-voice", "shared-load-adds-the-error-class"],
)
def test_a_failed_voice_load_is_logged_at_error_with_its_voice(
    caplog: pytest.LogCaptureFixture,
    drive: Callable[[VoiceSlot], None],
    error_types: list[str | None],
) -> None:
    with caplog.at_level(logging.DEBUG):
        drive(_slot())

    failed = [record for record in caplog.records if record.getMessage() == _EVENT]
    assert [record.levelno for record in failed] == [logging.ERROR] * len(error_types)
    assert {record.voice for record in failed} == {VoiceId.THORSTEN.value}
    assert [getattr(record, "error_type", None) for record in failed] == error_types
