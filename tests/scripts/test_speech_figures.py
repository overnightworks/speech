"""The figures a voice is held to in the proofs."""

from __future__ import annotations

import pytest

from speech_runtime_service.voices import VoiceId
from tests.support.script_support import script_module

FIGURES = script_module("speech_figures")


@pytest.mark.parametrize("voice", list(VoiceId))
def test_every_voice_has_a_first_byte_limit(voice: VoiceId) -> None:
    assert FIGURES.FIRST_BYTE_LIMIT_S[voice] > 0


def test_qwen_s_first_byte_may_come_after_its_load_and_its_whole_sentence() -> None:
    measured = FIGURES.QWEN_G0

    assert (
        FIGURES.FIRST_BYTE_LIMIT_S[VoiceId.QWEN]
        > measured.load_seconds + measured.first_audio_seconds
    )
