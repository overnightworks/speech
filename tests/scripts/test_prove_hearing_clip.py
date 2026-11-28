"""The hearing proofs' German clip is made through the speak route of a served voice."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import hearing_figures as figures
from shipped_catalogue import CATALOGUE
from speech_service.service import Runtime, RuntimeDependencies
from tests.support.fakes_support import FakeSpeaking
from tests.support.identity_support import callers_trusting
from tests.support.script_support import script_module
from tests.support.tcp_server_support import serving_over_tcp
from wav_audio import pcm_from_wav

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any

    from proof_identity import ProofIdentity
    from proof_minting import Minting

HEARING = script_module("prove_hearing")
LINES = script_module("voice_lines")
CLIENT = script_module("hearing_client")
SPEECH = Path(__file__).parents[1] / "fixtures" / "speech" / "nein_danke.wav"
SPEECH_RATE, SPEECH_PCM = pcm_from_wav(SPEECH.read_bytes())


class SpeakingARecording(FakeSpeaking):
    """A voice that says the recorded German speech for any text."""

    sample_rate = SPEECH_RATE

    def pcm_chunks(
        self, text: str, language: str, instructions: str | None = None
    ) -> Iterator[bytes]:
        """The recording in one chunk, whatever the request."""
        del text, language, instructions
        yield SPEECH_PCM


async def clip_made(
    minting: Minting, identity: ProofIdentity, make: Callable[[Any], bytes]
) -> bytes:
    """The clip `make` returns for a service serving the recorded voice."""
    runtime = Runtime(
        {HEARING.PIPER_PROOF_VOICE: SpeakingARecording()},
        memory_probe=lambda: 0,
        dependencies=RuntimeDependencies(catalogue=CATALOGUE),
    )
    callers = callers_trusting(identity.configuration_path, minting)
    async with serving_over_tcp(runtime, callers) as server:
        service = CLIENT.Service(server.base_url, minting, None)
        return await asyncio.to_thread(make, service)


def german_clip(minting: Minting, identity: ProofIdentity) -> bytes:
    return asyncio.run(
        clip_made(
            minting,
            identity,
            lambda service: HEARING.german_clip(service, HEARING.GERMAN_LONG_TURN),
        )
    )


def test_the_german_clip_joins_the_speech_of_each_sentence(
    minting: Minting, identity: ProofIdentity
) -> None:
    clip = german_clip(minting, identity)

    sentences = HEARING.GERMAN_LONG_TURN.count(". ") + 1
    assert 0 < figures.seconds_of(clip) <= sentences * len(SPEECH_PCM) / SPEECH_RATE


@pytest.mark.parametrize("longer_by", [1.0, 0.0], ids=["past one clip", "one clip"])
def test_a_clip_of_a_given_length_is_exactly_that_many_seconds(
    minting: Minting, identity: ProofIdentity, longer_by: float
) -> None:
    one_clip = figures.seconds_of(german_clip(minting, identity))
    seconds = one_clip + longer_by

    clip = asyncio.run(
        clip_made(minting, identity, lambda service: LINES.clip_of(service, seconds))
    )

    assert figures.seconds_of(clip) == pytest.approx(seconds, abs=1e-4)
