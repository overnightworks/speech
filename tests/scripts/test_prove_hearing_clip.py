"""The hearing proofs' German clip is made through the speak route of a served voice."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

import hearing_figures as figures
from shipped_catalogue import CATALOGUE
from speech_service.service import Runtime, RuntimeDependencies
from speech_service.voices import piper_voice_paths
from tests.support.fakes_support import FakeSpeaking
from tests.support.hear_support import NEIN_DANKE, VoiceWithDurationNoise
from tests.support.hearing_service_support import service_of
from tests.support.script_support import script_module
from wav_audio import pcm_from_wav

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from typing import Any

    from proof_identity import ProofIdentity
    from proof_minting import Minting

HEARING = script_module("prove_hearing")
LINES = script_module("voice_lines")
SPEECH_RATE, SPEECH_PCM = pcm_from_wav(NEIN_DANKE.read_bytes())


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
    async with service_of(runtime, minting, identity) as service:
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
    sentences = len(HEARING.SENTENCE_BOUNDARY.split(HEARING.GERMAN_LONG_TURN))
    one_sentence = asyncio.run(
        clip_made(
            minting,
            identity,
            lambda service: HEARING.german_clip(service, "Nein, danke."),
        )
    )

    clip = german_clip(minting, identity)

    assert len(clip) == sentences * len(one_sentence)


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


@pytest.fixture
def english_voice_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "voices"
    cache.mkdir()
    monkeypatch.setenv("SPEECH_VOICE_CACHE", str(cache))
    return cache


def test_both_languages_get_a_clip_that_is_one_turn(
    minting: Minting,
    identity: ProofIdentity,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clips = asyncio.run(
        clip_made(
            minting,
            identity,
            lambda service: HEARING.make_clips(service, VoiceWithDurationNoise),
        )
    )

    assert set(clips) == {language.code for language in HEARING.LANGUAGES}
    printed = capsys.readouterr().out.splitlines()
    measured = [line.split()[:2] for line in printed if "long_turn_clip" in line]
    assert measured == [["long_turn_clip", language.value] for language in clips]


def test_an_english_voice_in_the_cache_is_the_path_the_clip_loads(
    english_voice_cache: Path,
) -> None:
    onnx, _metadata = piper_voice_paths(english_voice_cache, HEARING.ENGLISH_VOICE)
    onnx.touch()

    assert HEARING.english_voice_path() == onnx


@pytest.mark.usefixtures("english_voice_cache")
def test_an_english_voice_absent_from_the_cache_names_the_fetch_command() -> None:
    with pytest.raises(SystemExit, match="fetch it with: speech-fetch-voice"):
        HEARING.english_voice_path()
