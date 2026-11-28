"""The speech stage proof end to end over a served fake runtime and a still clock."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import speaking_figures as figures
from shipped_catalogue import CATALOGUE, ON_DEMAND_VOICES, PIPER_VOICES
from speech_service.hearing_models import HearingModel
from speech_service.service import Runtime, RuntimeDependencies
from tests.support.clock_support import FakeClock
from tests.support.fakes_support import FakeSentenceVoice
from tests.support.hear_support import FakeHearing, loaded
from tests.support.hearing_service_support import service_of
from tests.support.script_support import script_module

if TYPE_CHECKING:
    from proof_identity import ProofIdentity
    from proof_minting import Minting
    from speech_client.wire import VoiceId

PROVE = script_module("prove_speech")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def served_runtime(voices: set[VoiceId], hearing: FakeHearing | None) -> Runtime:
    return Runtime(
        {voice: FakeSentenceVoice() for voice in voices},
        memory_probe=lambda: 0,
        hearing=None if hearing is None else loaded(hearing),
        dependencies=RuntimeDependencies(catalogue=CATALOGUE),
    )


def printed_by_the_proof(
    runtime: Runtime,
    minting: Minting,
    identity: ProofIdentity,
    capsys: pytest.CaptureFixture[str],
    starts: list[PROVE.Start] | None = None,
) -> list[str]:
    """The lines the proof prints against `runtime`, served by this process.

    Each Start the proof asks its serve port for is appended to `starts`.
    """

    async def scenario() -> None:
        async with service_of(runtime, minting, identity) as service:
            health = await asyncio.to_thread(PROVE.read_health, service.base_url)
            running = PROVE.RunningSpeech(os.getpid(), health, 0.0)
            started = PROVE.Started(
                dataclasses.replace(service, running=running),
                dict(os.environ),
                REPOSITORY_ROOT,
                identity.minting_path,
            )

            def serve(
                start: PROVE.Start,
            ) -> contextlib.AbstractContextManager[PROVE.Started]:
                if starts is not None:
                    starts.append(start)
                return contextlib.nullcontext(started)

            await asyncio.to_thread(PROVE.prove, serve, FakeClock().now)

    asyncio.run(scenario())
    return capsys.readouterr().out.splitlines()


@pytest.fixture(autouse=True)
def proof_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SPEECH_DEVICE", "cpu")
    monkeypatch.delenv("SPEECH_VOICES", raising=False)
    monkeypatch.delenv("SPEECH_HEARING_MODEL", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def starting_with(lines: list[str], prefix: str) -> list[str]:
    return [line for line in lines if line.startswith(prefix)]


def assert_ends_in_proof_ok_from_this_process(lines: list[str]) -> None:
    (compute,) = starting_with(lines, "gpu_compute_pids")
    (peak,) = starting_with(lines, "peak_rss_mib")
    assert "PROOF_OK" in lines
    assert compute.endswith(f"speech_pid={os.getpid()}")
    assert re.fullmatch(r"peak_rss_mib [1-9]\d*", peak)


def test_piper_only_ends_in_proof_ok_with_hearing_beside_speaking_skipped(
    minting: Minting, identity: ProofIdentity, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = served_runtime(set(PIPER_VOICES), None)

    lines = printed_by_the_proof(runtime, minting, identity, capsys)

    assert_ends_in_proof_ok_from_this_process(lines)
    assert "hear_beside_speaking skipped: no hearing model named" in lines
    assert not starting_with(lines, "speak_wire_de_rep")


def test_a_named_gpu_voice_is_measured_on_its_own(
    monkeypatch: pytest.MonkeyPatch,
    minting: Minting,
    identity: ProofIdentity,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gpu_voice = ON_DEMAND_VOICES[0]
    monkeypatch.setenv("SPEECH_VOICES", PROVE.offer_setting({*PIPER_VOICES, gpu_voice}))
    runtime = served_runtime({*PIPER_VOICES, gpu_voice}, None)

    starts: list[PROVE.Start] = []
    lines = printed_by_the_proof(runtime, minting, identity, capsys, starts)

    assert_ends_in_proof_ok_from_this_process(lines)
    assert len(starting_with(lines, "speak_wire_de_rep")) == figures.MEASURE_REPETITIONS
    (start,) = starts
    assert set(start.offered) == {*PIPER_VOICES, gpu_voice}


def test_with_a_hearing_model_a_recording_is_heard_while_two_voices_speak(
    monkeypatch: pytest.MonkeyPatch,
    minting: Minting,
    identity: ProofIdentity,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SPEECH_HEARING_MODEL", HearingModel.SMALL.value)
    runtime = served_runtime(set(PIPER_VOICES), FakeHearing(PROVE.SENTENCE))

    lines = printed_by_the_proof(runtime, minting, identity, capsys)

    assert_ends_in_proof_ok_from_this_process(lines)
    assert starting_with(lines, "hear_beside_speaking seconds=")
    assert "hear_beside_speaking skipped: no hearing model named" not in lines
