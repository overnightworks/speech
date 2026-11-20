"""The voice lines command judges by the slowest warm run and prints one JSON."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from hearing_figures import wav_of
from shipped_catalogue import ON_DEMAND_VOICES, PIPER_VOICES
from tests.support.script_support import script_module

if TYPE_CHECKING:
    from collections.abc import Iterator

    from speech_client.wire import VoiceId

LINES = script_module("voice_lines")
RATE = 24000
LOUD = 8000
VOICE = PIPER_VOICES[0]


def wav(seconds: float, *, silent_seconds: float = 0.0) -> bytes:
    count = round(seconds * RATE)
    samples = np.full(count, LOUD, dtype="<i2")
    samples[: round(silent_seconds * RATE)] = 0
    return wav_of(samples.tobytes())


@dataclass
class Speaker:
    """A speak whose first-byte times are given, one per request."""

    firsts: list[float]
    total: float = 1.0
    seconds: float = 2.0
    requests: int = 0

    def __call__(self, _text: str, _voice: VoiceId) -> tuple[bytes, float, float]:  # noqa: D102
        self.requests += 1
        return wav(self.seconds, silent_seconds=0.5), self.firsts.pop(0), self.total


def arguments(**changes: object) -> SimpleNamespace:
    given = {
        "voice": VOICE,
        "hearing_model": "small",
        "device": "cpu",
        "runs": 3,
        "clip_seconds": 5.0,
        "first_audio_max": None,
        "rtf_max": None,
        "final_max": None,
        "two_halves": False,
    }
    return SimpleNamespace(**{**given, **changes})


def finals(*seconds: float):  # noqa: ANN201 - a test fake
    values = iter(seconds)
    return lambda: next(values)


def test_a_line_is_judged_by_the_slowest_warm_run_not_the_median() -> None:
    result = LINES.measure(
        arguments(first_audio_max=0.1),
        Speaker([9.0, 0.05, 0.06, 0.3]),
        finals(1, 1, 1, 1),
    )

    figure = result["halves"]["single"]["first_audio_seconds"]
    assert figure["median"] == pytest.approx(0.06)
    assert figure["max"] == pytest.approx(0.3)
    assert result["lines"]["first_audio_seconds"]["met"] is False
    assert result["met"] is False


def test_the_warm_up_is_not_a_run() -> None:
    result = LINES.measure(
        arguments(), Speaker([9.0, 0.1, 0.1, 0.1]), finals(9, 1, 1, 1)
    )

    figure = result["halves"]["single"]["first_audio_seconds"]
    assert figure["runs"] == [0.1, 0.1, 0.1]
    assert result["halves"]["single"]["final_seconds"]["runs"] == [1, 1, 1]


def test_the_real_time_factor_is_total_time_over_audio_time() -> None:
    result = LINES.measure(
        arguments(rtf_max=0.6),
        Speaker([0.1] * 4, total=1.0, seconds=2.0),
        finals(1, 1, 1, 1),
    )

    single = result["halves"]["single"]
    assert single["real_time_factor"]["max"] == pytest.approx(0.5)
    assert single["total_seconds"]["max"] == pytest.approx(1.0)
    assert single["audio_seconds"]["max"] == pytest.approx(2.0)
    assert result["lines"]["real_time_factor"] == {"limit": 0.6, "met": True}


def test_the_first_audible_sample_is_reported_after_the_first_byte() -> None:
    result = LINES.measure(arguments(), Speaker([0.1] * 4), finals(1, 1, 1, 1))

    audible = result["halves"]["single"]["first_audible_seconds"]["median"]
    assert audible == pytest.approx(0.1 + 0.5)


def test_a_figure_without_a_threshold_is_reported_and_not_judged() -> None:
    result = LINES.measure(
        arguments(final_max=2.0), Speaker([0.1] * 4), finals(5, 1, 1, 1)
    )

    assert set(result["lines"]) == {"final_seconds"}
    assert "first_audio_seconds" in result["halves"]["single"]
    assert result["met"] is True


def test_a_speak_without_an_audible_sample_fails_the_run() -> None:
    def silent(_text: str, _voice: VoiceId) -> tuple[bytes, float, float]:
        return wav_of(bytes(4800)), 0.1, 1.0

    with pytest.raises(SystemExit) as raised:
        LINES.measure(arguments(), silent, finals(1, 1, 1, 1))

    assert raised.value.code != 0


def test_the_loaded_half_starts_only_after_the_signal_and_is_the_one_judged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    firsts = iter([0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5])

    def speak(_text: str, _voice: VoiceId) -> tuple[bytes, float, float]:
        events.append("speak")
        return wav(2.0), next(firsts), 1.0

    result = LINES.measure(
        arguments(two_halves=True, first_audio_max=0.3),
        speak,
        lambda: 1.0,
        wait=lambda: events.append("wait"),
    )

    warm_up_and_idle_runs = 1 + 3
    assert events.index("wait") == warm_up_and_idle_runs
    assert LINES.HALF_SIGNAL in capsys.readouterr().err
    assert result["halves"]["idle"]["first_audio_seconds"]["max"] == pytest.approx(0.1)
    assert result["halves"]["loaded"]["first_audio_seconds"]["max"] == pytest.approx(
        0.5
    )
    assert result["lines"]["first_audio_seconds"]["met"] is False


@pytest.fixture
def pipe() -> Iterator[tuple[int, int]]:
    read, write = os.pipe()
    yield read, write
    for descriptor in (read, write):
        with contextlib.suppress(OSError):
            os.close(descriptor)


def test_the_wait_for_the_caller_ends_on_its_line(pipe: tuple[int, int]) -> None:
    read, write = pipe
    os.write(write, b"go\n")

    with os.fdopen(read, closefd=False) as stream:
        LINES.wait_for_caller(stream, 5.0)


def test_a_silent_caller_ends_the_wait_with_an_exit(pipe: tuple[int, int]) -> None:
    read, _write = pipe

    with os.fdopen(read, closefd=False) as stream, pytest.raises(SystemExit):
        LINES.wait_for_caller(stream, 0.05)


def test_a_gone_caller_ends_the_wait_with_an_exit(pipe: tuple[int, int]) -> None:
    read, write = pipe
    os.close(write)

    with os.fdopen(read, closefd=False) as stream, pytest.raises(SystemExit):
        LINES.wait_for_caller(stream, 5.0)


def test_only_a_gpu_voice_is_told_to_load_at_start() -> None:
    assert LINES.start_settings(ON_DEMAND_VOICES[0])
    assert LINES.start_settings(PIPER_VOICES[0]) == {}


@dataclass
class FakeService:
    """The ports `prove` takes; the service prints noise on stdout."""

    speaker: Speaker

    @contextlib.contextmanager
    def service(self, *_: object) -> Iterator[SimpleNamespace]:  # noqa: D102
        sys.stdout.write("noise from the service\n")
        yield SimpleNamespace(base_url="", minting=None)

    def speak(self, _url: str, text: str, voice: VoiceId, _minting: object):  # noqa: ANN201, D102
        return self.speaker(text, voice)

    def hear(self, *_: object) -> tuple[float, str]:  # noqa: D102
        return 1.0, "text"

    def ports(self) -> dict[str, object]:  # noqa: D102
        return {
            "service": self.service,
            "speak": self.speak,
            "hear": self.hear,
            "clip": lambda *_: b"",
        }


@pytest.fixture(autouse=True)
def no_load_at_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LINES.LOAD_AT_START_SETTING, raising=False)


def test_the_command_refuses_without_a_device(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        LINES.parse_arguments(["--voice", VOICE, "--hearing-model", "small"])

    assert raised.value.code != 0
    assert "--device" in capsys.readouterr().err


@pytest.mark.parametrize(("limit", "exit_status"), [(None, 0), (5.0, 0), (0.01, 3)])
def test_the_command_prints_only_json_and_exits_by_the_lines(
    capsys: pytest.CaptureFixture[str],
    limit: float | None,
    exit_status: int,
) -> None:
    fake = FakeService(Speaker([0.1] * 4))
    parsed = arguments(first_audio_max=limit)
    ports = fake.ports()

    with pytest.raises(SystemExit) as raised:
        LINES.prove(parsed, **ports)

    assert raised.value.code == exit_status
    assert json.loads(capsys.readouterr().out)["voice"] == VOICE


def test_a_gone_caller_ends_the_command_before_the_loaded_half(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pipe: tuple[int, int],
) -> None:
    read, write = pipe
    os.close(write)
    fake = FakeService(Speaker([0.1] * 4))
    parsed = arguments(two_halves=True, runs=1)
    ports = fake.ports()

    with os.fdopen(read, closefd=False) as stdin, monkeypatch.context() as patched:
        patched.setattr(sys, "stdin", stdin)
        with pytest.raises(SystemExit) as raised:
            LINES.prove(parsed, **ports)

    assert raised.value.code != 0
    warm_up_and_idle_run = 1 + 1
    assert fake.speaker.requests == warm_up_and_idle_run
    assert capsys.readouterr().out == ""
