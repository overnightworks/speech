"""The hearing proof refuses a bad command line and a missing model before it starts."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from speech_runtime_service.hearing_models import HearingModel
from tests.support.script_support import script_module

if TYPE_CHECKING:
    from collections.abc import Iterator

HEARING = script_module("prove_hearing")
VOICES = Path.home() / "voices"
RUNS = "7"


def command(*changes: str) -> list[str]:
    return [
        "--device",
        "cpu",
        "--hearing-model",
        "small",
        "--record",
        "record.json",
        "--english-voice-directory",
        str(VOICES),
        *changes,
    ]


def test_the_device_the_models_and_the_paths_come_from_the_command_line() -> None:
    arguments = HEARING.parse_arguments(
        command("--hearing-model", "large-v3", "--runs", RUNS)
    )

    assert arguments.device == "cpu"
    assert arguments.hearing_models == [HearingModel.SMALL, HearingModel.LARGE_V3]
    assert arguments.record == Path("record.json")
    assert arguments.english_voice_directory == VOICES
    assert arguments.runs == int(RUNS)


def test_the_runs_default_to_the_fewest_a_median_needs() -> None:
    assert HEARING.parse_arguments(command()).runs == HEARING.figures.MIN_RUNS


def test_fewer_runs_than_a_median_needs_are_refused() -> None:
    arguments = command("--runs", "4")

    with pytest.raises(SystemExit):
        HEARING.parse_arguments(arguments)


@pytest.mark.parametrize("missing", ["--device", "--hearing-model", "--record"])
def test_a_command_without_a_required_option_is_refused(missing: str) -> None:
    arguments = command()
    position = arguments.index(missing)
    del arguments[position : position + 2]

    with pytest.raises(SystemExit):
        HEARING.parse_arguments(arguments)


def test_a_model_missing_from_the_cache_names_the_command_that_fetches_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEECH_HUGGINGFACE_CACHE", str(tmp_path))

    with pytest.raises(SystemExit, match="speech-runtime-fetch-hearing-model small"):
        HEARING.require_cached_models([HearingModel.SMALL])


def test_the_resident_memory_with_a_voice_includes_what_its_synthesis_took(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = {"used_mib": 1000}
    service = HEARING.Service("http://127.0.0.1:1", None, None)  # type: ignore[arg-type]
    synthesis_mib = 700

    @contextlib.contextmanager
    def fake_service(*_: str) -> Iterator[HEARING.Service]:
        yield service

    def fake_speak(*_: object) -> tuple[bytes, float, float]:
        card["used_mib"] += synthesis_mib
        return b"", 0.0, 0.0

    monkeypatch.setattr(HEARING, "hearing_service", fake_service)
    monkeypatch.setattr(HEARING, "speak_measured", fake_speak)
    monkeypatch.setattr(HEARING, "hear_file", lambda *_: (0.0, ""))
    monkeypatch.setattr(HEARING, "used_mib", lambda: card["used_mib"])

    peak = HEARING.measure_peak("cuda", "small", HEARING.DEFAULT_SPEAKING_MODEL, b"")

    assert peak["warmed_resident_over_baseline_mib"] == synthesis_mib


class FakeCard:
    """Replaces everything of `_prove` that needs a service or a card."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, fails_in: str = "") -> None:
        """Patch the proof to start no service; `fails_in` names a failing model."""
        self.fails_in = fails_in
        clips = {
            language: HEARING.Clip(b"", b"") for language in HEARING.HearingLanguage
        }
        for name in ("require_cached_models", "wait_for_load"):
            monkeypatch.setattr(HEARING, name, lambda *_: None)
        monkeypatch.setattr(HEARING, "card_line", lambda _: "card")
        monkeypatch.setattr(HEARING, "hearing_service", self.service)
        monkeypatch.setattr(HEARING, "make_clips", lambda *_: clips)
        monkeypatch.setattr(HEARING, "measure_language", lambda *_: {"figures": 1})
        monkeypatch.setattr(HEARING, "measure_peak", lambda *_: {"peak_mib": 7})
        monkeypatch.setattr(HEARING, "prove_taken_card", lambda *_: {"failures": []})
        monkeypatch.setattr(
            HEARING,
            "installed_voice_ids",
            lambda _: {HEARING.voice_for_model(m) for m in HEARING.GPU_SPEAKING_MODELS},
        )

    @contextlib.contextmanager
    def service(self, _device: str, model: str, _speaking: str) -> Iterator[object]:
        """A service that starts, or fails for the model named in `fails_in`."""
        if self.fails_in == model:
            message = f"{model} failed"
            raise RuntimeError(message)
        running = SimpleNamespace(ready_seconds=1.5)
        yield SimpleNamespace(running=running)


def prove(tmp_path: Path, *changes: str) -> dict[str, object]:
    record = tmp_path / "record.json"
    arguments = HEARING.parse_arguments(
        command("--record", str(record), "--device", "cuda", *changes)
    )
    with contextlib.suppress(SystemExit, RuntimeError):
        HEARING._prove(arguments)  # noqa: SLF001 - the proof's one entry below main
    return json.loads(record.read_text())


def test_a_run_of_one_model_on_cuda_names_the_other_as_not_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch)

    record = prove(tmp_path)

    assert record["not_measured"] == ["hearing model large-v3"]
    assert record["complete"] is False


def test_every_model_and_stage_measured_makes_the_record_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch)

    record = prove(tmp_path, "--hearing-model", "large-v3")

    assert record["not_measured"] == []
    assert record["complete"] is True


def test_a_failing_second_model_leaves_the_figures_of_the_first_and_names_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, fails_in="large-v3")

    record = prove(tmp_path, "--hearing-model", "large-v3")

    assert set(record["models"]) == {"small"}
    assert record["not_measured"] == [
        "hearing model large-v3",
        "stage gpu_memory",
        "stage taken_card",
    ]
    assert record["complete"] is False


def test_a_stage_that_fails_keeps_what_it_measured_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch)
    calls = []

    def peak(*_: object) -> dict[str, object]:
        calls.append(1)
        if len(calls) > 1:
            message = "second peak failed"
            raise RuntimeError(message)
        return {"peak_mib": 7}

    monkeypatch.setattr(HEARING, "measure_peak", peak)

    record = prove(tmp_path)

    assert record["gpu_memory"]["models"]["small"]["hearing_alone"] == {"peak_mib": 7}
    assert "stage gpu_memory" in record["not_measured"]
    assert "stage taken_card" in record["not_measured"]


def test_a_stage_that_does_not_apply_on_cpu_is_not_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch)
    record_path = tmp_path / "record.json"
    arguments = HEARING.parse_arguments(command("--record", str(record_path)))

    HEARING._prove(arguments)  # noqa: SLF001 - the proof's one entry below main

    assert json.loads(record_path.read_text())["not_measured"] == []
