"""The hearing proof refuses a bad command line and a missing model before it starts."""

from __future__ import annotations

from pathlib import Path

import pytest

from speech_runtime_service.hearing_models import HearingModel
from tests.support.script_support import script_module

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
