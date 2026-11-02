"""The hearing proof: its command line, record, wait, memory peak and taken card."""

from __future__ import annotations

import contextlib
import json
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import httpx2
import pytest

from speech_runtime_client.wire import RefusalCode, refusal
from speech_runtime_service.hearing_models import HearingModel
from tests.support.clock_support import FakeClock
from tests.support.nvidia_support import FakeNvidiaSmi
from tests.support.script_support import script_module

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterator

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = FakeNvidiaSmi(tmp_path, monkeypatch)
    card.use(1000)
    service = HEARING.Service("http://127.0.0.1:1", None, None)  # type: ignore[arg-type]
    synthesis_mib = 700

    @contextlib.contextmanager
    def serve(*_: str) -> Iterator[HEARING.Service]:
        yield service

    def speak(*_: object) -> tuple[bytes, float, float]:
        card.use(1000 + synthesis_mib)
        return b"", 0.0, 0.0

    ports = HEARING.Ports(serve=serve, speak=speak, hear=lambda *_: (0.0, ""))

    peak = HEARING.measure_peak(
        "cuda", "small", HEARING.DEFAULT_SPEAKING_MODEL, b"", ports
    )

    assert peak["warmed_resident_over_baseline_mib"] == synthesis_mib


class FakeCard:
    """Replaces everything of `prove` that needs a service or a card."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, cache: Path, *, fails_in: str = ""
    ) -> None:
        """Patch the proof to start no service; `fails_in` names a failing model."""
        self.fails_in = fails_in
        clips = {
            language: HEARING.Clip(b"", b"") for language in HEARING.HearingLanguage
        }
        monkeypatch.setenv("SPEECH_HUGGINGFACE_CACHE", str(cache))
        monkeypatch.setattr(HEARING, "require_cached_models", lambda *_: None)
        monkeypatch.setattr(HEARING, "card_line", lambda _: "card")
        monkeypatch.setattr(HEARING, "hearing_service", self.service)
        monkeypatch.setattr(HEARING, "make_clips", lambda *_: clips)
        monkeypatch.setattr(HEARING, "measure_language", lambda *_: {"figures": 1})
        monkeypatch.setattr(HEARING, "measure_peak", lambda *_: {"peak_mib": 7})
        monkeypatch.setattr(HEARING, "prove_taken_card", lambda *_: {"failures": []})

    @contextlib.contextmanager
    def service(self, _device: str, model: str, _speaking: str) -> Iterator[object]:
        """A service that starts, or fails for the model named in `fails_in`."""
        if self.fails_in == model:
            message = f"{model} failed"
            raise RuntimeError(message)
        running = SimpleNamespace(ready_seconds=1.5)
        yield SimpleNamespace(running=running)


def cuda_arguments(tmp_path: Path, *changes: str) -> argparse.Namespace:
    return HEARING.parse_arguments(
        command("--record", str(tmp_path / "record.json"), "--device", "cuda", *changes)
    )


def prove(tmp_path: Path, *changes: str) -> dict[str, object]:
    record = tmp_path / "record.json"
    arguments = cuda_arguments(tmp_path, *changes)
    with contextlib.suppress(SystemExit, RuntimeError):
        HEARING.prove(arguments, lambda: None)
    return json.loads(record.read_text())


def test_a_run_of_one_model_on_cuda_names_the_other_as_not_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path)

    record = prove(tmp_path)

    assert record["not_measured"] == ["hearing model large-v3"]
    assert record["complete"] is False


def test_every_model_and_stage_measured_makes_the_record_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path)

    record = prove(tmp_path, "--hearing-model", "large-v3")

    assert record["not_measured"] == []
    assert record["complete"] is True


def test_a_taken_card_that_differs_is_never_written_as_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path)
    monkeypatch.setattr(HEARING, "prove_taken_card", lambda *_: {"failures": ["x"]})
    written: list[object] = []
    write = HEARING.write_record

    def write_noting_completeness(path: Path, record: dict[str, object]) -> None:
        written.append(record["complete"])
        write(path, record)

    monkeypatch.setattr(HEARING, "write_record", write_noting_completeness)
    arguments = cuda_arguments(tmp_path, "--hearing-model", "large-v3")

    with pytest.raises(SystemExit, match="differs from cell 30: x"):
        HEARING.prove(arguments, lambda: None)

    assert True not in written


def test_a_failing_second_model_leaves_the_figures_of_the_first_and_names_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path, fails_in="large-v3")

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
    FakeCard(monkeypatch, tmp_path)
    calls = []

    def peak(*_: object) -> dict[str, object]:
        calls.append(1)
        if len(calls) > 1:
            message = "second peak failed"
            raise RuntimeError(message)
        return {"peak_mib": 7}

    monkeypatch.setattr(HEARING, "measure_peak", peak)

    record = prove(tmp_path, "--hearing-model", "large-v3")

    assert record["gpu_memory"]["models"]["small"]["hearing_alone"] == {"peak_mib": 7}
    assert record["not_measured"] == ["stage gpu_memory", "stage taken_card"]


def test_a_stage_that_does_not_apply_on_cpu_is_not_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path)
    record_path = tmp_path / "record.json"
    arguments = HEARING.parse_arguments(command("--record", str(record_path)))

    HEARING.prove(arguments, lambda: None)

    assert json.loads(record_path.read_text())["not_measured"] == []


HELD_MIB = 512
HEALTH_READY = {"speaking": {"ready": True}, "hearing": {"ready": True}}
HEALTH_NO_HEARING = {"speaking": {"ready": True}, "hearing": {"ready": False}}
HEALTH_LOADING = {"speaking": {"ready": False}, "hearing": {"ready": False}}


def wait_loaded_with(health: dict[str, object], *, speaks: bool) -> dict[str, object]:
    clock = FakeClock()
    return HEARING.wait_loaded(
        "http://127.0.0.1:1", lambda: speaks, lambda *_: health, clock.now, clock.jump
    )


@pytest.mark.parametrize(
    ("health", "speaks"),
    [(HEALTH_READY, False), (HEALTH_NO_HEARING, True)],
    ids=["hearing ready", "speaking served after a load without hearing"],
)
def test_the_service_has_loaded_when_hearing_is_ready_or_speaking_is_served(
    health: dict[str, object], *, speaks: bool
) -> None:
    assert wait_loaded_with(health, speaks=speaks) == health


@pytest.mark.parametrize(
    ("health", "speaks"),
    [(HEALTH_LOADING, True), (HEALTH_NO_HEARING, False)],
    ids=["still loading", "speaking ready but not served"],
)
def test_a_service_that_never_finishes_loading_fails_the_wait(
    health: dict[str, object], *, speaks: bool
) -> None:
    with pytest.raises(SystemExit, match="did not finish loading"):
        wait_loaded_with(health, speaks=speaks)


def taken_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    health: dict[str, object],
    file_response: httpx2.Response,
    freed: bool,
) -> dict[str, object]:
    FakeNvidiaSmi(tmp_path, monkeypatch).use(100)

    @contextlib.contextmanager
    def hold() -> Iterator[int]:
        yield HELD_MIB

    @contextlib.contextmanager
    def serve(*_: str) -> Iterator[object]:
        yield SimpleNamespace(
            base_url="http://127.0.0.1:1",
            minting=None,
            running=SimpleNamespace(health=health),
        )

    async def upgrade(_: object) -> int:
        return HTTPStatus.SERVICE_UNAVAILABLE

    ports = HEARING.Ports(
        hold=hold,
        serve=serve,
        post=lambda *_: file_response,
        upgrade=upgrade,
        speak=lambda *_: (b"wav", 0.0, 0.0),
        freed=lambda _: freed,
    )
    clips = {HEARING.HearingLanguage.DE: HEARING.Clip(b"", b"")}
    return HEARING.prove_taken_card("cuda", "small", clips, ports)


def unavailable() -> httpx2.Response:
    return httpx2.Response(
        HTTPStatus.SERVICE_UNAVAILABLE,
        content=refusal(RefusalCode.UNAVAILABLE).model_dump_json().encode(),
    )


def test_a_taken_card_that_behaves_as_cell_30_says_gives_no_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = taken_card(
        tmp_path,
        monkeypatch,
        health=HEALTH_NO_HEARING,
        file_response=unavailable(),
        freed=True,
    )

    assert result["failures"] == []
    assert result["held_mib"] == HELD_MIB


def test_a_taken_card_that_is_not_freed_afterwards_is_a_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = taken_card(
        tmp_path,
        monkeypatch,
        health=HEALTH_NO_HEARING,
        file_response=unavailable(),
        freed=False,
    )

    assert result["failures"] == ["the card was not freed"]


def test_hearing_ready_on_a_taken_card_is_a_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = taken_card(
        tmp_path,
        monkeypatch,
        health=HEALTH_READY,
        file_response=unavailable(),
        freed=True,
    )

    assert result["failures"] == ["hearing is ready on a taken card"]


def test_a_file_route_that_answers_ok_on_a_taken_card_is_a_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = taken_card(
        tmp_path,
        monkeypatch,
        health=HEALTH_NO_HEARING,
        file_response=httpx2.Response(HTTPStatus.OK),
        freed=True,
    )

    assert result["failures"] == ["the file route did not answer 503 unavailable"]


def test_uncached_voices_are_named_not_measured_against(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = HEARING.GPU_SPEAKING_MODELS[0]
    monkeypatch.setattr(HEARING, "measure_peak", lambda *_: {})
    memory: dict[str, object] = {}
    clips = {HEARING.HearingLanguage.DE: HEARING.Clip(b"", b"")}

    HEARING.measure_memory(
        "cuda", ["small"], clips, memory, {HEARING.voice_for_model(cached)}
    )

    assert memory["voices_not_in_cache"] == [
        HEARING.voice_for_model(model).value
        for model in HEARING.GPU_SPEAKING_MODELS[1:]
    ]
    assert list(memory["models"]["small"]["with_voice"]) == [
        HEARING.voice_for_model(cached).value
    ]


def test_a_cpu_run_that_fails_names_no_stage_that_never_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path, fails_in="small")
    record_path = tmp_path / "record.json"
    arguments = HEARING.parse_arguments(command("--record", str(record_path)))

    with pytest.raises(RuntimeError, match="small failed"):
        HEARING.prove(arguments, lambda: None)

    assert json.loads(record_path.read_text())["not_measured"] == [
        "hearing model small"
    ]


def test_a_proof_killed_after_its_first_model_has_left_that_model_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeCard(monkeypatch, tmp_path)
    record_path = tmp_path / "record.json"
    on_disk: list[dict[str, object]] = []
    started = fake.service

    @contextlib.contextmanager
    def second_model_starts(device: str, model: str, speaking: str) -> Iterator[object]:
        if model == "large-v3":
            on_disk.append(json.loads(record_path.read_text()))
        with started(device, model, speaking) as service:
            yield service

    monkeypatch.setattr(HEARING, "hearing_service", second_model_starts)

    prove(tmp_path, "--hearing-model", "large-v3")

    assert on_disk[0]["models"]["small"]["status"] == "measured"
    assert on_disk[0]["not_measured"] == [
        "hearing model large-v3",
        "stage gpu_memory",
        "stage taken_card",
    ]


def test_a_language_that_fails_keeps_the_other_and_leaves_its_model_unfinished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path)

    def hear(_service: object, language: object, *_: object) -> dict[str, object]:
        if language.code == HEARING.HearingLanguage.EN:
            message = "english failed"
            raise RuntimeError(message)
        return {"figures": 1}

    monkeypatch.setattr(HEARING, "measure_language", hear)

    record = prove(tmp_path)

    small = record["models"]["small"]
    assert small["languages"] == {"de": {"figures": 1}}
    assert small["status"] == "unfinished"
    assert "hearing model small" in record["not_measured"]


def test_the_record_is_on_disk_after_each_language_and_each_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCard(monkeypatch, tmp_path)
    record_path = tmp_path / "record.json"
    seen: list[dict[str, object]] = []

    def read_then(value: dict[str, object]) -> Callable[..., dict[str, object]]:
        def measure(*_: object) -> dict[str, object]:
            seen.append(json.loads(record_path.read_text()))
            return value

        return measure

    monkeypatch.setattr(HEARING, "measure_language", read_then({"figures": 1}))
    monkeypatch.setattr(HEARING, "measure_peak", read_then({"peak_mib": 7}))
    monkeypatch.setattr(HEARING, "prove_taken_card", read_then({"failures": []}))

    prove(tmp_path)

    first_language, second_language, first_peak = seen[:3]
    assert "stage gpu_memory" not in seen[-1]["not_measured"]
    assert first_language["models"] == {}
    assert list(second_language["models"]["small"]["languages"]) == ["de"]
    assert first_peak["models"]["small"]["status"] == "measured"
