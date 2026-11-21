"""Hearing starts beside speaking; a missing resource isolates it."""

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette.status import HTTP_200_OK, HTTP_503_SERVICE_UNAVAILABLE

from shipped_catalogue import CATALOGUE
from speech_client.wire import HearingHealth, RefusalCode, VoiceId
from speech_service import hearing_models
from speech_service.clock import LoopClock
from speech_service.config import Settings
from speech_service.hearing import (
    Hearing,
    ResourceUnavailableError,
    hearing_from_settings,
)
from speech_service.hearing_models import (
    HearingModel,
    HearingModelMissingError,
)
from speech_service.service import Runtime, RuntimeDependencies
from speech_service.sound_events import SoundModelMissingError, SoundTaggers
from tests.support.fakes_support import FakeSpeaking
from tests.support.hear_support import (
    FakeHearing,
    hear_app,
    multipart_body,
    post_recording,
    recording,
    write_snapshot,
)
from tests.support.sound_support import (
    LABELS,
    FakeSoundModel,
    loaded_taggers,
    opener_of,
    write_model_directory,
)
from tests.support.speak_support import assert_refusal, post_speech, speak_body

if TYPE_CHECKING:
    from pathlib import Path


def _starting(hearing: FakeHearing | None) -> Runtime:
    return Runtime(
        {VoiceId.THORSTEN: None},
        memory_probe=lambda: 0,
        dependencies=RuntimeDependencies(
            catalogue=CATALOGUE,
            engine_factory=lambda _voice: FakeSpeaking(),
            missing_artifacts=lambda _voice: (),
        ),
        hearing=Hearing(hearing, LoopClock()),
    )


def test_a_load_that_lacks_the_resource_leaves_speaking_up_hearing_unavailable() -> (
    None
):
    """WIRE-40, WIRE-129: a load that lacks the resource leaves speaking up."""
    engine = FakeHearing(load_error=ResourceUnavailableError("the card is full"))
    runtime = _starting(engine)

    runtime.load()
    client = TestClient(hear_app(runtime))

    health = client.get("/health").json()
    transcription = post_recording(client, multipart_body(file=recording()))
    speech = post_speech(client, speak_body(VoiceId.THORSTEN))
    assert health["hearing"]["ready"] is False
    assert health["speaking"]["ready"] is True
    assert_refusal(
        transcription, HTTP_503_SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE, None
    )
    assert speech.status_code == HTTP_200_OK
    assert engine.closed == 1


def test_a_load_that_fails_otherwise_stops_the_start_naming_the_setting() -> None:
    engine = FakeHearing(load_error=RuntimeError("weights are corrupt"))
    runtime = _starting(engine)

    with pytest.raises(RuntimeError, match="SPEECH_HEARING_MODEL"):
        runtime.load()

    assert engine.closed == 1


def test_a_loaded_engine_is_ready_in_health() -> None:
    """WIRE-128, WIRE-129: a loaded engine is ready in health."""
    engine = FakeHearing()
    runtime = _starting(engine)

    runtime.load()

    assert runtime.health().hearing == HearingHealth(
        model=engine.model_name, ready=True
    )


def test_closing_the_runtime_frees_the_hearing_engine() -> None:
    engine = FakeHearing()
    runtime = _starting(engine)
    runtime.load()

    runtime.close()

    assert engine.closed == 1


def test_without_a_hearing_model_health_names_hearing_unavailable() -> None:
    """WIRE-128: without a hearing model health names hearing unavailable."""
    runtime = _starting(None)

    runtime.load()

    assert runtime.health().hearing == HearingHealth(model="unavailable", ready=False)


def test_a_configured_model_that_was_not_fetched_stops_the_start(
    tmp_path: Path,
) -> None:
    settings = Settings(
        hearing_model=HearingModel.SMALL,
        huggingface_cache=tmp_path,
        sound_events=False,
    )

    with pytest.raises(HearingModelMissingError, match="SPEECH_HEARING_MODEL"):
        Runtime.from_settings(settings)


def test_a_configured_model_that_was_fetched_is_named_in_health_before_it_loads(
    tmp_path: Path,
) -> None:
    """WIRE-128, WIRE-129: a fetched model is named in health before it loads."""
    hearing_models.fetch_model(HearingModel.SMALL, tmp_path, write_snapshot)
    settings = Settings(
        hearing_model=HearingModel.SMALL,
        huggingface_cache=tmp_path,
        sound_events=False,
    )

    runtime = Runtime.from_settings(settings)

    assert runtime.hearing.health() == {
        "model": HearingModel.SMALL.value,
        "ready": False,
    }


def _fetched_settings(tmp_path: Path, **settings: object) -> Settings:
    hearing_models.fetch_model(HearingModel.SMALL, tmp_path / "hearing", write_snapshot)
    return Settings(
        hearing_model=HearingModel.SMALL,
        huggingface_cache=tmp_path / "hearing",
        **settings,
    )


def test_a_start_with_sounds_on_and_no_sound_model_ends_naming_the_way_out(
    tmp_path: Path,
) -> None:
    """Scenario row 5."""
    settings = _fetched_settings(tmp_path, sound_model_directory=tmp_path / "none")

    with pytest.raises(SoundModelMissingError) as refused:
        Runtime.from_settings(settings)

    assert "SPEECH_SOUND_MODEL_DIRECTORY" in str(refused.value)
    assert "SPEECH_SOUND_EVENTS=false" in str(refused.value)


def test_a_start_with_sounds_off_needs_no_sound_model(tmp_path: Path) -> None:
    """Scenario row 8."""
    settings = _fetched_settings(
        tmp_path, sound_events=False, sound_model_directory=tmp_path / "none"
    )

    Runtime.from_settings(settings)


def test_a_start_without_hearing_never_reads_the_sound_settings(
    tmp_path: Path,
) -> None:
    """Scenario row 9."""
    settings = Settings(sound_model_directory=tmp_path / "none")

    hearing_from_settings(settings, LoopClock())


@pytest.mark.parametrize(
    ("labels", "input_shape"),
    [
        ([label for label in LABELS if label != "Cough"], (1, 32000)),
        (LABELS, (1, 16000)),
    ],
    ids=["a-mapped-label-missing", "wrong-input-shape"],
)
def test_a_sound_model_that_fails_its_load_stops_the_start_and_frees_the_engine(
    tmp_path: Path, labels: list[str], input_shape: tuple[int, int]
) -> None:
    """Scenario rows 6, 7."""
    engine = FakeHearing()
    model = FakeSoundModel(input_shape=input_shape, answer_labels=labels)
    sounds = SoundTaggers(write_model_directory(tmp_path, labels), opener_of(model))
    hearing = Hearing(engine, LoopClock(), sounds=sounds)

    with pytest.raises(RuntimeError, match="SPEECH_SOUND_MODEL_DIRECTORY"):
        hearing.load()

    assert engine.closed == 1
    assert hearing.capture() is None


def test_the_sound_model_loads_only_after_the_engine_is_ready(tmp_path: Path) -> None:
    opened: list[object] = []
    engine = FakeHearing(load_error=ResourceUnavailableError("the card is full"))

    def opener(path: object) -> FakeSoundModel:
        opened.append(path)
        return FakeSoundModel()

    directory = write_model_directory(tmp_path)
    hearing = Hearing(engine, LoopClock(), sounds=SoundTaggers(directory, opener))

    hearing.load()

    assert opened == []
    assert hearing.open_sound_stream() is None


def test_a_loaded_hearing_opens_a_sound_stream_and_closing_it_stops_the_scoring(
    tmp_path: Path,
) -> None:
    taggers, _ = loaded_taggers(tmp_path)
    hearing = Hearing(FakeHearing(), LoopClock(), sounds=taggers)
    hearing.load()
    stream = hearing.open_sound_stream()
    assert stream is not None
    stream.add(bytes(64_000))
    window = stream.next_window()
    assert window is not None

    hearing.close()

    with pytest.raises(RuntimeError):
        hearing.submit_sound(window)
