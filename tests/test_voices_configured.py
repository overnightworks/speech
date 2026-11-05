"""The start setting is the one source of the offer: Piper always, GPU voices named."""

from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from speech_runtime_provider_contract import QWEN_MODEL_ID, QWEN_SAMPLE_RATE
from starlette.status import HTTP_200_OK, HTTP_404_NOT_FOUND
from starlette.websockets import WebSocketDisconnect

from speech_runtime_client.wire import Health, RefusalCode
from speech_runtime_service.config import Settings
from speech_runtime_service.control import create_control_app
from speech_runtime_service.pcm import pcm_from_wav
from speech_runtime_service.voices import DEFAULT_SPEAKING_MODEL, VoiceId
from tests.support.fakes_support import FakeSpeaking
from tests.support.gpu_voice_support import (
    PIPER_PCM,
    PIPER_SAMPLE_RATE,
    StageSetup,
    gpu_voice_stage,
)
from tests.support.identity_support import valid_callers
from tests.support.speak_support import (
    app_over,
    assert_refusal,
    assert_unavailable,
    fake_runtime,
    post_speech,
    speak_body,
)

if TYPE_CHECKING:
    from pathlib import Path

QWEN = VoiceId.QWEN.value
PIPER = VoiceId.PIPER.value
_PIPER_ONLY = StageSetup(gpu_voices=frozenset())


def test_by_default_only_piper_is_offered_and_it_speaks_without_a_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with gpu_voice_stage(tmp_path, monkeypatch, _PIPER_ONLY) as stage:
        client = TestClient(stage.app())
        spoken = post_speech(client, speak_body(PIPER))
        health = Health.model_validate(client.get("/health").json())

    assert spoken.status_code == HTTP_200_OK
    assert pcm_from_wav(spoken.content) == (PIPER_SAMPLE_RATE, PIPER_PCM)
    assert health.speaking.ready is True
    assert list(health.speaking.voices) == [PIPER]
    assert health.speaking.voices[PIPER].model == DEFAULT_SPEAKING_MODEL
    assert stage.starts() == []


def test_a_named_gpu_voice_is_offered_but_not_started_until_a_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with gpu_voice_stage(tmp_path, monkeypatch) as stage:
        health = Health.model_validate(TestClient(stage.app()).get("/health").json())

    assert health.speaking.ready is True
    assert health.hearing.ready is True
    assert set(health.speaking.voices) == {PIPER, QWEN}
    qwen = health.speaking.voices[QWEN]
    assert (qwen.model, qwen.ready, qwen.sample_rate) == (
        QWEN_MODEL_ID,
        False,
        QWEN_SAMPLE_RATE,
    )
    assert stage.starts() == []


def test_a_listed_voice_that_is_not_offered_is_unavailable_and_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with gpu_voice_stage(tmp_path, monkeypatch, _PIPER_ONLY) as stage:
        refused = post_speech(TestClient(stage.app()), speak_body(QWEN))

    assert_unavailable(refused)
    assert stage.starts() == []


def test_an_offered_voice_the_app_does_not_list_is_forbidden_and_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with gpu_voice_stage(tmp_path, monkeypatch) as stage:
        app = create_control_app(
            stage.settings, stage.runtime, valid_callers([VoiceId.PIPER])
        )
        refused = post_speech(TestClient(app), speak_body(QWEN))

    assert_refusal(refused, HTTPStatus.FORBIDDEN, RefusalCode.FORBIDDEN, "voice")
    assert stage.starts() == []


def test_a_selection_file_from_an_older_build_is_ignored_and_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    older_selection = state / "selected-voice"
    older_selection.write_text(QWEN, encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("SPEECH_STATE_DIRECTORY", str(state))
    monkeypatch.setenv("HOME", str(home))
    with gpu_voice_stage(
        tmp_path, monkeypatch, StageSetup(gpu_voices=frozenset(), started=False)
    ) as stage:
        before = sorted(tmp_path.rglob("*"))
        stage.runtime.load()
        client = TestClient(app_over(stage.runtime))
        spoken = post_speech(client, speak_body(PIPER))
        refused = post_speech(client, speak_body(QWEN))
        after = sorted(tmp_path.rglob("*"))

    assert pcm_from_wav(spoken.content) == (PIPER_SAMPLE_RATE, PIPER_PCM)
    assert refused.status_code != HTTP_200_OK
    assert older_selection.read_text(encoding="utf-8") == QWEN
    assert after == before


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/voices"),
        ("POST", "/voices/piper/load"),
        ("POST", "/voices/qwen3-tts-0.6b/download"),
        ("POST", "/voices/piper/sample/de"),
    ],
)
def test_voice_administration_is_not_mounted(method: str, path: str) -> None:
    speaking = FakeSpeaking()
    client = TestClient(
        create_control_app(Settings(), fake_runtime(speaking), valid_callers())
    )

    response = client.request(method, path)

    assert response.status_code == HTTP_404_NOT_FOUND
    assert speaking.heard_text == ""


def test_hearing_is_not_mounted() -> None:
    client = TestClient(create_control_app(Settings(), fake_runtime(), valid_callers()))

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/hear"):
        pass
