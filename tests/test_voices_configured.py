"""The start setting is the one source of the offer."""

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from speech_provider_contract import QWEN_SAMPLE_RATE
from starlette.status import HTTP_200_OK, HTTP_404_NOT_FOUND
from starlette.websockets import WebSocketDisconnect

from speech_client.wire import VOICES, Health, Language, VoiceId
from speech_service.config import Settings
from speech_service.control import create_control_app
from speech_service.pcm import pcm_from_wav
from speech_service.voices import PIPER_VOICES
from tests.support.fakes_support import FakeSpeaking
from tests.support.gpu_voice_support import (
    PIPER_PCM,
    PIPER_SAMPLE_RATE,
    StageSetup,
    gpu_voice_stage,
)
from tests.support.identity_support import (
    MINTING,
    OTHER_MINTING,
    callers_of_two_apps,
    valid_callers,
    valid_headers,
)
from tests.support.speak_support import (
    app_over,
    assert_unavailable,
    fake_runtime,
    post_speech,
    speak_body,
)
from tests.support.worker_support import spoken_language_codes

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_OFFER = StageSetup(voices=frozenset(PIPER_VOICES))
_ONE_PIPER_VOICE = StageSetup(voices=frozenset({VoiceId.THORSTEN}))


def test_by_default_every_piper_voice_is_offered_and_speaks_without_a_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WIRE-07, WIRE-107, WIRE-125: by default every piper voice is offered."""
    with gpu_voice_stage(tmp_path, monkeypatch, _DEFAULT_OFFER) as stage:
        client = TestClient(stage.app())
        spoken = [post_speech(client, speak_body(voice)) for voice in PIPER_VOICES]
        health = Health.model_validate(client.get("/health").json())

    assert [pcm_from_wav(response.content) for response in spoken] == [
        (PIPER_SAMPLE_RATE, PIPER_PCM)
    ] * len(PIPER_VOICES)
    assert health.speaking.ready is True
    assert list(health.speaking.voices) == [voice.value for voice in PIPER_VOICES]
    assert {
        (reported.model, reported.name, reported.languages, reported.ready)
        for reported in health.speaking.voices.values()
    } == {
        (VOICES[voice].model, VOICES[voice].name, VOICES[voice].languages, True)
        for voice in PIPER_VOICES
    }
    assert stage.starts() == []


def test_a_named_gpu_voice_is_offered_but_not_started_until_a_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WIRE-07, WIRE-107, WIRE-126: a named GPU voice is offered, loaded on request."""
    with gpu_voice_stage(tmp_path, monkeypatch) as stage:
        health = Health.model_validate(TestClient(stage.app()).get("/health").json())

    assert health.speaking.ready is True
    assert health.hearing.ready is True
    assert set(health.speaking.voices) == {VoiceId.THORSTEN, VoiceId.RYAN}
    ryan = health.speaking.voices[VoiceId.RYAN]
    assert (ryan.model, ryan.name, ryan.languages, ryan.ready, ryan.sample_rate) == (
        VOICES[VoiceId.RYAN].model,
        VOICES[VoiceId.RYAN].name,
        VOICES[VoiceId.RYAN].languages,
        False,
        QWEN_SAMPLE_RATE,
    )
    assert stage.starts() == []


def test_speaking_is_ready_when_only_a_gpu_voice_is_offered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WIRE-124: speaking is ready when only a gpu voice is offered."""
    offer = StageSetup(voices=frozenset({VoiceId.RYAN}))
    with gpu_voice_stage(tmp_path, monkeypatch, offer) as stage:
        health = Health.model_validate(TestClient(stage.app()).get("/health").json())

    assert (health.speaking.ready, health.hearing.ready) == (True, True)
    assert list(health.speaking.voices) == [VoiceId.RYAN.value]


def test_every_registered_app_speaks_an_offered_gpu_voice_through_one_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WIRE-07, WIRE-09, WIRE-114: each registered app speaks an offered GPU voice."""
    with gpu_voice_stage(tmp_path, monkeypatch, StageSetup("normal")) as stage:
        client = TestClient(
            create_control_app(stage.settings, stage.runtime, callers_of_two_apps())
        )
        spoken = [
            post_speech(
                client, speak_body(VoiceId.RYAN), valid_headers(minting=minting)
            )
            for minting in (MINTING, OTHER_MINTING)
        ]

    assert [response.status_code for response in spoken] == [HTTP_200_OK] * 2
    assert [pcm_from_wav(response.content)[0] for response in spoken] == [
        QWEN_SAMPLE_RATE
    ] * 2
    assert len(stage.starts()) == 1


@pytest.mark.parametrize("language", VOICES[VoiceId.RYAN].languages)
def test_ryan_is_spoken_by_the_worker_in_the_language_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, language: Language
) -> None:
    """WIRE-11: ryan is spoken with the provider's language byte of the language."""
    provider_language_byte = {Language.DE: 1, Language.EN: 2}[language]
    with gpu_voice_stage(tmp_path, monkeypatch, StageSetup("normal")) as stage:
        spoken = post_speech(
            TestClient(stage.app()), speak_body(VoiceId.RYAN, language=language)
        )

    assert spoken.status_code == HTTP_200_OK
    assert spoken_language_codes(stage.cache) == [provider_language_byte]


@pytest.mark.parametrize("voice", [v for v in VoiceId if v is not VoiceId.THORSTEN])
def test_a_catalogue_voice_that_is_not_offered_is_unavailable_and_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, voice: VoiceId
) -> None:
    """WIRE-07, WIRE-103: a catalogue voice that is not offered is unavailable."""
    with gpu_voice_stage(tmp_path, monkeypatch, _ONE_PIPER_VOICE) as stage:
        client = TestClient(stage.app())
        refused = post_speech(client, speak_body(voice))
        health = Health.model_validate(client.get("/health").json())

    assert_unavailable(refused)
    assert voice.value not in health.speaking.voices
    assert stage.starts() == []


def test_a_selection_file_from_an_older_build_is_ignored_and_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    older_selection = state / "selected-voice"
    older_selection.write_text(VoiceId.RYAN, encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("SPEECH_STATE_DIRECTORY", str(state))
    monkeypatch.setenv("HOME", str(home))
    with gpu_voice_stage(
        tmp_path, monkeypatch, StageSetup(voices=_ONE_PIPER_VOICE.voices, started=False)
    ) as stage:
        before = sorted(tmp_path.rglob("*"))
        stage.runtime.load()
        client = TestClient(app_over(stage.runtime))
        spoken = post_speech(client, speak_body(VoiceId.THORSTEN))
        refused = post_speech(client, speak_body(VoiceId.RYAN))
        after = sorted(tmp_path.rglob("*"))

    assert pcm_from_wav(spoken.content) == (PIPER_SAMPLE_RATE, PIPER_PCM)
    assert refused.status_code != HTTP_200_OK
    assert older_selection.read_text(encoding="utf-8") == VoiceId.RYAN
    assert after == before


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/voices"),
        ("POST", "/voices/ryan/load"),
        ("POST", "/voices/ryan/download"),
        ("POST", "/voices/ryan/sample/de"),
    ],
)
def test_voice_administration_is_not_mounted(method: str, path: str) -> None:
    """WIRE-130: voice administration is not mounted."""
    speaking = FakeSpeaking()
    client = TestClient(
        create_control_app(Settings(), fake_runtime(speaking), valid_callers())
    )

    response = client.request(method, path)

    assert response.status_code == HTTP_404_NOT_FOUND
    assert speaking.heard_text == ""


def test_hearing_is_not_mounted() -> None:
    """WIRE-130: hearing is not mounted."""
    client = TestClient(create_control_app(Settings(), fake_runtime(), valid_callers()))

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/hear"):
        pass
