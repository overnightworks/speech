"""A GPU voice speaks each language it offers before it reports ready."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from speech_provider_contract import QwenSpeaker

from speech_client.wire import VOICES, Health, Language, VoiceId
from speech_service.provider_process import ProviderLaunch, ProviderProcess
from speech_service.speaking import QwenSpeaking
from tests.support.gpu_voice_support import StageSetup, gpu_voice_stage
from tests.support.launcher_support import (
    SERVER_WAIT_SECONDS,
    assert_reaped,
    closed_provider,
    provider_launcher,
    received_requests,
    release_speech,
    wait_for_held_speech,
    wait_for_starts,
)
from tests.support.speak_support import post_speech, speak_body

if TYPE_CHECKING:
    from pathlib import Path

_LANGUAGE_NAMES = {Language.DE: "German", Language.EN: "English"}


def _ryan_ready(client: TestClient) -> bool:
    health = Health.model_validate(client.get("/health").json())
    return health.speaking.voices[VoiceId.RYAN].ready


def test_a_voice_is_not_ready_until_it_spoke_one_request_per_language_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WIRE-126: a voice that has not warmed is not ready."""
    languages = VOICES[VoiceId.RYAN].languages
    with (
        gpu_voice_stage(
            tmp_path, monkeypatch, StageSetup("hold-first-speech")
        ) as stage,
        ThreadPoolExecutor() as pool,
    ):
        client = TestClient(stage.app())
        spoken = pool.submit(post_speech, client, speak_body(VoiceId.RYAN))
        wait_for_held_speech(stage.cache)

        ready_while_held = _ryan_ready(client)
        release_speech(stage.cache)
        response = spoken.result(timeout=SERVER_WAIT_SECONDS)
        ready_after = _ryan_ready(client)

    assert ready_while_held is False
    assert response.status_code == HTTPStatus.OK
    assert ready_after is True
    warmed = [r["language"] for r in received_requests(stage.cache)][: len(languages)]
    assert warmed == [_LANGUAGE_NAMES[language] for language in languages]


def test_a_voice_reports_not_ready_while_its_warm_up_request_is_held(
    tmp_path: Path,
) -> None:
    """WIRE-126: ready holds only once the warm-up finished."""
    root, cache = closed_provider(tmp_path, "hold-first-speech")
    voice = QwenSpeaking(
        ProviderProcess(
            ProviderLaunch(
                executable=provider_launcher(root), arguments=("--cache", str(cache))
            )
        ),
        QwenSpeaker.RYAN,
        (Language.DE,),
    )
    try:
        with ThreadPoolExecutor() as pool:
            loading = pool.submit(voice.load)
            wait_for_held_speech(cache)
            ready_while_held = voice.ready
            release_speech(cache)
            loading.result(timeout=SERVER_WAIT_SECONDS)
        ready_after = voice.ready
    finally:
        release_speech(cache)
        voice.close()

    assert ready_while_held is False
    assert ready_after is True


def test_a_refused_warm_up_fails_the_load_reaps_the_server_and_logs_no_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """WIRE-143: a voice that has not warmed fails its load."""
    with (
        gpu_voice_stage(
            tmp_path, monkeypatch, StageSetup("refuse-speech", started=False)
        ) as stage,
        caplog.at_level(logging.DEBUG),
        pytest.raises(RuntimeError, match=f"{VoiceId.RYAN.value} failed to load"),
    ):
        stage.runtime.slots[VoiceId.RYAN].load()

    (pid,) = wait_for_starts(stage.cache, 1)
    assert_reaped(pid)
    (warm_up, *_) = received_requests(stage.cache)
    logged = "\n".join(
        f"{r.getMessage()} {r.exc_text or ''} {vars(r)}" for r in caplog.records
    )
    assert str(warm_up["input"]) not in logged
