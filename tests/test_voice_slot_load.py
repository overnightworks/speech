"""A GPU voice loads once on its first request; its failure leaves Piper and hearing."""

from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from speech_runtime_provider_contract import QWEN_ARTIFACTS, QWEN_SAMPLE_RATE

from speech_runtime_client.wire import REALTIME_PATH, Health, RefusalCode
from speech_runtime_service.pcm import pcm_from_wav
from speech_runtime_service.voices import VoiceId
from tests.support.gpu_voice_support import (
    PIPER_SAMPLE_RATE,
    LoadWaits,
    StageSetup,
    gpu_voice_stage,
    health_of,
)
from tests.support.hear_support import (
    HEARD_TEXT,
    FakeHearing,
    multipart_body,
    post_recording,
    recording,
)
from tests.support.identity_support import valid_headers
from tests.support.live_support import COMPLETED, append, audio, commit, update
from tests.support.speak_support import (
    SpeakCall,
    assert_call_refusal,
    assert_refusal,
    assert_unavailable,
    gate_acquisition_began,
    gate_is_free,
    post_speech,
    speak_body,
)
from tests.support.worker_support import (
    WORKER_WAIT_SECONDS,
    LoadOutcome,
    assert_reaped,
    hold_exits,
    is_running,
    release_exit,
    release_load,
    wait_for_held_exit,
    wait_for_starts,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from httpx2 import Response as ClientResponse

    from tests.support.gpu_voice_support import GpuVoiceStage

QWEN = VoiceId.QWEN.value
PIPER = VoiceId.PIPER.value
_SERVING = StageSetup("normal")


def _health(client: TestClient) -> Health:
    return Health.model_validate(client.get("/health").json())


def _assert_spoken_at(response: ClientResponse, rate: int) -> None:
    assert response.status_code == HTTPStatus.OK
    assert pcm_from_wav(response.content)[0] == rate


def _assert_piper_and_hearing_serve(client: TestClient) -> None:
    _assert_spoken_at(post_speech(client, speak_body(PIPER)), PIPER_SAMPLE_RATE)
    heard = post_recording(client, multipart_body(file=recording()))
    assert heard.status_code == HTTPStatus.OK


def _assert_a_live_turn_is_heard(client: TestClient) -> None:
    with client.websocket_connect(
        f"{REALTIME_PATH}?intent=transcription", headers=valid_headers()
    ) as socket:
        assert socket.receive_json()["type"] == "session.created"
        socket.send_text(update())
        assert socket.receive_json()["type"] == "session.updated"
        socket.send_text(append(audio(0.5)))
        socket.send_text(commit())
        final = [socket.receive_json() for _ in range(4)][-1]

    assert (final["type"], final["transcript"]) == (COMPLETED, HEARD_TEXT)


def _speak_in_background(
    pool: ThreadPoolExecutor, stage: GpuVoiceStage, voice: str = QWEN
) -> Callable[[], ClientResponse]:
    pending = pool.submit(post_speech, TestClient(stage.app()), speak_body(voice))
    return lambda: pending.result(timeout=WORKER_WAIT_SECONDS)


def test_an_offered_voice_loads_on_its_first_request_and_then_reports_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (
        gpu_voice_stage(tmp_path, monkeypatch) as stage,
        ThreadPoolExecutor() as pool,
    ):
        client = TestClient(stage.app())
        spoken = _speak_in_background(pool, stage)
        (pid,) = wait_for_starts(stage.cache, 1)
        release_load(stage.cache, pid, LoadOutcome.READY)

        _assert_spoken_at(spoken(), QWEN_SAMPLE_RATE)
        health = _health(client)

    assert stage.starts() == [pid]
    assert health.speaking.voices[QWEN].ready is True


def test_requests_during_one_load_share_it_and_are_served_one_after_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (
        gpu_voice_stage(tmp_path, monkeypatch) as stage,
        ThreadPoolExecutor() as pool,
    ):
        waits = LoadWaits(stage.runtime)
        spoken = [_speak_in_background(pool, stage) for _ in range(3)]
        waits.wait_for(3)
        (pid,) = wait_for_starts(stage.cache, 1)
        release_load(stage.cache, pid, LoadOutcome.READY)

        for response in spoken:
            _assert_spoken_at(response(), QWEN_SAMPLE_RATE)

    assert stage.starts() == [pid]


def test_a_cancelled_waiter_leaves_the_shared_load_running_for_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario(stage: GpuVoiceStage) -> tuple[list[int], Health]:
        app = stage.app()
        waits = LoadWaits(stage.runtime)
        calls = [SpeakCall(app, speak_body(QWEN)) for _ in range(3)]
        await asyncio.to_thread(waits.wait_for, 3)
        (pid,) = await asyncio.to_thread(wait_for_starts, stage.cache, 1)
        calls[0].task.cancel()
        await asyncio.gather(calls[0].task, return_exceptions=True)
        release_load(stage.cache, pid, LoadOutcome.READY)
        for call in calls[1:]:
            await call.finish()
        return [call.status_code for call in calls[1:]], await health_of(app)

    with gpu_voice_stage(tmp_path, monkeypatch) as stage:
        statuses, health = asyncio.run(scenario(stage))

    assert statuses == [HTTPStatus.OK, HTTPStatus.OK]
    assert len(stage.starts()) == 1
    assert health.speaking.voices[QWEN].ready is True


def test_the_busy_wait_of_a_request_that_waited_for_the_load_starts_at_its_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    busy_wait_seconds = 0.2

    async def scenario(stage: GpuVoiceStage) -> list[int]:
        app = stage.app()
        waits = LoadWaits(stage.runtime)
        calls = [SpeakCall(app, speak_body(QWEN)) for _ in range(2)]
        await asyncio.to_thread(waits.wait_for, 2)
        # Both requests had their tokens accepted before they took hold of the load.
        passed = time.monotonic() + busy_wait_seconds
        (pid,) = await asyncio.to_thread(wait_for_starts, stage.cache, 1)
        while time.monotonic() <= passed:  # noqa: ASYNC110 - the load outlasts the busy wait
            await asyncio.sleep(0.01)
        release_load(stage.cache, pid, LoadOutcome.READY)
        for call in calls:
            await call.finish()
        return [call.status_code for call in calls]

    with gpu_voice_stage(
        tmp_path, monkeypatch, busy_wait_seconds=busy_wait_seconds
    ) as stage:
        statuses = asyncio.run(scenario(stage))

    assert statuses == [HTTPStatus.OK, HTTPStatus.OK]


def test_piper_hearing_and_health_answer_while_a_gpu_voice_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (
        gpu_voice_stage(tmp_path, monkeypatch) as stage,
        ThreadPoolExecutor() as pool,
    ):
        client = TestClient(stage.app())
        spoken = _speak_in_background(pool, stage)
        (pid,) = wait_for_starts(stage.cache, 1)

        _assert_piper_and_hearing_serve(client)
        _assert_a_live_turn_is_heard(client)
        health = _health(client)
        release_load(stage.cache, pid, LoadOutcome.READY)
        _assert_spoken_at(spoken(), QWEN_SAMPLE_RATE)

    assert health.speaking.ready is True
    assert health.speaking.voices[PIPER].ready is True
    assert health.speaking.voices[QWEN].ready is False
    assert health.hearing.ready is True


@pytest.mark.parametrize(
    ("behavior", "outcome", "waiters"),
    [
        ("held-ready", LoadOutcome.FAILED, 2),
        ("held-ready", LoadOutcome.EXIT, 2),
        ("startup-exit", None, 1),
        ("startup-failed", None, 1),
    ],
)
def test_a_failed_load_refuses_every_waiter_and_leaves_piper_and_hearing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    outcome: LoadOutcome | None,
    waiters: int,
) -> None:
    fatal = threading.Event()
    with (
        gpu_voice_stage(tmp_path, monkeypatch, StageSetup(behavior)) as stage,
        ThreadPoolExecutor() as pool,
    ):
        stage.runtime.set_fatal_callback(fatal.set)
        client = TestClient(stage.app())
        waits = LoadWaits(stage.runtime)
        refused = [_speak_in_background(pool, stage) for _ in range(waiters)]
        waits.wait_for(waiters)
        (pid,) = wait_for_starts(stage.cache, 1)
        if outcome is not None:
            release_load(stage.cache, pid, outcome)

        for response in refused:
            assert_unavailable(response())
        health = _health(client)
        _assert_piper_and_hearing_serve(client)

    assert health.speaking.voices[QWEN].ready is False
    assert health.speaking.ready is True
    assert fatal.is_set() is False
    assert_reaped(pid)


def test_the_request_after_a_failed_load_starts_a_new_worker_once_the_first_is_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (
        gpu_voice_stage(tmp_path, monkeypatch) as stage,
        ThreadPoolExecutor() as pool,
    ):
        refused = _speak_in_background(pool, stage)
        (first,) = wait_for_starts(stage.cache, 1)
        release_load(stage.cache, first, LoadOutcome.FAILED)
        assert_unavailable(refused())
        spoken = _speak_in_background(pool, stage)
        _, second = wait_for_starts(stage.cache, 2)
        assert_reaped(first)
        release_load(stage.cache, second, LoadOutcome.READY)

        _assert_spoken_at(spoken(), QWEN_SAMPLE_RATE)


def test_a_waiter_of_a_failed_load_is_refused_though_a_retry_loaded_before_it_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario(stage: GpuVoiceStage) -> SpeakCall:
        waits = LoadWaits(stage.runtime)
        call = SpeakCall(stage.app(), speak_body(QWEN))
        await asyncio.to_thread(waits.wait_for, 1)
        (first,) = await asyncio.to_thread(wait_for_starts, stage.cache, 1)
        # The loop stays blocked until the retry has loaded, so the waiter of the
        # failed load resumes only once a ready engine replaced it.
        failed_load = stage.runtime.load_on_demand(VoiceId.QWEN)
        release_load(stage.cache, first, LoadOutcome.FAILED)
        failed_load.result(timeout=WORKER_WAIT_SECONDS)
        retry = stage.runtime.load_on_demand(VoiceId.QWEN)
        _, second = wait_for_starts(stage.cache, 2)
        release_load(stage.cache, second, LoadOutcome.READY)
        retry.result(timeout=WORKER_WAIT_SECONDS)
        await call.finish()
        return call

    with gpu_voice_stage(tmp_path, monkeypatch) as stage:
        call = asyncio.run(scenario(stage))

    assert_call_refusal(
        call, HTTPStatus.SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE, "voice"
    )


def test_a_request_during_a_failed_loads_cleanup_shares_its_refusal_and_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (
        gpu_voice_stage(tmp_path, monkeypatch) as stage,
        ThreadPoolExecutor() as pool,
    ):
        hold_exits(stage.cache)
        client = TestClient(stage.app())
        waits = LoadWaits(stage.runtime)
        first = _speak_in_background(pool, stage)
        (pid,) = wait_for_starts(stage.cache, 1)
        release_load(stage.cache, pid, LoadOutcome.FAILED)
        wait_for_held_exit(stage.cache, pid)
        second = _speak_in_background(pool, stage)
        waits.wait_for(2)

        health = _health(client)
        _assert_piper_and_hearing_serve(client)
        still_running = is_running(pid)
        release_exit(stage.cache, pid)
        assert_unavailable(first())
        assert_unavailable(second())

    assert still_running is True
    assert health.speaking.voices[QWEN].ready is False
    assert stage.starts() == [pid]
    assert_reaped(pid)


@pytest.mark.parametrize("death", ["killed while idle", "failed synthesis"])
def test_a_worker_that_dies_after_it_was_ready_turns_its_voice_unready_and_reloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, death: str
) -> None:
    fatal = threading.Event()
    with gpu_voice_stage(tmp_path, monkeypatch, _SERVING) as stage:
        stage.runtime.set_fatal_callback(fatal.set)
        client = TestClient(stage.app(), raise_server_exceptions=False)
        _assert_spoken_at(post_speech(client, speak_body(QWEN)), QWEN_SAMPLE_RATE)
        (first,) = stage.starts()
        if death == "killed while idle":
            os.kill(first, signal.SIGKILL)
        else:
            post_speech(client, speak_body(QWEN, "failed"))
        deadline = time.monotonic() + WORKER_WAIT_SECONDS
        while _health(client).speaking.voices[QWEN].ready:
            assert time.monotonic() < deadline
            time.sleep(0.01)

        _assert_spoken_at(post_speech(client, speak_body(QWEN)), QWEN_SAMPLE_RATE)

    _, _reloaded = stage.starts()
    assert_reaped(first)
    assert fatal.is_set() is False


def test_a_crash_mid_synthesis_refuses_the_waiter_after_its_lease_and_the_next_reloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario(
        stage: GpuVoiceStage,
    ) -> tuple[object, SpeakCall, list[int], bool]:
        app = stage.app()
        warm = SpeakCall(app, speak_body(QWEN))
        await warm.finish()
        (pid,) = stage.starts()
        streaming = SpeakCall(app, speak_body(QWEN, "wait-cancel"))
        assert await streaming.status() == HTTPStatus.OK
        queued = gate_acquisition_began(stage.runtime, VoiceId.QWEN)
        waiting = SpeakCall(app, speak_body(QWEN))
        await queued.wait()
        os.kill(pid, signal.SIGKILL)
        (streamed,) = await asyncio.gather(streaming.task, return_exceptions=True)
        await waiting.finish()
        free = gate_is_free(stage.runtime, VoiceId.QWEN)
        later = [SpeakCall(app, speak_body(voice)) for voice in (PIPER, QWEN)]
        for call in later:
            await call.finish()
        return streamed, waiting, [call.status_code for call in later], free

    with gpu_voice_stage(tmp_path, monkeypatch, _SERVING) as stage:
        streamed, waiting, statuses, free = asyncio.run(scenario(stage))

    assert isinstance(streamed, Exception)
    assert_call_refusal(
        waiting, HTTPStatus.SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE, "voice"
    )
    assert free is True
    assert statuses == [HTTPStatus.OK, HTTPStatus.OK]
    _crashed, _reloaded = stage.starts()


@pytest.mark.parametrize("missing", ["weights", "entrypoint"])
def test_an_offered_voice_whose_files_are_missing_is_refused_and_named_in_the_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    missing: str,
) -> None:
    with gpu_voice_stage(
        tmp_path, monkeypatch, StageSetup("normal", qwen_installed=missing != "weights")
    ) as stage:
        entrypoint = stage.settings.provider_root / "qwen/.venv/bin"
        (worker,) = entrypoint.iterdir()
        if missing == "entrypoint":
            worker.unlink()
        client = TestClient(stage.app())
        refused = [post_speech(client, speak_body(QWEN)) for _ in range(2)]
        _assert_piper_and_hearing_serve(client)
        health = _health(client)

    for response in refused:
        assert_unavailable(response)
    assert stage.starts() == []
    assert health.speaking.voices[QWEN].ready is False
    named = QWEN_ARTIFACTS if missing == "weights" else (worker.name,)
    refusal_log = [r.getMessage() for r in caplog.records if QWEN in r.getMessage()]
    assert all(any(name in line for line in refusal_log) for name in named)


def test_while_the_start_load_runs_no_voice_is_ready_and_every_voice_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ear = _HeldEar()
    with gpu_voice_stage(
        tmp_path, monkeypatch, StageSetup("normal", hearing=ear, started=False)
    ) as stage:
        loader = threading.Thread(target=stage.runtime.load)
        loader.start()
        try:
            assert ear.loading.wait(WORKER_WAIT_SECONDS)
            client = TestClient(stage.app())
            health = _health(client)
            refused = [
                post_speech(client, speak_body(voice)) for voice in (PIPER, QWEN)
            ]
        finally:
            ear.release_load.set()
            loader.join(WORKER_WAIT_SECONDS)

    assert health.speaking.ready is False
    assert not any(voice.ready for voice in health.speaking.voices.values())
    for response in refused:
        assert_unavailable(response)
    assert stage.starts() == []


class _HeldEar(FakeHearing):
    """An ear whose load waits for the test, so the start load stays running."""

    def __init__(self) -> None:
        super().__init__()
        self.loading = threading.Event()
        self.release_load = threading.Event()

    def load(self) -> None:
        self.loading.set()
        assert self.release_load.wait(WORKER_WAIT_SECONDS)


def test_once_shutdown_begins_health_and_the_refusal_agree_that_no_voice_is_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with gpu_voice_stage(tmp_path, monkeypatch, _SERVING) as stage:
        client = TestClient(stage.app())
        _assert_spoken_at(post_speech(client, speak_body(QWEN)), QWEN_SAMPLE_RATE)
        stage.runtime.begin_shutdown()
        health = _health(client)
        refused = [post_speech(client, speak_body(voice)) for voice in (PIPER, QWEN)]

    assert health.speaking.ready is False
    assert not any(voice.ready for voice in health.speaking.voices.values())
    for response in refused:
        assert_unavailable(response)


def test_a_loaded_gpu_voice_stays_loaded_and_no_route_lists_voices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with gpu_voice_stage(tmp_path, monkeypatch, _SERVING) as stage:
        client = TestClient(stage.app())
        _assert_spoken_at(post_speech(client, speak_body(QWEN)), QWEN_SAMPLE_RATE)
        listed = client.get("/v1/voices", headers=valid_headers())
        health = _health(client)
        _assert_spoken_at(post_speech(client, speak_body(QWEN)), QWEN_SAMPLE_RATE)
        (pid,) = stage.starts()
        running = is_running(pid)

    assert_refusal(listed, HTTPStatus.NOT_FOUND, RefusalCode.NOT_FOUND, None)
    assert health.speaking.voices[QWEN].ready is True
    assert running is True
