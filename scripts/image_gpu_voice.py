"""The image proof's GPU voice cells, on a runner without a card.

The baked table lets the app that may hear name a GPU voice. By default the
image does not offer it; `tests/fixtures/image/compose.gpu-voice.yaml` offers it
as hal's override will, and its worker then finds no CUDA device.
"""

# ruff: noqa: T201 - the proof prints its cell lines as the command line output

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Final

from image_judgement import (
    GpuVoiceReport,
    baked_caller,
    listed_gpu_voice,
    require_gpu_voice_down,
    require_log_clean,
    require_no_worker_residue,
    require_offer,
    require_refusal,
    require_transcript,
    require_workers_found_no_cuda_device,
    workers_ended_before_ready,
)
from image_stack import (
    FIXTURES,
    HEARING_LANGUAGE,
    SERVICE,
    container_of,
    docker,
    run_in,
    speak_marked,
    start_with,
)
from prove_speech import SENTENCE
from speech_runtime_client.wire import Health, RefusalCode, RequestField
from speech_runtime_service.voices import RESIDENT_VOICE

if TYPE_CHECKING:
    from image_stack import Compose, Session
    from speech_runtime_service.voices import VoiceId

GPU_VOICE_OVERRIDE: Final = FIXTURES / "compose.gpu-voice.yaml"
VOICE_PARAM: Final[RequestField] = "voice"
# A worker is reaped, and its record logged, just after its refusal answers.
WORKER_LOG_WAIT_SECONDS: Final = 60


def gpu_voice_report(session: Session) -> GpuVoiceReport:
    """What the service container shows of the GPU voice."""
    output = run_in(session.compose, SERVICE, "gpu_voice.py", None)
    return GpuVoiceReport(**json.loads(output))


def require_refused_unavailable(session: Session, voice: VoiceId) -> None:
    """Fail unless an admitted request for `voice` is refused `unavailable`."""
    reply = session.speak(session.admitted_headers(), voice)
    require_refusal(reply, RefusalCode.UNAVAILABLE, param=VOICE_PARAM)


def health_of(session: Session) -> Health:
    """`/health` through the client's typed answer."""
    return Health.model_validate(session.health())


def cell_gpu_voice_not_offered(session: Session) -> None:
    """Cell 22: by default each app's listed GPU voice is refused, no worker starts."""
    refused = []
    for app in sorted(session.config.apps):
        caller = replace(session, baked=baked_caller(session.config, app))
        voice = listed_gpu_voice(session.config, app)
        require_refused_unavailable(caller, voice)
        refused.append(f"{app} naming {voice}")
    require_offer(health_of(session), [RESIDENT_VOICE])
    running = gpu_voice_report(session).workers
    session.collect_log()
    started = workers_ended_before_ready(session.logs[-1])
    if running or started:
        message = f"no GPU voice is offered, yet {running=} {started=} workers"
        raise SystemExit(message)
    print(
        f"cell 22: held; {' and '.join(refused)} refused as"
        f" {RefusalCode.UNAVAILABLE} param={VOICE_PARAM}, /health offers"
        f" {RESIDENT_VOICE} alone, workers running=0 started=0"
    )


def wait_for_workers(stack: Compose, count: int) -> str:
    """The service log once it says `count` workers ended before ready."""
    deadline = time.monotonic() + WORKER_LOG_WAIT_SECONDS
    while True:
        log = stack.run("logs", "--no-color", SERVICE).stdout
        if workers_ended_before_ready(log) >= count:
            return log
        if time.monotonic() > deadline:
            message = f"the log does not show {count} worker(s) ending before ready"
            raise SystemExit(message)
        time.sleep(1)


def container_start(stack: Compose) -> tuple[str, str, int]:
    """The service container's id, its start time and its restart count."""
    container = container_of(stack, SERVICE)
    inspected = json.loads(docker("inspect", container))[0]
    return container, inspected["State"]["StartedAt"], inspected["RestartCount"]


def cell_gpu_voice_offered(session: Session, wav: bytes) -> None:
    """Cell 23: offered on a runner without a card, the GPU voice fails alone."""
    stack = start_with(session, GPU_VOICE_OVERRIDE)
    stack.up(SERVICE)
    offered = replace(session, compose=stack)
    voice = listed_gpu_voice(session.config, offered.baked.app)
    health = health_of(offered)
    require_offer(health, [RESIDENT_VOICE, voice])
    require_gpu_voice_down(health, voice)
    print(f"cell 23: held; healthy, {voice} offered and not ready")
    started = container_start(stack)
    before = gpu_voice_report(offered)
    require_refused_unavailable(offered, voice)
    require_workers_found_no_cuda_device(wait_for_workers(stack, 1), 1)
    require_no_worker_residue(before, gpu_voice_report(offered))
    print(
        f"cell 23: held; {offered.baked.app} naming {voice} started a worker that"
        f" imported its model library on its private cache in the read-only image,"
        f" found no CUDA device and left nothing in /tmp once reaped, refused as"
        f" {RefusalCode.UNAVAILABLE} param={VOICE_PARAM}"
    )
    require_gpu_voice_down(health_of(offered), voice)
    speak_marked(offered)
    session.heard.append(
        require_transcript(
            offered.hear(offered.admitted_headers(), wav, HEARING_LANGUAGE), SENTENCE
        )
    )
    print(
        f"cell 23: held; then {voice} not ready, speaking and hearing ready,"
        f" {RESIDENT_VOICE} speaks and {offered.baked.app}'s recording is heard"
    )
    require_refused_unavailable(offered, voice)
    require_workers_found_no_cuda_device(wait_for_workers(stack, 2), 2)
    require_no_worker_residue(before, gpu_voice_report(offered))
    if container_start(stack) != started:
        message = f"the service container restarted: {started}"
        raise SystemExit(message)
    session.collect_log()
    require_log_clean(session.logs[-1], [SENTENCE], session.heard, session.tokens)
    print(
        "cell 23: held; a second request started a second worker and was refused"
        " again on finding no CUDA device, nothing is left in /tmp, the"
        " container did not restart, the log holds no spoken text"
    )
