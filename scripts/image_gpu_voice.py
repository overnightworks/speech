"""The image proof's GPU voice cells, with the card and without it.

The baked table lets the app that may hear name a GPU voice. By default the
image does not offer it; `tests/fixtures/image/compose.gpu-voice.yaml` offers it
as hal's override will. On a runner without a card its worker finds no CUDA
device; on the host it loads on the card at its first request and speaks, and
the card cells show the card untouched by default, the voice refused while
another process holds the card, and loaded once the card is free again.
"""

# ruff: noqa: T201 - the proof prints its cell lines as the command line output

from __future__ import annotations

import json
import time
from dataclasses import replace
from http import HTTPStatus
from typing import TYPE_CHECKING, Final

from hearing_card import MemorySampler, card_is_freed, held_card, used_mib
from image_judgement import (
    GpuVoiceReport,
    card_figures,
    compute_pids,
    listed_gpu_voice,
    require_gpu_voice_down,
    require_log_clean,
    require_marked_wav,
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
    container_pids,
    docker,
    hearer_of,
    run_in,
    speak_marked,
    start_with,
)
from prove_speech import SENTENCE, query_nvidia
from speech_figures import FIRST_BYTE_LIMIT_S, QWEN_G0
from speech_runtime_client.wire import Health, RefusalCode, RequestField
from speech_runtime_service.voices import RESIDENT_VOICE

if TYPE_CHECKING:
    from image_stack import Compose, Reply, Session
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


def speak_voice(session: Session, voice: VoiceId) -> Reply:
    """One admitted speak with `voice`, which must answer a marked WAV."""
    reply = session.speak(session.admitted_headers(), voice)
    if reply.status != HTTPStatus.OK:
        message = f"{voice} answered {reply.status}: {reply.body[:300]!r}"
        raise SystemExit(message)
    require_marked_wav(reply.body)
    return reply


def require_voice_ready(session: Session, voice: VoiceId) -> None:
    """Fail unless `/health` reports `voice` ready beside speaking and hearing."""
    health = health_of(session)
    if not (
        health.speaking.ready
        and health.hearing.ready
        and health.speaking.voices[voice.value].ready
    ):
        message = f"/health does not report {voice} ready beside a ready service"
        raise SystemExit(message)


def require_clean_log(session: Session) -> None:
    """Fail when the service's log so far holds spoken or heard text or a token."""
    session.collect_log()
    require_log_clean(session.logs[-1], [SENTENCE], session.heard, session.tokens)


def require_piper_and_hearing(session: Session, wav: bytes) -> None:
    """Fail unless the resident voice speaks and `wav` is heard as the sentence."""
    speak_marked(session)
    hearer = hearer_of(session)
    session.heard.append(
        require_transcript(
            hearer.hear(hearer.admitted_headers(), wav, HEARING_LANGUAGE), SENTENCE
        )
    )


def cell_gpu_voice_not_offered(session: Session) -> None:
    """Cell 22: by default each app's listed GPU voice is refused, no worker starts."""
    refused = []
    for caller in (session, hearer_of(session)):
        voice = listed_gpu_voice(session.config, caller.baked.app)
        require_refused_unavailable(caller, voice)
        refused.append(f"{caller.baked.app} naming {voice}")
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


def start_offered(session: Session) -> tuple[Session, VoiceId]:
    """Recreate the service offering its GPU voice; the app that names it, and it.

    The voice is offered and not ready, beside a ready service.
    """
    stack = start_with(session, GPU_VOICE_OVERRIDE)
    stack.up(SERVICE)
    offered = replace(hearer_of(session), compose=stack)
    voice = listed_gpu_voice(session.config, offered.baked.app)
    health = health_of(offered)
    require_offer(health, [RESIDENT_VOICE, voice])
    require_gpu_voice_down(health, voice)
    return offered, voice


def cell_gpu_voice_offered(session: Session, wav: bytes, *, card: bool) -> None:
    """Cell 23: offered, the GPU voice speaks on the card, or fails alone without it."""
    offered, voice = start_offered(session)
    print(f"cell 23: held; healthy, {voice} offered and not ready")
    if card:
        speak_on_the_card(offered, voice, wav)
    else:
        fail_without_a_card(offered, voice, wav)


def speak_on_the_card(offered: Session, voice: VoiceId, wav: bytes) -> None:
    """Cell 23 with the card: the voice loads on its first request and speaks."""
    started = container_start(offered.compose)
    baseline = used_mib()
    with MemorySampler() as sampler:
        first = speak_voice(offered, voice)
    again = speak_voice(offered, voice)
    limit = FIRST_BYTE_LIMIT_S[voice]
    if first.first_byte_seconds > limit:
        message = (
            f"{voice}'s first request answered its first byte after"
            f" {first.first_byte_seconds:.1f} s, past {limit:.1f} s"
        )
        raise SystemExit(message)
    require_voice_ready(offered, voice)
    figures = card_figures(first, again, baseline, sampler.samples)
    print(
        f"cell 23: held; {voice} loaded on its first request and spoke, first byte"
        f" {first.first_byte_seconds:.1f} s within {limit:.1f} s;"
        f" load_s={figures.load_seconds:.1f} (G0 {QWEN_G0.load_seconds})"
        f" first_audio_s={figures.first_audio_seconds:.2f}"
        f" (G0 {QWEN_G0.first_audio_seconds})"
        f" rtf={figures.real_time_factor:.2f} (G0 {QWEN_G0.real_time_factor})"
        f" peak_card_mib={figures.peak_card_mib} (G0 {QWEN_G0.peak_card_mib})"
    )
    require_piper_and_hearing(offered, wav)
    if container_start(offered.compose) != started:
        message = f"the service container restarted: {started}"
        raise SystemExit(message)
    require_clean_log(offered)
    print(
        f"cell 23: held; then {RESIDENT_VOICE} speaks, {offered.baked.app}'s"
        " recording is heard, the container did not restart, and the log holds"
        " no spoken text"
    )


def fail_without_a_card(offered: Session, voice: VoiceId, wav: bytes) -> None:
    """Cell 23 without a card: each request starts a worker that finds no device."""
    stack = offered.compose
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
    require_piper_and_hearing(offered, wav)
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
    require_clean_log(offered)
    print(
        "cell 23: held; a second request started a second worker and was refused"
        " again on finding no CUDA device, nothing is left in /tmp, the"
        " container did not restart, the log holds no spoken text"
    )


def require_card_untouched(session: Session) -> None:
    """Fail when a process of the service container computes on the card."""
    on_card = compute_pids(
        query_nvidia("--query-compute-apps=pid", "--format=csv,noheader")
    ) & container_pids(session.compose, SERVICE)
    if on_card:
        message = f"the service container computes on the card: pids {on_card}"
        raise SystemExit(message)


def cell_card_untouched(session: Session) -> bytes:
    """Cell 24: by default no process of the service computes on the card.

    A process keeps the CUDA context it made, so a check after speaking and
    after hearing sees any that either made. Returns the spoken WAV.
    """
    wav = speak_marked(session)
    require_card_untouched(session)
    require_piper_and_hearing(session, wav)
    require_card_untouched(session)
    print(
        f"cell 24: held; with the card reserved and no GPU voice offered,"
        f" {RESIDENT_VOICE} speaks and {hearer_of(session).baked.app}'s recording"
        " is heard, and no process of the service container computes on the card"
    )
    return wav


def cell_card_held_then_freed(session: Session, wav: bytes) -> None:
    """Cells 25 and 26: refused while the card is held, loaded once it is free."""
    baseline = used_mib()
    offered, voice = start_offered(session)
    with held_card() as held_mib:
        require_refused_unavailable(offered, voice)
        require_gpu_voice_down(health_of(offered), voice)
        require_piper_and_hearing(offered, wav)
        print(
            f"cell 25: held; with {held_mib} MiB of the card held by another"
            f" process, {voice} is refused as {RefusalCode.UNAVAILABLE}"
            f" param={VOICE_PARAM} and stays not ready, {RESIDENT_VOICE} speaks"
            " and the recording is heard"
        )
    if not card_is_freed(baseline):
        message = f"the card did not return to {baseline} MiB once the holder ended"
        raise SystemExit(message)
    reply = speak_voice(offered, voice)
    require_voice_ready(offered, voice)
    require_clean_log(offered)
    print(
        f"cell 26: held; once the card was free, the next request loaded {voice},"
        f" which spoke after {reply.first_byte_seconds:.1f} s and is ready, and the"
        " log holds no spoken text"
    )


def card_cells(session: Session) -> None:
    """Cells 24 to 26, on the image the rebuild left in the build cache."""
    session.compose.run("build", SERVICE)
    session.compose.up()
    wav = cell_card_untouched(session)
    cell_card_held_then_freed(session, wav)
