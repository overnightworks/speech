"""The image proof's GPU voice cells: the image does not carry the GPU voice.

Every registered app may name a GPU voice. By default the image does not offer
it; `tests/fixtures/image/compose.gpu-voice.yaml` offers it as hal's override
will, and the service then finds the voice not installed.
"""

# ruff: noqa: T201 - the proof prints its cell lines as the command line output

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Final

from image_judgement import (
    first_gpu_voice,
    require_gpu_voice_down,
    require_log_clean,
    require_offer,
    require_refusal,
    require_transcript,
)
from image_stack import (
    FIXTURES,
    HEARING_LANGUAGE,
    SERVICE,
    container_of,
    docker,
    speak_marked,
    start_with,
)
from prove_speech import SENTENCE
from speech_client.wire import Health, RefusalCode, RequestField
from speech_service.voices import PIPER_VOICES

if TYPE_CHECKING:
    from image_stack import Compose, Session
    from speech_client.wire import VoiceId

GPU_VOICE_OVERRIDE: Final = FIXTURES / "compose.gpu-voice.yaml"
VOICE_PARAM: Final[RequestField] = "voice"


def require_refused_unavailable(session: Session, voice: VoiceId) -> None:
    """Fail unless an admitted request for `voice` is refused `unavailable`."""
    reply = session.speak(session.admitted_headers(), voice)
    require_refusal(reply, RefusalCode.UNAVAILABLE, param=VOICE_PARAM)


def health_of(session: Session) -> Health:
    """`/health` through the client's typed answer."""
    return Health.model_validate(session.health())


def cell_gpu_voice_not_offered(session: Session) -> None:
    """Cell 22: by default each app's GPU voice is refused and not offered."""
    refused = []
    for app in sorted(session.config.apps):
        caller = session.as_app(app)
        voice = first_gpu_voice()
        require_refused_unavailable(caller, voice)
        refused.append(f"{app} naming {voice}")
    require_offer(health_of(session), PIPER_VOICES)
    print(
        f"cell 22: held; {' and '.join(refused)} refused as"
        f" {RefusalCode.UNAVAILABLE} param={VOICE_PARAM}, /health offers"
        " the Piper voices alone"
    )


def container_start(stack: Compose) -> tuple[str, str, int]:
    """The service container's id, its start time and its restart count."""
    container = container_of(stack, SERVICE)
    inspected = json.loads(docker("inspect", container))[0]
    return container, inspected["State"]["StartedAt"], inspected["RestartCount"]


def cell_gpu_voice_offered(session: Session, wav: bytes) -> None:
    """Cell 23: offered but not carried, the service stays healthy without it."""
    stack = start_with(session, GPU_VOICE_OVERRIDE)
    stack.up(SERVICE)
    offered = replace(session, compose=stack)
    voice = first_gpu_voice()
    health = health_of(offered)
    require_offer(health, [*PIPER_VOICES, voice])
    require_gpu_voice_down(health, voice)
    print(f"cell 23: held; healthy, {voice} offered and not ready")
    started = container_start(stack)
    require_refused_unavailable(offered, voice)
    print(
        f"cell 23: held; {offered.baked.app} naming {voice} refused as"
        f" {RefusalCode.UNAVAILABLE} param={VOICE_PARAM}, the voice is not installed"
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
        f" {offered.baked.voice} speaks and {offered.baked.app}'s recording is heard"
    )
    if container_start(stack) != started:
        message = f"the service container restarted: {started}"
        raise SystemExit(message)
    session.collect_log()
    require_log_clean(session.logs[-1], [SENTENCE], session.heard, session.tokens)
    print("cell 23: held; the container did not restart, the log holds no spoken text")
