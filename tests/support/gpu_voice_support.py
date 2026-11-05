"""A service started from its settings, over a fake Piper, ear and worker."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

from speech_runtime_provider_contract import (
    QWEN_ARTIFACTS,
    QWEN_MODEL_ID,
    QWEN_REVISION,
)

from speech_runtime_client.wire import HEALTH_PATH, Health
from speech_runtime_service import service
from speech_runtime_service.config import Settings
from speech_runtime_service.control import create_control_app
from speech_runtime_service.hearing import Hearing
from speech_runtime_service.service import Runtime
from speech_runtime_service.voices import DEFAULT_SPEAKING_MODEL, VoiceId
from tests.support.hear_support import FakeHearing
from tests.support.identity_support import valid_callers
from tests.support.speak_support import ASGI_TIMEOUT_SECONDS, wav_body
from tests.support.worker_support import (
    WORKER_WAIT_SECONDS,
    closed_provider,
    release_exit,
    worker_starts,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from concurrent.futures import Future
    from pathlib import Path

    import pytest
    from fastapi import FastAPI

    from speech_runtime_service.clock import Clock

PIPER_SAMPLE_RATE = 22_050
PIPER_PCM = b"\x00\x01"


def touch_snapshot(
    cache: Path, repository: str, revision: str, files: Iterable[str]
) -> None:
    """The files of `repository` at `revision`, as the Hugging Face cache keeps them."""
    snapshot = cache / f"models--{repository.replace('/', '--')}" / "snapshots"
    for name in files:
        artifact = snapshot / revision / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()


def install_qwen(cache: Path) -> None:
    """Qwen's pinned weights in `cache`."""
    touch_snapshot(cache, QWEN_MODEL_ID, QWEN_REVISION, QWEN_ARTIFACTS)


def install_piper(monkeypatch: pytest.MonkeyPatch, voice_cache: Path) -> None:
    """Piper's files in `voice_cache`, loaded by a fake `piper` module."""

    class LocalVoice:
        config = SimpleNamespace(sample_rate=PIPER_SAMPLE_RATE)

        def synthesize(self, _text: str) -> Iterable[SimpleNamespace]:
            return [SimpleNamespace(audio_int16_bytes=PIPER_PCM)]

    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=lambda _path: LocalVoice())),
    )
    voice_cache.mkdir(exist_ok=True)
    (voice_cache / f"{DEFAULT_SPEAKING_MODEL}.onnx").touch()
    (voice_cache / f"{DEFAULT_SPEAKING_MODEL}.onnx.json").touch()


@dataclass(frozen=True, slots=True)
class GpuVoiceStage:
    """The runtime `Runtime.from_settings` built, its app, and the worker's cache."""

    runtime: Runtime
    settings: Settings
    cache: Path
    hearing: FakeHearing

    def app(self) -> FastAPI:
        """The private app over the runtime; its one app lists every voice and hears."""
        return create_control_app(self.settings, self.runtime, valid_callers(hear=True))

    def starts(self) -> list[int]:
        """The pids of every worker started so far."""
        return worker_starts(self.cache)


@dataclass(frozen=True, slots=True)
class StageSetup:
    """The fake worker's behaviour and what the stage installs and offers.

    The start load has run unless `started` is false.
    """

    behavior: str = "held-ready"
    gpu_voices: frozenset[VoiceId] = frozenset({VoiceId.QWEN})
    qwen_installed: bool = True
    hearing: FakeHearing | None = None
    started: bool = True


@contextlib.contextmanager
def gpu_voice_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setup: StageSetup | None = None,
    **settings: float,
) -> Iterator[GpuVoiceStage]:
    """A service built by `Runtime.from_settings` as `setup` says.

    On exit every held worker exit is released and the runtime is closed.
    """
    setup = setup or StageSetup()
    behavior = setup.behavior
    root, cache = closed_provider(tmp_path, behavior)
    install_piper(monkeypatch, tmp_path / "piper")
    if setup.qwen_installed:
        install_qwen(cache)
    ear = setup.hearing or FakeHearing()

    def fake_hearing(_settings: Settings, clock: Clock) -> Hearing:
        return Hearing(ear, clock)

    monkeypatch.setattr(service, "hearing_from_settings", fake_hearing)
    configured = Settings(
        gpu_voices=setup.gpu_voices,
        provider_root=root,
        huggingface_cache=cache,
        voice_cache=tmp_path / "piper",
        device="cpu",
        **settings,
    )
    runtime = Runtime.from_settings(configured)
    try:
        if setup.started:
            runtime.load()
        yield GpuVoiceStage(runtime, configured, cache, ear)
    finally:
        for pid in worker_starts(cache):
            release_exit(cache, pid)
        runtime.close()


async def health_of(app: FastAPI) -> Health:
    """GET /health through the raw ASGI boundary, on the running loop."""
    messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "path": HEALTH_PATH,
        "raw_path": HEALTH_PATH.encode(),
        "query_string": b"",
        "headers": [],
        "client": ("test", 123),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=ASGI_TIMEOUT_SECONDS)
    return Health.model_validate_json(wav_body(messages))


class LoadWaits:
    """Counts the requests that took hold of a voice's load through the runtime."""

    def __init__(self, runtime: Runtime) -> None:
        """Observe `runtime.load_on_demand`, which hands each request its load."""
        self._count = 0
        self._changed = threading.Condition()
        hand_out = runtime.load_on_demand

        def counted(voice: VoiceId) -> Future[None] | None:
            load = hand_out(voice)
            with self._changed:
                self._count += 1
                self._changed.notify_all()
            return load

        runtime.load_on_demand = counted

    def wait_for(self, count: int) -> None:
        """Return once `count` requests hold the load; fails after a bounded wait."""
        with self._changed:
            assert self._changed.wait_for(
                lambda: self._count >= count, timeout=WORKER_WAIT_SECONDS
            ), f"{count} requests never waited for the load"
