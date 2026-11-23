"""A service started from its settings, over a fake Piper, ear and Qwen server."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import httpx2
from speech_provider_contract import (
    QWEN_ARTIFACTS,
    QWEN_MODEL_ID,
    QWEN_REVISION,
)

from shipped_catalogue import PIPER_VOICES
from speech_client.wire import HEALTH_PATH, Health, VoiceId
from speech_service.__main__ import StopPolicy
from speech_service.config import Settings, setting_variable
from speech_service.control import create_control_app
from speech_service.hearing import DRAIN_TURN_SECONDS, Hearing
from speech_service.onnxruntime_telemetry import ONNXRUNTIME_TELEMETRY_SWITCH
from speech_service.service import Runtime
from speech_service.voices import piper_voice_paths
from tests.support.hear_support import FakeHearing
from tests.support.identity_support import (
    valid_callers,
    valid_headers,
    write_caller_configuration,
)
from tests.support.launcher_support import (
    SERVER_WAIT_SECONDS,
    closed_provider,
    fail_every_reap,
    release_exit,
    server_starts,
    wait_for_starts,
)
from tests.support.serve_support import RecordedStops, serve_runtime
from tests.support.speak_support import ASGI_TIMEOUT_SECONDS, wav_body
from tests.support.tcp_server_support import (
    answers_http,
    free_loopback_port,
    wait_until_listening,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping
    from concurrent.futures import Future

    import pytest
    from fastapi import FastAPI

    from speech_service.service import SpeakingEngine

PIPER_SAMPLE_RATE = 22_050
PIPER_PCM = b"\x00\x01"
LOOPBACK = "127.0.0.1"
STOP_GRACE_SECONDS = 1.0


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


_RECORDING_PIPER = f"""
import os
from pathlib import Path
from types import SimpleNamespace


class PiperVoice:
    @staticmethod
    def load(_path, **_options):
        Path(__file__).with_name("piper-observed").write_text(
            repr(os.environ.get({ONNXRUNTIME_TELEMETRY_SWITCH!r})), encoding="utf-8"
        )
        return SimpleNamespace(config=SimpleNamespace(sample_rate=22050))
"""


@dataclass(frozen=True)
class StartedService:
    """`python -m speech_service` listening, with its Qwen server's load held."""

    process: subprocess.Popen[bytes]
    port: int
    cache: Path
    qwen_server: int
    modules: Path

    def telemetry_switch_at_piper_load(self) -> str | None:
        """The telemetry switch as the service held it when a Piper voice loaded.

        Piper voices load before the Qwen server starts, so the record exists once
        the service listens.
        """
        return ast.literal_eval(
            (self.modules / "piper-observed").read_text(encoding="utf-8")
        )


@contextmanager
def started_service(
    tmp_path: Path,
    *,
    environment_changes: Mapping[str, str | None] | None = None,
) -> Iterator[StartedService]:
    """Start the service over a held fake Qwen load; end it and its server on exit.

    The service serves one Piper and one Qwen voice; a recording fake `piper` module
    is importable by it.
    `environment_changes` sets (a string) or removes (`None`) variables of the
    inherited environment.
    """
    port = free_loopback_port()
    speech_root = Path(__file__).resolve().parents[2]
    provider_root, cache = closed_provider(tmp_path, "held-ready")
    install_qwen(cache)
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "piper.py").write_text(_RECORDING_PIPER, encoding="utf-8")
    voice_cache = tmp_path / "piper"
    voice_cache.mkdir()
    for voice in PIPER_VOICES:
        for path in piper_voice_paths(voice_cache, voice.value):
            path.touch()
    served_voices = f"{VoiceId.THORSTEN.value},{VoiceId.RYAN.value}"
    environment = {
        **os.environ,
        setting_variable("port"): str(port),
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(tmp_path / "hf"),
        setting_variable("config_file"): str(write_caller_configuration(tmp_path)),
        setting_variable("device"): "cpu",
        setting_variable("voices"): served_voices,
        setting_variable("voice_cache"): str(voice_cache),
        setting_variable("load_gpu_voices_at_start"): "1",
        setting_variable("provider_root"): str(provider_root),
        setting_variable("huggingface_cache"): str(cache),
        "PYTHONPATH": os.pathsep.join(
            [str(speech_root), str(speech_root / "scripts"), str(modules)]
        ),
    }
    for name, value in (environment_changes or {}).items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    with subprocess.Popen(
        [sys.executable, "-m", "speech_service"], cwd=tmp_path, env=environment
    ) as process:
        try:
            wait_until_listening(process, port)
            (qwen_server,) = wait_for_starts(cache, 1)
            yield StartedService(process, port, cache, qwen_server, modules)
        finally:
            process.kill()
            for started in server_starts(cache):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(started, signal.SIGKILL)


def install_piper(monkeypatch: pytest.MonkeyPatch, voice_cache: Path) -> None:
    """Piper's files in `voice_cache`, loaded by a fake `piper` module."""

    class LocalVoice:
        config = SimpleNamespace(sample_rate=PIPER_SAMPLE_RATE)

        def synthesize(self, _text: str) -> Iterable[SimpleNamespace]:
            return [SimpleNamespace(audio_int16_bytes=PIPER_PCM)]

    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(
            PiperVoice=SimpleNamespace(load=lambda _path, **_options: LocalVoice())
        ),
    )
    voice_cache.mkdir(exist_ok=True)
    for voice in PIPER_VOICES:
        for path in piper_voice_paths(voice_cache, voice.value):
            path.touch()


@dataclass(frozen=True, slots=True)
class GpuVoiceStage:
    """The runtime `Runtime.from_settings` built, its app and the server cache."""

    runtime: Runtime
    settings: Settings
    cache: Path
    hearing: FakeHearing

    def app(self) -> FastAPI:
        """The private app over the runtime, with one registered app."""
        return create_control_app(self.settings, self.runtime, valid_callers())

    def starts(self) -> list[int]:
        """The pids of every server started so far."""
        return server_starts(self.cache)


@dataclass(frozen=True, slots=True)
class StageSetup:
    """The fake server's behaviour and what the stage installs and offers.

    The start load has run unless `started` is false. With `reaps_fail`, every
    server's cleanup fails once it is reaped, so the stage's close may raise.
    With `health_deadline_seconds`, a server that never answers fails its load
    after that long.
    """

    behavior: str = "held-ready"
    voices: frozenset[VoiceId] = frozenset({VoiceId.THORSTEN, VoiceId.RYAN})
    qwen_installed: bool = True
    hearing: FakeHearing | None = None
    started: bool = True
    reaps_fail: bool = False
    health_deadline_seconds: float | None = None
    drain_turn_seconds: float = DRAIN_TURN_SECONDS


@contextlib.contextmanager
def gpu_voice_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setup: StageSetup | None = None,
    **settings: float,
) -> Iterator[GpuVoiceStage]:
    """A service built by `Runtime.from_settings` as `setup` says.

    On exit every held server exit is released and the runtime is closed.
    """
    setup = setup or StageSetup()
    behavior = setup.behavior
    root, cache = closed_provider(tmp_path, behavior)
    install_piper(monkeypatch, tmp_path / "piper")
    if setup.qwen_installed:
        install_qwen(cache)
    ear = setup.hearing or FakeHearing()
    if setup.reaps_fail:
        fail_every_reap(monkeypatch)
    if setup.health_deadline_seconds is not None:
        monkeypatch.setattr(
            "speech_service.provider_process.HEALTH_DEADLINE_SECONDS",
            setup.health_deadline_seconds,
        )
    configured = Settings(
        voices=setup.voices,
        provider_root=root,
        huggingface_cache=cache,
        voice_cache=tmp_path / "piper",
        device="cpu",
        **settings,
    )
    runtime = Runtime.from_settings(configured)
    runtime.hearing = Hearing(
        ear, runtime.clock, drain_turn_seconds=setup.drain_turn_seconds
    )
    try:
        if setup.started:
            runtime.load()
        yield GpuVoiceStage(runtime, configured, cache, ear)
    finally:
        for pid in server_starts(cache):
            release_exit(cache, pid)
        try:
            runtime.close()
        except RuntimeError:
            if not setup.reaps_fail:
                raise


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

        def counted(voice: VoiceId) -> Future[SpeakingEngine | None] | None:
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
                lambda: self._count >= count, timeout=SERVER_WAIT_SECONDS
            ), f"{count} requests never waited for the load"


def speak_headers() -> dict[str, str]:
    """The headers of a speak request with a valid pair of tokens."""
    return {"content-type": "application/json", **valid_headers()}


class Serving:
    """`__main__`'s serve loop over a stage's runtime, on a loopback listener.

    It stops as its SIGTERM handler stops it; `fatal` notes the fatal callback.
    """

    def __init__(self, stage: GpuVoiceStage) -> None:
        """Hold the stage's runtime; nothing serves until `start`."""
        self._stage = stage
        self._stops = RecordedStops()
        self.fatal = threading.Event()
        runtime = stage.runtime
        set_fatal_callback = runtime.set_fatal_callback

        def noting_fatal(callback: Callable[[], None]) -> None:
            def both() -> None:
                self.fatal.set()
                callback()

            set_fatal_callback(both)

        runtime.set_fatal_callback = noting_fatal
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind((LOOPBACK, 0))
        self.listener.listen()
        self.port: int = self.listener.getsockname()[1]
        self._task: asyncio.Task[int] | None = None

    async def start(self) -> httpx2.AsyncClient:
        """Serve; once the start load ended, a client whose tokens are valid."""
        self._task = asyncio.create_task(
            serve_runtime(
                self._stage.settings,
                self._stage.runtime,
                StopPolicy(
                    install_signals=self._stops, grace_seconds=STOP_GRACE_SECONDS
                ),
                listener=self.listener,
            )
        )
        deadline = time.monotonic() + SERVER_WAIT_SECONDS
        while not await asyncio.to_thread(answers_http, self.port):
            assert time.monotonic() < deadline
        while self._stage.runtime.loading:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        return httpx2.AsyncClient(
            base_url=f"http://{LOOPBACK}:{self.port}",
            timeout=SERVER_WAIT_SECONDS,
            headers=speak_headers(),
        )

    def sigterm(self) -> None:
        """Stop the serve loop as its SIGTERM handler does."""
        self._stops.stop()

    @property
    def task(self) -> asyncio.Task[int]:
        """The running serve loop; its result is the exit status."""
        assert self._task is not None
        return self._task
