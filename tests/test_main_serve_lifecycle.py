"""Serving listens on the configured TCP address and ends with its stop signal."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from shipped_catalogue import CATALOGUE
from speech_client.wire import VoiceId
from speech_service import __main__
from speech_service.__main__ import StopPolicy
from speech_service.config import Settings, setting_variable
from speech_service.service import Runtime, RuntimeDependencies
from tests.support.fakes_support import FakeSpeaking
from tests.support.gpu_voice_support import install_qwen
from tests.support.identity_support import (
    caller_configuration_text,
    write_caller_configuration,
)
from tests.support.launcher_support import (
    SERVER_WAIT_SECONDS,
    assert_reaped,
    closed_provider,
    server_starts,
    wait_for_starts,
)
from tests.support.serve_support import RecordedStops, serve_runtime
from tests.support.tcp_server_support import (
    answers_http,
    free_loopback_port,
    listens_on,
    wait_until_listening,
)

if TYPE_CHECKING:
    import socket

    import uvicorn

_LOADABLE_RUNTIME = """
import os
from pathlib import Path
from speech_service.catalogue import load_catalogue
from speech_service.service import Runtime, RuntimeDependencies
from speech_client.wire import VoiceId
from tests.support.fakes_support import FakeSpeaking


def loadable_runtime():
    return Runtime(
        {VoiceId.THORSTEN: None},
        dependencies=RuntimeDependencies(
            catalogue=load_catalogue(Path(os.environ["SPEECH_CATALOGUE_FILE"])),
            engine_factory=lambda _voice: FakeSpeaking(),
            missing_artifacts=lambda _voice: (),
        ),
    )
"""

_SIGTERM_SERVER = (
    """
import asyncio
import os
from speech_service import __main__
from speech_service.config import Settings
from tests.support.serve_support import serve_runtime
"""
    + _LOADABLE_RUNTIME
    + """
settings = Settings(port=int(os.environ["SPEECH_PORT"]))
raise SystemExit(
    asyncio.run(
        serve_runtime(settings, loadable_runtime())
    )
)
"""
)


def test_sigterm_stops_the_real_server_and_exits_zero_with_nothing_listening(
    tmp_path: Path,
) -> None:
    port = free_loopback_port()
    speech_root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "SPEECH_PORT": str(port),
        "SPEECH_CONFIG_FILE": str(write_caller_configuration(tmp_path)),
        "PYTHONPATH": os.pathsep.join([str(speech_root), str(speech_root / "scripts")]),
    }
    with subprocess.Popen(
        [sys.executable, "-c", _SIGTERM_SERVER], env=environment
    ) as server:
        try:
            deadline = time.monotonic() + 30
            while not answers_http(port):
                assert server.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.05)
            server.send_signal(signal.SIGTERM)
            assert server.wait(timeout=30) == 0
        finally:
            server.kill()

    assert not listens_on(port)


def test_python_dash_m_starts_the_service_listening_on_the_configured_port(
    tmp_path: Path,
) -> None:
    """WIRE-01: python dash m starts the service listening on the configured port."""
    port = free_loopback_port()
    speech_root = Path(__file__).resolve().parents[1]
    provider_root, cache = closed_provider(tmp_path, "held-ready")
    install_qwen(cache)
    environment = {
        **os.environ,
        setting_variable("port"): str(port),
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(tmp_path / "hf"),
        setting_variable("config_file"): str(write_caller_configuration(tmp_path)),
        setting_variable("device"): "cpu",
        setting_variable("voices"): VoiceId.RYAN.value,
        setting_variable("load_gpu_voices_at_start"): "1",
        setting_variable("provider_root"): str(provider_root),
        setting_variable("huggingface_cache"): str(cache),
        "PYTHONPATH": os.pathsep.join([str(speech_root), str(speech_root / "scripts")]),
    }
    with subprocess.Popen(
        [sys.executable, "-m", "speech_service"],
        cwd=tmp_path,
        env=environment,
    ) as server:
        try:
            wait_until_listening(server, port)
            (qwen_server,) = wait_for_starts(cache, 1)
            server.terminate()
            assert server.wait(timeout=30) == 0
        finally:
            server.kill()
            for started in server_starts(cache):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(started, signal.SIGKILL)

    assert_reaped(qwen_server, SERVER_WAIT_SECONDS)
    assert list(tmp_path.glob("*.sock")) == []


_SERVICE_SCRIPT = next(
    name
    for name, target in tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["scripts"].items()
    if target.startswith("speech_service.__main__:")
)
_ENTRY_POINTS = [
    [sys.executable, "-m", "speech_service"],
    [str(Path(sys.executable).parent / _SERVICE_SCRIPT)],
]


@pytest.mark.parametrize("command", _ENTRY_POINTS, ids=["module", "console script"])
def test_help_prints_usage_and_exits_zero_without_loading_settings(
    command: list[str],
) -> None:
    environment = {**os.environ, "SPEECH_PORT": "not-a-port"}
    finished = subprocess.run(
        [*command, "--help"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert finished.returncode == 0
    assert "usage:" in finished.stdout
    assert "SPEECH_CONFIG_FILE" in finished.stdout
    assert "docs/wire.md" in finished.stdout
    assert finished.stderr == ""


@pytest.mark.parametrize("command", _ENTRY_POINTS, ids=["module", "console script"])
@pytest.mark.parametrize("argument", ["--serve-faster", "--he"])
def test_an_unknown_argument_is_a_usage_error_without_starting(
    command: list[str], argument: str
) -> None:
    finished = subprocess.run(
        [*command, argument],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert finished.returncode != 0
    assert "usage:" in finished.stderr


def test_serve_listens_only_on_one_tcp_socket_of_the_configured_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WIRE-01: serve listens only on one tcp socket of the configured address."""
    built: list[uvicorn.Server] = []
    stops = RecordedStops()
    build_server = __main__.build_server

    started = asyncio.Event()

    def observed_server(app: object) -> uvicorn.Server:
        server = build_server(app)
        start = server.startup

        async def startup(sockets: list[socket.socket] | None = None) -> None:
            await start(sockets=sockets)
            started.set()

        server.startup = startup
        built.append(server)
        return server

    async def scenario() -> tuple[int, list[object]]:
        serving = asyncio.create_task(
            serve_runtime(
                Settings(port=port), runtime, StopPolicy(install_signals=stops)
            )
        )
        await started.wait()
        addresses = [
            bound.getsockname()
            for listening in built[0].servers
            for bound in listening.sockets
        ]
        stops.stop()
        return await serving, addresses

    port = free_loopback_port()
    runtime = _loadable_runtime()
    monkeypatch.setattr(__main__, "build_server", observed_server)

    outcome, addresses = asyncio.run(scenario())

    assert outcome == 0
    assert addresses == [("127.0.0.1", port)]


def _loadable_runtime() -> Runtime:
    return Runtime(
        {VoiceId.THORSTEN: None},
        dependencies=RuntimeDependencies(
            catalogue=CATALOGUE,
            engine_factory=lambda _voice: FakeSpeaking(),
            missing_artifacts=lambda _voice: (),
        ),
    )


@pytest.mark.parametrize(
    ("apps", "named"),
    [
        ('workload_client = "quiz-client"\nvoices = ["alloy"]', "voices"),
        ('workload_client = "quiz-client"\nspeak = true', "speak"),
        ('workload_client = "quiz-client"\nhear = true', "hear"),
    ],
    ids=["voices key", "speak flag", "hear flag"],
)
def test_a_bad_app_table_stops_the_start(tmp_path: Path, apps: str, named: str) -> None:
    """WIRE-112, WIRE-114, WIRE-116: a removed or unknown app key stops the start."""
    speech_root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "SPEECH_PORT": str(free_loopback_port()),
        "SPEECH_CONFIG_FILE": str(
            write_caller_configuration(
                tmp_path, caller_configuration_text(apps=f"[apps.quiz]\n{apps}\n")
            )
        ),
        "PYTHONPATH": str(speech_root),
    }

    finished = subprocess.run(
        [sys.executable, "-m", "speech_service"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert finished.returncode != 0
    crash = json.loads(finished.stdout.splitlines()[-1])
    assert crash["event"] == "process.crashed"
    assert f"apps.quiz.{named}" in crash["exception"]
    assert "alloy" not in finished.stdout + finished.stderr
