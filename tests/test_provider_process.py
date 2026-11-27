"""The service owns Qwen's server process from its start to its stop."""

from __future__ import annotations

import contextlib
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from speech_provider_contract import QwenSpeaker

from shipped_catalogue import CATALOGUE
from speech_client.wire import Language, VoiceId
from speech_service.__main__ import StopPolicy
from speech_service.provider_process import (
    CLOSE_SECONDS,
    HEALTH_DEADLINE_SECONDS,
    OWNER_JOIN_SECONDS,
    ProviderCleanupFailedError,
    ProviderFailedError,
    ProviderLaunch,
    ProviderNotReadyError,
    ProviderProcess,
)
from speech_service.service import Runtime, RuntimeDependencies
from speech_service.speaking import QwenSpeaking
from tests.support.elapsed_support import well_before
from tests.support.launcher_support import (
    LoadOutcome,
    assert_reaped,
    closed_provider,
    debug_log,
    fail_every_reap,
    health_deadline_from_scenario,
    numba_cache,
    observed,
    provider_launcher,
    provider_process,
    provider_server,
    qwen_voice,
    release_load,
    wait_for_starts,
)
from tests.support.poll_support import poll
from tests.support.process_state import has_ended

if TYPE_CHECKING:
    from collections.abc import Callable

_SCRATCH_VARIABLES = ("NUMBA_CACHE_DIR", "TMPDIR", "HOME")
_OWNER = "speech_service.provider_process"


def _runtime(factory: Callable[[VoiceId], QwenSpeaking]) -> Runtime:
    return Runtime(
        {VoiceId.THORSTEN: None},
        dependencies=RuntimeDependencies(
            catalogue=CATALOGUE,
            engine_factory=factory,
            missing_artifacts=lambda _voice: (),
        ),
    )


def _finishes_within_timeout(target: Callable[[], object]) -> bool:
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=5)
    return not thread.is_alive()


def _load_in_thread(
    server: ProviderProcess,
) -> tuple[threading.Thread, list[ProviderFailedError]]:
    failures: list[ProviderFailedError] = []

    def load() -> None:
        try:
            server.load()
        except ProviderFailedError as failure:
            failures.append(failure)

    loading = threading.Thread(target=load, daemon=True)
    loading.start()
    return loading, failures


def _load_quietly(server: ProviderProcess) -> None:
    with contextlib.suppress(ProviderFailedError):
        server.load()


@pytest.fixture(params=["python3.12", "python3.14"])
def python_directory(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_the_server_starts_from_a_load_thread_that_ends_and_keeps_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, python_directory: str
) -> None:
    root, cache = closed_provider(tmp_path)
    cublas = root / f"qwen/.venv/lib/{python_directory}/site-packages/nvidia/cublas/lib"
    cublas.mkdir(parents=True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("PATH", "parent-executable-search-path")
    real_popen = subprocess.Popen
    launches: list[tuple[list[str], dict[str, object]]] = []

    def recording_popen(
        arguments: list[str], **keywords: object
    ) -> subprocess.Popen[bytes]:
        launches.append((arguments, keywords))
        return real_popen(arguments, **keywords)

    monkeypatch.setattr(f"{_OWNER}.subprocess.Popen", recording_popen)
    server = provider_process(root, cache)

    assert _finishes_within_timeout(server.load)

    assert server.ready is True
    assert "speech-provider-owner" in {thread.name for thread in threading.enumerate()}
    cwd, environment, pid, arguments = observed(cache)
    server.close()
    assert cwd == str(root / "qwen")
    assert arguments["expected_parent_pid"] == os.getpid()
    assert arguments["cache"] == str(cache)
    assert Path(arguments["socket"]).name == "server.sock"
    assert environment["LD_LIBRARY_PATH"] == str(cublas)
    assert environment["PATH"] == os.defpath
    assert {environment[name] for name in _SCRATCH_VARIABLES} == {environment["TMPDIR"]}
    [(launch_arguments, launch)] = launches
    assert launch_arguments[0] == str(provider_launcher(root))
    assert launch["stdin"] == subprocess.DEVNULL
    assert launch["stdout"] == subprocess.DEVNULL
    assert launch["stderr"] == subprocess.PIPE
    assert launch["start_new_session"] is True
    assert_reaped(pid)


@pytest.mark.parametrize("behavior", ["normal", "startup-exit"])
def test_each_server_keeps_its_state_in_a_private_directory_gone_once_reaped(
    tmp_path: Path, behavior: str
) -> None:
    caches = []
    servers = []
    for name in ("first", "second"):
        root, cache = closed_provider(tmp_path / name, behavior)
        servers.append(provider_process(root, cache))
        caches.append(cache)
        _load_quietly(servers[-1])
    for server in servers:
        server.close()

    (first, first_mode), (second, second_mode) = map(numba_cache, caches)
    assert first != second
    assert first_mode == second_mode == stat.S_IRWXU
    assert not first.exists()
    assert not second.exists()
    assert not any(debug_log(cache).exists() for cache in caches)


def test_each_start_listens_on_a_fresh_socket_in_a_directory_only_its_user_enters(
    tmp_path: Path,
) -> None:
    root, cache = closed_provider(tmp_path)
    sockets = []
    for _ in range(2):
        server = provider_process(root, cache)
        server.load()
        sockets.append(Path(observed(cache)[3]["socket"]))
        assert (cache / "socket-directory-mode").read_text(encoding="utf-8") == str(
            stat.S_IRWXU
        )
        server.close()

    first, second = sockets
    assert first.parent != second.parent
    assert not first.parent.exists()


def test_a_socket_nothing_listens_on_never_counts_as_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(f"{_OWNER}.HEALTH_DEADLINE_SECONDS", 0.5)
    server = provider_server(tmp_path, "stale-socket")
    health_deadline_from_scenario(monkeypatch, _OWNER, tmp_path / "cache")

    with pytest.raises(ProviderFailedError):
        server.load()

    assert server.ready is False
    _, _, pid, _ = observed(tmp_path / "cache")
    server.close()
    assert_reaped(pid)


def test_the_health_deadline_fails_the_load_and_reaps_a_term_resistant_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(f"{_OWNER}.HEALTH_DEADLINE_SECONDS", 0.3)
    monkeypatch.setattr(f"{_OWNER}.TERMINATE_GRACE_SECONDS", 0.05)
    server = provider_server(tmp_path, "stale-socket term-resistant")
    health_deadline_from_scenario(monkeypatch, _OWNER, tmp_path / "cache")

    loading, failures = _load_in_thread(server)
    try:
        loading.join(timeout=StopPolicy().grace_seconds)

        assert not loading.is_alive(), "the load did not fail inside the stop's grace"
        assert len(failures) == 1
        _, _, pid, _ = observed(tmp_path / "cache")
        assert_reaped(pid, StopPolicy().grace_seconds)
    finally:
        server.close()


def test_a_stop_during_the_health_wait_does_not_wait_out_the_health_deadline(
    tmp_path: Path,
) -> None:
    root, cache = closed_provider(tmp_path, "held-ready")
    server = provider_process(root, cache)
    loading, refused = _load_in_thread(server)
    (pid,) = wait_for_starts(cache, 1)

    started = time.monotonic()
    server.close()
    loading.join(timeout=well_before(HEALTH_DEADLINE_SECONDS))

    assert time.monotonic() - started < well_before(HEALTH_DEADLINE_SECONDS)
    assert not loading.is_alive()
    assert refused
    assert_reaped(pid)


def test_an_exit_after_ready_ends_readiness_at_once(tmp_path: Path) -> None:
    server = provider_server(tmp_path, "idle-exit")
    server.load()
    (tmp_path / "cache/exit-now").touch()

    poll(lambda: not server.ready, within=5)

    assert server.ready is False
    with pytest.raises(ProviderNotReadyError):
        server.connect(1)
    server.close()


def test_a_marked_failure_stops_the_server_and_refuses_a_new_load(
    tmp_path: Path,
) -> None:
    server = provider_server(tmp_path)
    server.load()
    _, _, pid, _ = observed(tmp_path / "cache")

    server.mark_failed()

    assert server.ready is False
    with pytest.raises(ProviderFailedError):
        server.load()
    server.close()
    assert_reaped(pid)


@pytest.mark.parametrize("behavior", ["startup-exit", "startup-oversized"])
def test_a_server_that_ends_before_it_answers_fails_without_an_owned_process(
    tmp_path: Path, behavior: str
) -> None:
    server = provider_server(tmp_path, behavior)
    with pytest.raises(RuntimeError, match="speech provider failed"):
        server.load()
    _, _, pid, _ = observed(tmp_path / "cache")
    server.close()
    assert_reaped(pid)


def test_load_after_a_failed_startup_is_refused(tmp_path: Path) -> None:
    server = provider_server(tmp_path, "startup-exit")
    with pytest.raises(ProviderFailedError):
        server.load()
    with pytest.raises(ProviderFailedError):
        server.load()
    server.close()


def test_a_request_before_load_is_refused_as_not_ready(tmp_path: Path) -> None:
    server = provider_server(tmp_path)
    with pytest.raises(ProviderNotReadyError):
        server.connect(1)


def test_a_server_without_a_private_directory_is_refused_and_closes_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fatal = threading.Event()
    server = provider_server(tmp_path)
    server.set_fatal_callback(fatal.set)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path / "no-temporary-root"))
    loader, refused = _load_in_thread(server)
    loader.join(timeout=5)

    assert refused
    server.close()
    assert fatal.is_set() is False


def test_the_constructor_has_no_thread_process_or_filesystem_side_effect(
    tmp_path: Path,
) -> None:
    before = {thread.ident for thread in threading.enumerate()}
    server = ProviderProcess(
        ProviderLaunch(executable=tmp_path / "missing", arguments=())
    )
    server.close()
    assert {thread.ident for thread in threading.enumerate()} == before
    assert list(tmp_path.iterdir()) == []


def test_shutdown_during_the_health_wait_closes_pending_without_publication(
    tmp_path: Path,
) -> None:
    root, cache = closed_provider(tmp_path, "held-ready")
    runtime = _runtime(
        lambda _voice: qwen_voice(root, cache, QwenSpeaker.RYAN, (Language.DE,))
    )

    def start() -> None:
        with contextlib.suppress(RuntimeError):
            runtime.load()

    activation = threading.Thread(target=start)
    activation.start()
    (pid,) = wait_for_starts(cache, 1)

    runtime.begin_shutdown()
    runtime.close()
    activation.join(timeout=5)

    assert not activation.is_alive()
    assert runtime.capture(VoiceId.THORSTEN) is None
    assert_reaped(pid)


def test_a_missing_launcher_leaves_no_retained_engine(tmp_path: Path) -> None:
    fatal = threading.Event()
    runtime = _runtime(
        lambda _voice: QwenSpeaking(
            ProviderProcess(
                ProviderLaunch(executable=tmp_path / "missing", arguments=())
            ),
            QwenSpeaker.RYAN,
            (Language.DE,),
        )
    )
    runtime.set_fatal_callback(fatal.set)
    with pytest.raises(RuntimeError, match="failed to load"):
        runtime.load()
    assert runtime.capture(VoiceId.THORSTEN) is None
    assert not any(
        thread.name == "speech-provider-owner" and thread.is_alive()
        for thread in threading.enumerate()
    )
    assert fatal.is_set() is False


def test_a_real_child_is_reaped_when_the_owner_fails_after_the_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fatal = threading.Event()
    server = provider_server(tmp_path, "term-resistant")
    server.set_fatal_callback(fatal.set)
    real_popen = subprocess.Popen
    started: list[int] = []

    class WithoutStderr:
        stderr = None

        def __init__(self, child: subprocess.Popen[bytes]) -> None:
            self._child = child

        def __getattr__(self, name: str) -> object:
            return getattr(self._child, name)

    def recording_popen(*arguments: object, **keywords: object) -> WithoutStderr:
        child = real_popen(*arguments, **keywords)
        started.append(child.pid)
        return WithoutStderr(child)

    monkeypatch.setattr(f"{_OWNER}.TERMINATE_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(f"{_OWNER}.subprocess.Popen", recording_popen)
    with pytest.raises(RuntimeError, match="speech provider failed"):
        server.load()
    assert fatal.wait(timeout=5)
    server.close()
    assert len(started) == 1
    assert_reaped(started[0])


def test_a_cleanup_failure_reaps_and_reports_a_fatal_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fatal = threading.Event()
    started = fail_every_reap(monkeypatch)
    server = provider_server(tmp_path)
    server.set_fatal_callback(fatal.set)
    server.load()

    with pytest.raises(ProviderCleanupFailedError, match="cleanup failed"):
        server.close()

    assert fatal.wait(timeout=5)
    assert len(started) == 1
    assert_reaped(started[0])


def test_a_term_resistant_server_reaches_kill_and_reap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(f"{_OWNER}.TERMINATE_GRACE_SECONDS", 0.05)
    server = provider_server(tmp_path, "term-resistant")
    server.load()
    _, _, pid, _ = observed(tmp_path / "cache")
    server.close()
    assert_reaped(pid, StopPolicy().grace_seconds)


def test_a_term_resistant_server_is_reaped_inside_the_stops_grace(
    tmp_path: Path,
) -> None:
    server = provider_server(tmp_path, "term-resistant")
    server.load()
    _, _, pid, _ = observed(tmp_path / "cache")
    started = time.monotonic()
    server.close()
    assert time.monotonic() - started < StopPolicy().grace_seconds
    assert_reaped(pid, StopPolicy().grace_seconds)


def test_the_whole_close_bound_is_below_the_stops_grace() -> None:
    assert StopPolicy().grace_seconds > CLOSE_SECONDS + OWNER_JOIN_SECONDS


def test_a_stage_process_that_outlives_its_server_ends_with_the_service_stop(
    tmp_path: Path,
) -> None:
    server = provider_server(tmp_path, "with-stage")
    server.load()
    cache = tmp_path / "cache"
    _, _, pid, _ = observed(cache)
    stage = int((cache / "stage").read_text())

    os.kill(pid, signal.SIGKILL)
    poll(lambda: not server.ready, within=5)
    server.close()

    assert not server.ready
    assert has_ended(stage)


def test_a_server_whose_start_ends_by_exit_is_reaped_before_the_load_is_refused(
    tmp_path: Path,
) -> None:
    root, cache = closed_provider(tmp_path, "held-ready")
    server = provider_process(root, cache)
    loading = threading.Thread(target=_load_quietly, args=(server,))
    loading.start()
    (pid,) = wait_for_starts(cache, 1)

    release_load(cache, pid, LoadOutcome.EXIT)
    loading.join(timeout=5)

    assert not loading.is_alive()
    assert_reaped(pid)
    server.close()


_SERVICE = """
import sys
import threading
from pathlib import Path

from speech_service.provider_process import ProviderLaunch, ProviderProcess

root, cache = Path(sys.argv[1]), Path(sys.argv[2])
server = ProviderProcess(ProviderLaunch(
    executable=root / "qwen/.venv/bin/speech-qwen-server",
    arguments=("--cache", str(cache)),
))
loading = threading.Thread(target=server.load)
loading.start()
loading.join()
print((cache / "server-starts").read_text().split()[0], flush=True)
threading.Event().wait()
"""


def test_the_server_outlives_a_load_thread_and_dies_with_the_service_process(
    tmp_path: Path,
) -> None:
    root, cache = closed_provider(tmp_path)

    with subprocess.Popen(
        [sys.executable, "-c", _SERVICE, str(root), str(cache)],
        stdout=subprocess.PIPE,
        text=True,
    ) as service:
        server_pid = int(service.stdout.readline())
        survived_the_load_thread = not has_ended(server_pid, within=0.5)

        service.kill()
        service.wait()

        assert survived_the_load_thread
        assert has_ended(server_pid)
