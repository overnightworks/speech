"""The parent keeps provider protocol failures private and terminal."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from io import BytesIO
from pathlib import Path

import pytest
from anyio import run, run_process
from speech_provider_contract import (
    QWEN_MODEL_ID,
    QWEN_SAMPLE_RATE,
    Frame,
    FrameDecoder,
    FrameKind,
    ProtocolError,
    read_frame,
    ready_payload,
    write_frame,
)

from speech_service.provider_process import (
    SYNTHESIS_FAILED,
    ProviderFailedError,
    ProviderLaunch,
    ProviderNotReadyError,
    ProviderProcess,
    ProviderProtocolError,
    UnsupportedLanguageError,
)
from speech_service.service import Runtime, RuntimeDependencies
from speech_service.voices import VoiceId
from tests.support.worker_support import (
    assert_reaped,
    closed_provider,
    debug_log,
    fail_every_reap,
    numba_cache,
    observed,
    provider_worker,
    wait_for_starts,
)

_SCRATCH_VARIABLES = ("NUMBA_CACHE_DIR", "TMPDIR")


def _engine(tmp_path: Path, behavior: str = "normal") -> ProviderProcess:
    root, cache = closed_provider(tmp_path, behavior)
    return _provider_process(root, cache)


def _qwen_arguments(cache: Path) -> tuple[str, ...]:
    return ("--device", "cpu", "--cache", str(cache), "--speaker", "Ryan")


def _provider_process(root: Path, cache: Path) -> ProviderProcess:
    return ProviderProcess(
        ProviderLaunch(
            model_name=QWEN_MODEL_ID,
            sample_rate=QWEN_SAMPLE_RATE,
            executable=provider_worker(root),
            arguments=_qwen_arguments(cache),
        )
    )


def test_protocol_round_trips_partial_reads_and_consecutive_frames() -> None:
    encoded = BytesIO()
    write_frame(encoded, Frame(FrameKind.PCM, 1, b"\x00\x01"))
    write_frame(encoded, Frame(FrameKind.COMPLETE, 1, b""))
    decoder = FrameDecoder()
    frames = []
    for byte in encoded.getvalue():
        decoder.feed(bytes([byte]))
        frame = decoder.next_frame()
        if frame is not None:
            frames.append(frame)
    decoder.finish()
    assert frames == [
        Frame(FrameKind.PCM, 1, b"\x00\x01"),
        Frame(FrameKind.COMPLETE, 1, b""),
    ]


@pytest.mark.parametrize("sample_rate", [24_000, 48_000])
def test_version_one_ready_encodes_the_declared_native_rate(sample_rate: int) -> None:
    frame = Frame(FrameKind.READY, 0, ready_payload(sample_rate))

    encoded = BytesIO()
    write_frame(encoded, frame)

    assert read_frame(BytesIO(encoded.getvalue())) == frame
    assert ready_payload(24_000) == b"\x00\x01\x00\x00]\xc0"


def test_parent_rejects_a_worker_that_advertises_another_rate(tmp_path: Path) -> None:
    root, cache = closed_provider(tmp_path, "other-rate")
    engine = _provider_process(root, cache)

    with pytest.raises(RuntimeError, match="speech provider failed"):
        engine.load()

    engine.close()


@pytest.mark.parametrize(
    "encoded",
    [
        struct.pack("!IBQ", 9, 255, 1),
        struct.pack("!IBQ", 9, FrameKind.PCM, 1),
        struct.pack("!IBQ", 10, FrameKind.FAILED, 1) + b"\x02",
        struct.pack("!IBQ", 26, FrameKind.FAILED, 0) + b"\x01builtins.OSError",
        struct.pack("!IBQ", 4107, FrameKind.FAILED, 1) + b"\x01" + b"a" * 4097,
        *(
            struct.pack("!IBQ", 11, FrameKind.FAILED, 1) + b"\x01" + byte
            for byte in (b"\x00", b"\t", b"\r", b"\x7f", b"\x80")
        ),
        struct.pack("!IBQ", 15, FrameKind.READY, 0) + b"\x00\x02\x00\x00]\xc0",
        struct.pack("!IBQ", 10, FrameKind.SYNTHESIZE, 1) + b"\x03",
        struct.pack("!IBQ", 12, FrameKind.PCM, 1) + b"\x00\x01\x02",
    ],
)
def test_protocol_rejects_illegal_lengths_kinds_and_payloads(encoded: bytes) -> None:
    stream = BytesIO(encoded)
    with pytest.raises(ProtocolError):
        read_frame(stream)


def test_protocol_reports_a_payload_cut_short_after_its_header() -> None:
    stream = BytesIO(struct.pack("!IBQ", 12, FrameKind.PCM, 1) + b"\x00")
    with pytest.raises(EOFError):
        read_frame(stream)


@pytest.mark.parametrize(
    "payload", [b"\x01", b"\x01builtins.OSError", b"\x01" + b"a\n" * 2048]
)
def test_protocol_accepts_a_request_failure_with_or_without_its_cause(
    payload: bytes,
) -> None:
    frame = Frame(FrameKind.FAILED, 1, payload)

    encoded = BytesIO()
    write_frame(encoded, frame)

    assert read_frame(BytesIO(encoded.getvalue())) == frame


@pytest.mark.parametrize("payload", [b"", b"\x03text", b"\x01\xff"])
def test_protocol_refuses_invalid_synthesis_payloads(payload: bytes) -> None:
    stream = BytesIO()
    frame = Frame(FrameKind.SYNTHESIZE, 1, payload)
    with pytest.raises(ProtocolError):
        write_frame(stream, frame)


@pytest.fixture(params=["python3.12", "python3.14"])
def python_dir(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.mark.parametrize("behavior", ["normal", "startup-exit"])
def test_each_worker_keeps_its_temporary_state_in_a_private_directory_gone_once_reaped(
    tmp_path: Path, behavior: str
) -> None:
    caches = []
    engines = []
    for worker in ("first", "second"):
        root, cache = closed_provider(tmp_path / worker, behavior)
        engines.append(_provider_process(root, cache))
        caches.append(cache)
        with contextlib.suppress(ProviderFailedError):
            engines[-1].load()
    for engine in engines:
        engine.close()

    (first, first_mode), (second, second_mode) = map(numba_cache, caches)
    assert first != second
    assert first_mode == second_mode == stat.S_IRWXU
    assert not first.exists()
    assert not second.exists()
    assert not any(debug_log(cache).exists() for cache in caches)


def test_a_worker_without_a_private_directory_is_refused_and_closes_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fatal = threading.Event()
    engine = _engine(tmp_path)
    engine.set_fatal_callback(fatal.set)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "no-temporary-root"))
    refused: list[ProviderFailedError] = []

    def load() -> None:
        try:
            engine.load()
        except ProviderFailedError as error:
            refused.append(error)

    loader = threading.Thread(target=load, daemon=True)
    loader.start()
    loader.join(timeout=5)

    assert refused
    engine.close()
    assert fatal.is_set() is False


def test_direct_closed_entrypoint_streams_after_transient_loader_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, python_dir: str
) -> None:
    root, cache = closed_provider(tmp_path)
    cublas = root / f"qwen/.venv/lib/{python_dir}/site-packages/nvidia/cublas/lib"
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

    monkeypatch.setattr(
        "speech_service.provider_process.subprocess.Popen", recording_popen
    )
    engine = _provider_process(root, cache)
    loaded = threading.Thread(target=engine.load)
    loaded.start()
    loaded.join(timeout=5)
    assert not loaded.is_alive()
    response: list[bytes] = []
    request = threading.Thread(
        target=lambda: response.extend(engine.pcm_chunks("Hallo", "de-DE"))
    )
    request.start()
    request.join(timeout=5)
    assert not request.is_alive()
    assert response == [b"\x00\x01"]
    assert any(
        thread.name == "speech-provider-owner" for thread in threading.enumerate()
    )
    cwd, environment, pid, arguments = observed(cache)
    assert list(engine.pcm_chunks("later", "en")) == [b"\x00\x01"]
    engine.close()
    assert cwd == str(root / "qwen")
    assert arguments == {
        "expected_parent_pid": str(os.getpid()),
        "device": "cpu",
        "cache": str(cache),
        "speaker": "Ryan",
    }
    expected_environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PATH": os.defpath,
        "LD_LIBRARY_PATH": str(cublas),
        "TORCH_DISABLE_NATIVE_JIT": "1",
        **dict.fromkeys(_SCRATCH_VARIABLES, environment["TMPDIR"]),
    }
    assert set(environment) <= {*expected_environment, "LC_CTYPE"}
    assert all(environment[key] == value for key, value in expected_environment.items())
    [(launch_arguments, launch)] = launches
    assert launch_arguments == [
        str(provider_worker(root)),
        "--expected-parent-pid",
        str(os.getpid()),
        *_qwen_arguments(cache),
    ]
    assert isinstance(launch.pop("stderr"), io.IOBase)
    assert launch == {
        "shell": False,
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "cwd": root / "qwen",
        "env": expected_environment,
        "bufsize": 0,
        "close_fds": True,
        "start_new_session": True,
    }
    assert_reaped(pid)


@pytest.mark.parametrize("first", ["wait-cancel", "normal"])
def test_cancel_drain_keeps_the_worker_reusable_and_late_cancel_isolated(
    tmp_path: Path, first: str
) -> None:
    engine = _engine(tmp_path)
    engine.load()
    chunks = engine.pcm_chunks(first, "de")
    assert next(chunks) == b"\x00\x01"
    chunks.close()

    assert list(engine.pcm_chunks("later", "en")) == [b"\x00\x01"]
    assert (tmp_path / "cache/request-ids").read_text(encoding="utf-8").split() == [
        "1",
        "2",
    ]
    engine.close()


@pytest.mark.parametrize(
    "text",
    ["stale", "wrong-direction", "unexpected-cancelled", "midstream-exit"],
)
def test_stale_wrong_direction_out_of_order_and_midstream_exit_fail_closed(
    tmp_path: Path, text: str
) -> None:
    engine = _engine(tmp_path)
    engine.load()
    with pytest.raises(ProviderProtocolError):
        list(engine.pcm_chunks(text, "de"))
    engine.close()


def test_request_failure_is_sanitized_and_marks_the_adapter_unready(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine = _engine(tmp_path)
    engine.load()
    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(RuntimeError, match=SYNTHESIS_FAILED),
    ):
        list(engine.pcm_chunks("failed", "de"))
    assert engine.ready is False
    engine.close()

    assert f"{SYNTHESIS_FAILED}, request 1, cause: none" in caplog.text


@pytest.mark.parametrize(
    "behavior", ["startup-exit", "startup-oversized", "startup-failed"]
)
def test_early_exit_and_oversized_handshake_fail_without_an_owned_process(
    tmp_path: Path, behavior: str
) -> None:
    engine = _engine(tmp_path, behavior)
    with pytest.raises(RuntimeError, match="speech provider failed"):
        engine.load()
    _, _, pid, _ = observed(tmp_path / "cache")
    engine.close()
    assert_reaped(pid)


def test_load_after_a_failed_startup_is_refused(tmp_path: Path) -> None:
    engine = _engine(tmp_path, "startup-exit")
    with pytest.raises(ProviderFailedError):
        engine.load()
    with pytest.raises(ProviderFailedError):
        engine.load()
    engine.close()


def test_unsupported_language_is_refused_before_any_request(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.load()
    with pytest.raises(UnsupportedLanguageError):
        list(engine.pcm_chunks("hello", "fr"))
    assert engine.ready is True
    engine.close()


def test_request_before_load_is_refused_as_not_ready(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with pytest.raises(ProviderNotReadyError):
        list(engine.pcm_chunks("hello", "de"))


def test_duplicate_terminal_is_rejected_before_a_later_request(tmp_path: Path) -> None:
    engine = _engine(tmp_path, "duplicate-terminal")
    engine.load()
    assert list(engine.pcm_chunks("first", "de")) == [b"\x00\x01"]
    with pytest.raises(ProviderProtocolError):
        list(engine.pcm_chunks("second", "de"))
    engine.close()


def test_partial_cancel_frame_cannot_extend_the_single_drain_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("speech_service.provider_process.CANCEL_DRAIN_SECONDS", 0.05)
    monkeypatch.setattr("speech_service.provider_process.TERMINATE_GRACE_SECONDS", 0.05)
    engine = _engine(tmp_path, "partial-cancel")
    engine.load()
    chunks = engine.pcm_chunks("cancel", "de")
    assert next(chunks) == b"\x00\x01"
    started = time.monotonic()
    chunks.close()
    assert time.monotonic() - started < 1
    _, _, pid, _ = observed(tmp_path / "cache")
    engine.close()
    assert engine.ready is False
    assert_reaped(pid)


def test_qwen_cancel_beyond_drain_reaps_then_runtime_loads_a_fresh_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("speech_service.provider_process.CANCEL_DRAIN_SECONDS", 0.05)
    monkeypatch.setattr("speech_service.provider_process.TERMINATE_GRACE_SECONDS", 0.05)
    root, cache = closed_provider(tmp_path, "partial-cancel")
    constructed: list[ProviderProcess] = []

    def factory(_voice: VoiceId) -> ProviderProcess:
        engine = _provider_process(root, cache)
        constructed.append(engine)
        return engine

    runtime = Runtime(
        None,
        dependencies=RuntimeDependencies(
            engine_factory=factory,
            missing_artifacts=lambda _voice: (),
        ),
    )
    runtime.load()
    first = constructed[0]
    chunks = first.pcm_chunks("cancel", "de")
    assert next(chunks) == b"\x00\x01"
    chunks.close()
    _, _, first_pid, _ = observed(cache)

    runtime.health()

    assert runtime.capture(VoiceId.PIPER) is None
    assert_reaped(first_pid)
    runtime.close()


def test_constructor_has_no_thread_process_or_filesystem_side_effect(
    tmp_path: Path,
) -> None:
    before = {thread.ident for thread in threading.enumerate()}
    engine = ProviderProcess(
        ProviderLaunch(
            model_name=QWEN_MODEL_ID,
            sample_rate=QWEN_SAMPLE_RATE,
            executable=tmp_path / "missing",
            arguments=("--device", "cpu", "--cache", str(tmp_path / "cache")),
        )
    )
    engine.close()
    assert {thread.ident for thread in threading.enumerate()} == before
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("behavior", "text"), [("normal", "wait-cancel"), ("partial-cancel", "cancel")]
)
def test_close_allows_bounded_active_cancel_before_reaping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    text: str,
) -> None:
    monkeypatch.setattr("speech_service.provider_process.CANCEL_DRAIN_SECONDS", 0.2)
    monkeypatch.setattr("speech_service.provider_process.TERMINATE_GRACE_SECONDS", 0.05)
    engine = _engine(tmp_path, behavior)
    engine.load()
    chunks = engine.pcm_chunks(text, "de")
    assert next(chunks) == b"\x00\x01"
    closed: list[bool] = []

    def close_engine() -> None:
        engine.close()
        closed.append(True)

    closing = threading.Thread(target=close_engine)
    closing.start()
    threading.Event().wait(0.05)

    started = time.monotonic()
    chunks.close()
    closing.join(timeout=5)

    assert not closing.is_alive()
    assert closed == [True]
    assert time.monotonic() - started < 1
    assert (tmp_path / "cache/cancel-observed").exists()
    _, _, pid, _ = observed(tmp_path / "cache")
    assert_reaped(pid)


def test_shutdown_during_handshake_closes_pending_without_publication(
    tmp_path: Path,
) -> None:
    root, cache = closed_provider(tmp_path, "held-ready")
    runtime = Runtime(
        None,
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: _provider_process(root, cache),
            missing_artifacts=lambda _voice: (),
        ),
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
    assert runtime.capture(VoiceId.PIPER) is None
    assert_reaped(pid)


def test_missing_entrypoint_failed_exec_leaves_no_retained_engine(
    tmp_path: Path,
) -> None:
    fatal = threading.Event()
    runtime = Runtime(
        None,
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: ProviderProcess(
                ProviderLaunch(
                    model_name=QWEN_MODEL_ID,
                    sample_rate=QWEN_SAMPLE_RATE,
                    executable=tmp_path / "missing",
                    arguments=("--device", "cpu", "--cache", str(tmp_path)),
                )
            ),
            missing_artifacts=lambda _voice: (),
        ),
    )
    runtime.set_fatal_callback(fatal.set)
    with pytest.raises(RuntimeError, match="failed to load"):
        runtime.load()
    assert runtime.capture(VoiceId.PIPER) is None
    assert not any(
        thread.name == "speech-provider-owner" and thread.is_alive()
        for thread in threading.enumerate()
    )
    assert fatal.is_set() is False


def test_real_child_is_reaped_when_publication_setup_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fatal = threading.Event()
    engine = _engine(tmp_path, "term-resistant")
    engine.set_fatal_callback(fatal.set)
    real_popen = subprocess.Popen
    started: list[int] = []

    def recording_popen(*arguments: object, **keywords: object) -> subprocess.Popen:
        child = real_popen(*arguments, **keywords)
        started.append(child.pid)
        return child

    class BrokenDecoder:
        def __init__(self) -> None:
            raise RuntimeError

    monkeypatch.setattr("speech_service.provider_process.FrameDecoder", BrokenDecoder)
    monkeypatch.setattr(
        "speech_service.provider_process.subprocess.Popen", recording_popen
    )
    with pytest.raises(RuntimeError, match="speech provider failed"):
        engine.load()
    assert fatal.wait(timeout=5)
    engine.close()
    assert len(started) == 1
    assert_reaped(started[0])


def test_cleanup_failure_reaps_and_reports_fatal_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fatal = threading.Event()
    started = fail_every_reap(monkeypatch)
    engine = _engine(tmp_path)
    engine.set_fatal_callback(fatal.set)
    engine.load()

    with pytest.raises(RuntimeError, match="speech provider cleanup failed"):
        engine.close()

    assert fatal.wait(timeout=5)
    assert len(started) == 1
    assert_reaped(started[0])


def test_term_resistant_leader_reaches_kill_and_reap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("speech_service.provider_process.TERMINATE_GRACE_SECONDS", 0.05)
    engine = _engine(tmp_path, "term-resistant")
    engine.load()
    _, _, pid, _ = observed(tmp_path / "cache")
    engine.close()
    assert_reaped(pid)


def test_parent_death_signal_follows_the_real_creator_thread(tmp_path: Path) -> None:
    provider_source = Path(__file__).parents[1] / "providers/qwen/src"
    contract_source = Path(__file__).parents[1] / "providers/provider-contract/src"
    helper = r'''
import contextlib
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ready = Path(sys.argv[1])
child_code = """
import os, sys, time
from pathlib import Path
from speech_qwen.__main__ import install_parent_death_signal
install_parent_death_signal(int(sys.argv[1]))
Path(sys.argv[2]).write_text(str(os.getpid()))
while True: time.sleep(1)
"""
owned = []
def create():
    child = subprocess.Popen([
        sys.executable, "-c", child_code, str(os.getpid()), str(ready)
    ])
    owned.append(child)
    for _ in range(500):
        if ready.exists():
            return
        threading.Event().wait(0.01)
    raise RuntimeError("child never installed PDEATHSIG")
creator = threading.Thread(target=create)
creator.start()
creator.join()
child = owned[0]
print("responsive", flush=True)
raise SystemExit(0 if child.wait(timeout=5) == -9 else 1)
'''
    environment = {
        "PYTHONPATH": os.pathsep.join((str(provider_source), str(contract_source)))
    }

    async def execute_helper() -> tuple[int, bytes]:
        completed = await run_process(
            [sys.executable, "-c", helper, str(tmp_path / "ready")],
            env=environment,
            check=False,
        )
        return completed.returncode, completed.stdout

    returncode, output = run(execute_helper)
    assert returncode == 0
    assert output == b"responsive\n"
