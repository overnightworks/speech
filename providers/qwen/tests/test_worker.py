"""The provider emits only ordered frames and a failure that carries no payload."""

from __future__ import annotations

import builtins
import multiprocessing
import os
import queue
import struct
import sys
import threading
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import speech_provider_contract as worker
from speech_provider_contract import (
    MAX_FAILURE_CAUSE_BYTES,
    Frame,
    FrameKind,
    ProviderFailure,
    ProviderFunctions,
    failure_cause,
    read_frame,
    write_frame,
)

from speech_qwen import __main__ as bootstrap

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def _encoded(*frames: Frame) -> bytes:
    stream = BytesIO()
    for frame in frames:
        write_frame(stream, frame)
    return stream.getvalue()


def _frames(encoded: bytes) -> list[Frame]:
    stream = BytesIO(encoded)
    frames = []
    while stream.tell() < len(encoded):
        frames.append(read_frame(stream))
    return frames


def _assert_failed_with_cause(
    frame: Frame, request_id: int, type_name: str, detail: str
) -> None:
    assert (frame.kind, frame.request_id) == (FrameKind.FAILED, request_id)
    cause = failure_cause(frame.payload)
    assert type_name in cause
    assert detail not in cause


def _run_worker(
    monkeypatch: pytest.MonkeyPatch,
    input_bytes: bytes,
    loader: Callable[[str, Path], object],
    synthesizer: Callable[..., Iterator[bytes]],
) -> tuple[int, list[Frame], list[tuple[str, Path]]]:
    observed: list[tuple[str, Path]] = []

    def recording_loader(device: str, cache: Path) -> object:
        observed.append((device, cache))
        return loader(device, cache)

    monkeypatch.setattr(
        worker.sys, "stdin", SimpleNamespace(buffer=BytesIO(input_bytes))
    )
    read_fd, write_fd = os.pipe()
    result = worker.serve_provider(
        device="cpu",
        cache="/closed/cache",
        protocol_stdout=write_fd,
        sample_rate=24_000,
        functions=ProviderFunctions(
            recording_loader,
            synthesizer,
            lambda model: [*synthesizer(model, "Hallo.", "de")],
        ),
    )
    with os.fdopen(read_fd, "rb") as output:
        frames = _frames(output.read())
    return result, frames, observed


def test_run_loads_warms_then_maps_active_input_eof_to_sanitized_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def synthesize(_model: object, text: str, language: str) -> Iterator[bytes]:
        calls.append((text, language))
        if text != "Hallo.":
            yield b"\x00\x01"

    result, frames, observed = _run_worker(
        monkeypatch,
        _encoded(Frame(FrameKind.SYNTHESIZE, 1, b"\x02Hello")),
        lambda _device, _cache: object(),
        synthesize,
    )

    assert result == 1
    assert observed == [("cpu", Path("/closed/cache"))]
    assert calls == [("Hallo.", "de"), ("Hello", "en")]
    ready, failed = frames
    assert ready == Frame(FrameKind.READY, 0, b"\x00\x01\x00\x00]\xc0")
    _assert_failed_with_cause(failed, 1, "builtins.EOFError", "Hello")
    assert not any(
        thread.name == "speech-provider-input" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_run_joins_a_cooperative_reader_that_is_still_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class HeldQueue(queue.Queue):
        def put(self, item: object, *args: object, **kwargs: object) -> None:
            super().put(item, *args, **kwargs)
            if isinstance(item, BaseException):
                queued.set()
                assert release.wait(timeout=5)

    monkeypatch.setattr(worker.queue, "Queue", HeldQueue)
    monkeypatch.setattr(worker.sys, "stdin", SimpleNamespace(buffer=BytesIO()))
    read_fd, write_fd = os.pipe()
    outcomes: list[int] = []

    def run_worker() -> None:
        outcomes.append(
            worker.serve_provider(
                device="cpu",
                cache="/closed/cache",
                protocol_stdout=write_fd,
                sample_rate=24_000,
                functions=ProviderFunctions(lambda *_args: object(), lambda *_args: ()),
            )
        )
        finished.set()

    running = threading.Thread(target=run_worker)
    running.start()
    assert queued.wait(timeout=5)
    returned_before_reader = finished.wait(timeout=0.1)
    release.set()
    running.join(timeout=5)
    with os.fdopen(read_fd, "rb") as output:
        _frames(output.read())

    assert returned_before_reader is False
    assert not running.is_alive()
    assert outcomes == [1]
    assert not any(
        thread.name == "speech-provider-input" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_startup_failure_emits_only_the_sanitized_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_detail = "provider secret detail"

    def fail(_device: str, _cache: Path) -> object:
        raise RuntimeError(provider_detail)

    result, frames, _ = _run_worker(monkeypatch, b"", fail, lambda *_args: ())
    assert result == 1
    assert frames == [Frame(FrameKind.FAILED, 0, b"\x01")]
    assert provider_detail.encode() not in _encoded(*frames)


def test_malformed_input_emits_one_sanitized_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = struct.pack("!I", 8)
    result, frames, _ = _run_worker(
        monkeypatch,
        malformed,
        lambda _device, _cache: object(),
        lambda *_args: (),
    )
    assert result == 1
    assert frames == [
        Frame(FrameKind.READY, 0, b"\x00\x01\x00\x00]\xc0"),
        Frame(FrameKind.FAILED, 0, b"\x01"),
    ]


def test_cancellation_after_an_inflight_step_and_late_cancel_do_not_cross_ids() -> None:
    commands: queue.Queue[Frame | BaseException] = queue.Queue()
    output = BytesIO()
    entered = threading.Event()
    release = threading.Event()

    def synthesize(_model: object, text: str, _language: str) -> Iterator[bytes]:
        if text == "first":
            entered.set()
            assert release.wait(timeout=5)
        yield b"\x00\x01"

    commands.put(Frame(FrameKind.SYNTHESIZE, 1, b"\x01first"))
    serving = threading.Thread(
        target=worker.serve_commands, args=(commands, output, synthesize, object())
    )
    serving.start()
    assert entered.wait(timeout=5)
    commands.put(Frame(FrameKind.CANCEL, 1, b""))
    release.set()
    _wait_for_frames(output, 1)
    commands.put(Frame(FrameKind.CANCEL, 1, b""))
    commands.put(Frame(FrameKind.SYNTHESIZE, 2, b"\x02second"))
    _wait_for_frames(output, 3)
    commands.put(EOFError())
    serving.join(timeout=5)

    assert not serving.is_alive()
    assert _frames(output.getvalue()) == [
        Frame(FrameKind.CANCELLED, 1, b""),
        Frame(FrameKind.PCM, 2, b"\x00\x01"),
        Frame(FrameKind.COMPLETE, 2, b""),
        Frame(FrameKind.FAILED, 0, b"\x01"),
    ]


def test_cancel_closes_the_active_iterator_before_the_next_request() -> None:
    commands: queue.Queue[Frame | BaseException] = queue.Queue()
    output = BytesIO()
    first_chunk = threading.Event()
    release_first_chunk = threading.Event()
    closed = threading.Event()

    class HeldIterator:
        def __init__(self) -> None:
            self._emitted = False

        def __iter__(self) -> HeldIterator:
            return self

        def __next__(self) -> bytes:
            if self._emitted:
                raise StopIteration
            self._emitted = True
            first_chunk.set()
            assert release_first_chunk.wait(timeout=5)
            return b"\x00\x01"

        def close(self) -> None:
            closed.set()

    def synthesize(_model: object, text: str, _language: str) -> Iterator[bytes]:
        if text == "first":
            return HeldIterator()
        assert closed.is_set()
        return iter((b"\x00\x01",))

    commands.put(Frame(FrameKind.SYNTHESIZE, 1, b"\x01first"))
    serving = threading.Thread(
        target=worker.serve_commands, args=(commands, output, synthesize, object())
    )
    serving.start()
    try:
        assert first_chunk.wait(timeout=5)
        commands.put(Frame(FrameKind.CANCEL, 1, b""))
        release_first_chunk.set()
        assert closed.wait(timeout=5)
        commands.put(Frame(FrameKind.SYNTHESIZE, 2, b"\x01second"))
        _wait_for_frames(output, 3)
    finally:
        commands.put(EOFError())
        serving.join(timeout=5)

    assert not serving.is_alive()
    assert closed.is_set()
    assert _frames(output.getvalue()) == [
        Frame(FrameKind.CANCELLED, 1, b""),
        Frame(FrameKind.PCM, 2, b"\x00\x01"),
        Frame(FrameKind.COMPLETE, 2, b""),
        Frame(FrameKind.FAILED, 0, b"\x01"),
    ]


@pytest.mark.parametrize(
    ("commands", "failure_id"),
    [
        ([Frame(FrameKind.PCM, 1, b"\x00\x01")], 0),
        ([Frame(FrameKind.READY, 0, b"\x00\x01\x00\x00]\xc0")], 0),
        ([Frame(FrameKind.CANCEL, 7, b"")], 0),
        (
            [
                Frame(FrameKind.SYNTHESIZE, 2, b"\x01one"),
                Frame(FrameKind.SYNTHESIZE, 2, b"\x01stale"),
            ],
            2,
        ),
    ],
)
def test_wrong_direction_out_of_order_and_stale_commands_fail_once(
    commands: list[Frame], failure_id: int
) -> None:
    command_queue: queue.Queue[Frame | BaseException] = queue.Queue()
    output = BytesIO()
    for command in commands:
        command_queue.put(command)
    command_queue.put(EOFError())
    worker.serve_commands(command_queue, output, lambda *_args: (), object())
    frames = _frames(output.getvalue())
    if failure_id:
        _assert_failed_with_cause(frames[-1], failure_id, "ProtocolError", "stale")
    else:
        assert frames[-1] == Frame(FrameKind.FAILED, 0, b"\x01")
    assert sum(frame.kind is FrameKind.FAILED for frame in frames) == 1


def _wait_for_frames(output: BytesIO, expected: int) -> None:
    deadline = threading.Event()
    for _ in range(500):
        if len(_frames(output.getvalue())) >= expected:
            return
        deadline.wait(0.01)
    pytest.fail("provider did not produce the expected frames")


def test_model_and_pcm_validation_failure_emit_only_active_failure() -> None:
    detail = "raw model detail"
    for synthesizer, type_name in (
        (
            lambda *_args: (_ for _ in ()).throw(RuntimeError(detail)),
            "builtins.RuntimeError",
        ),
        (lambda *_args: iter((b"\x00",)), "ProtocolError"),
    ):
        commands: queue.Queue[Frame | BaseException] = queue.Queue()
        output = BytesIO()
        commands.put(Frame(FrameKind.SYNTHESIZE, 9, b"\x01Hallo"))
        worker.serve_commands(commands, output, synthesizer, object())
        (frame,) = _frames(output.getvalue())
        _assert_failed_with_cause(frame, 9, type_name, detail)
        assert detail.encode() not in output.getvalue()


@pytest.mark.parametrize(
    ("failure", "terminal"),
    [
        ("normal", FrameKind.COMPLETE),
        ("synthesis", FrameKind.FAILED),
        ("frame", FrameKind.FAILED),
    ],
)
def test_every_request_exit_closes_its_iterator_before_terminal_output(
    monkeypatch: pytest.MonkeyPatch, failure: str, terminal: FrameKind
) -> None:
    events: list[FrameKind | str] = []

    class ClosableIterator:
        def __iter__(self) -> ClosableIterator:
            return self

        def __next__(self) -> bytes:
            if failure == "synthesis":
                raise RuntimeError
            if failure == "frame":
                return b"\x00"
            raise StopIteration

        def close(self) -> None:
            events.append("closed")

    original_write_frame = worker.write_frame

    def record_frame(stream: object, frame: Frame) -> None:
        events.append(frame.kind)
        original_write_frame(stream, frame)

    monkeypatch.setattr(worker, "write_frame", record_frame)
    commands: queue.Queue[Frame | BaseException] = queue.Queue()
    output = BytesIO()
    commands.put(Frame(FrameKind.SYNTHESIZE, 9, b"\x01Hallo"))
    if failure == "normal":
        serving = threading.Thread(
            target=worker.serve_commands,
            args=(commands, output, lambda *_args: ClosableIterator(), object()),
        )
        serving.start()
        try:
            _wait_for_frames(output, 1)
        finally:
            commands.put(EOFError())
            serving.join(timeout=5)
        assert not serving.is_alive()
    else:
        worker.serve_commands(
            commands, output, lambda *_args: ClosableIterator(), object()
        )
    assert events.index("closed") < events.index(terminal)
    if failure != "normal":
        (frame,) = _frames(output.getvalue())
        _assert_failed_with_cause(
            frame,
            9,
            "builtins.RuntimeError" if failure == "synthesis" else "ProtocolError",
            "Hallo",
        )


def test_active_protocol_failure_closes_the_iterator_before_sanitized_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[FrameKind | str] = []

    class ClosableIterator:
        def __init__(self) -> None:
            self._emitted = False

        def __iter__(self) -> ClosableIterator:
            return self

        def __next__(self) -> bytes:
            if self._emitted:
                raise StopIteration
            self._emitted = True
            return b"\x00\x01"

        def close(self) -> None:
            events.append("closed")

    original_write_frame = worker.write_frame

    def record_frame(stream: object, frame: Frame) -> None:
        events.append(frame.kind)
        original_write_frame(stream, frame)

    monkeypatch.setattr(worker, "write_frame", record_frame)
    commands: queue.Queue[Frame | BaseException] = queue.Queue()
    output = BytesIO()
    commands.put(Frame(FrameKind.SYNTHESIZE, 9, b"\x01Hallo"))
    commands.put(Frame(FrameKind.PCM, 9, b"\x00\x01"))

    worker.serve_commands(commands, output, lambda *_args: ClosableIterator(), object())
    assert events == ["closed", FrameKind.FAILED]
    (frame,) = _frames(output.getvalue())
    _assert_failed_with_cause(frame, 9, "ProtocolError", "Hallo")


def test_iterator_cleanup_failure_emits_only_the_active_failure() -> None:
    class FailingCloseIterator:
        def __iter__(self) -> FailingCloseIterator:
            return self

        def __next__(self) -> bytes:
            raise StopIteration

        def close(self) -> None:
            message = "cleanup detail"
            raise RuntimeError(message)

    commands: queue.Queue[Frame | BaseException] = queue.Queue()
    output = BytesIO()
    commands.put(Frame(FrameKind.SYNTHESIZE, 9, b"\x01Hallo"))

    worker.serve_commands(
        commands, output, lambda *_args: FailingCloseIterator(), object()
    )
    (frame,) = _frames(output.getvalue())
    _assert_failed_with_cause(frame, 9, "builtins.RuntimeError", "cleanup detail")
    assert b"cleanup detail" not in output.getvalue()


def _raise_at_depth(depth: int, error: Exception, cause: Exception | None) -> None:
    if depth:
        _raise_at_depth(depth - 1, error, cause)
    raise error from cause


def _deep_chain(kinds: list[type[Exception]], depth: int) -> Exception:
    cause: Exception | None = None
    for kind in kinds:
        try:
            _raise_at_depth(depth, kind(), cause)
        except Exception as raised:  # noqa: BLE001 - each kind becomes the next link
            cause = raised
    assert cause is not None
    return cause


def _failed_frame_for(error: Exception) -> Frame:
    def synthesize(*_args: object) -> Iterator[bytes]:
        raise error

    commands: queue.Queue[Frame | BaseException] = queue.Queue()
    output = BytesIO()
    commands.put(Frame(FrameKind.SYNTHESIZE, 9, b"\x01Hallo"))
    worker.serve_commands(commands, output, synthesize, object())
    (frame,) = _frames(output.getvalue())
    return frame


def test_a_long_deep_chain_keeps_the_links_nearest_its_root_within_the_bound() -> None:
    long_name = type("Long" * 250, (Exception,), {})
    root_first = [KeyError, ValueError, long_name, TypeError, LookupError, OSError]

    frame = _failed_frame_for(_deep_chain(root_first, 300))

    cause = failure_cause(frame.payload)
    lines = cause.split("\n")
    assert len(cause) <= MAX_FAILURE_CAUSE_BYTES
    assert lines[0].startswith("builtins.KeyError -> builtins.ValueError -> ")
    assert lines[0].endswith(" -> builtins.TypeError (+2 outer)")
    assert "LookupError" not in lines[0]
    first_line_label_bytes, other_line_bytes = 128, 256
    assert len(lines[0].split(" -> ")[2]) == first_line_label_bytes
    assert all(len(line) <= other_line_bytes for line in lines[1:])
    assert lines[1] == "builtins.KeyError"
    assert lines[12].endswith(", in _raise_at_depth")
    assert lines[13] == "    raise error from cause"
    assert lines[14] == "builtins.ValueError"


def test_a_failing_cause_encoder_still_sends_a_bare_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreadable(_error: BaseException) -> object:
        raise RuntimeError

    monkeypatch.setattr(worker, "failure_links", unreadable)

    assert _failed_frame_for(RuntimeError()) == Frame(
        FrameKind.FAILED, 9, bytes([ProviderFailure.PROVIDER_FAILURE])
    )


def _run_bootstrap(write_fd: int) -> None:
    os.dup2(write_fd, 1)
    os.close(write_fd)
    sys.argv = [
        "worker",
        "--expected-parent-pid",
        str(os.getppid()),
        "--device",
        "cuda",
        "--cache",
        "/closed/cache",
        "--speaker",
        "Ryan",
    ]
    installed = False
    install = bootstrap.install_parent_death_signal

    def recording_install(expected_parent_pid: int) -> None:
        nonlocal installed
        install(expected_parent_pid)
        installed = True

    bootstrap.install_parent_death_signal = recording_install
    real_import = builtins.__import__

    def importing(name: str, *args: object, **kwargs: object) -> object:
        if name == "speech_qwen.model":
            assert installed
            vars(builtins)["print"]("import print noise", flush=True)
            os.write(1, b"import native noise\n")

            return SimpleNamespace(
                load_model=lambda *_args: object(), pcm_chunks=lambda *_args: ()
            )
        return real_import(name, *args, **kwargs)

    def fake_serve_provider(**arguments: object) -> int:
        protocol = int(arguments["protocol_stdout"])
        os.write(protocol, b"protocol-only")
        os.close(protocol)
        return 0

    bootstrap.serve_provider = fake_serve_provider
    bootstrap.ProviderFunctions = lambda *functions: functions
    builtins.__import__ = importing
    raise SystemExit(bootstrap.main())


def test_bootstrap_redirects_python_and_fd_one_before_provider_import() -> None:
    read_fd, write_fd = os.pipe()
    process = multiprocessing.get_context("fork").Process(
        target=_run_bootstrap, args=(write_fd,)
    )
    process.start()
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as output:
        assert output.read() == b"protocol-only"
    process.join(timeout=5)
    assert process.exitcode == 0
