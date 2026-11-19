"""A server that fails to start leaves its cause in the log, without the sentence."""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import tomllib
from pathlib import Path

import pytest
from speech_provider_contract import QwenSpeaker

from speech_service.provider_process import (
    ProviderFailedError,
    ProviderId,
    ProviderLaunch,
    ProviderProcess,
    ProviderStartFailure,
    ProviderSynthesisFailedError,
    provider_entrypoint,
)
from speech_service.speaking import QwenSpeaking
from tests.support.launcher_support import (
    closed_provider,
    provider_launcher,
    wait_for_starts,
)

_SENTENCE = "Ein Satz, den nur der Aufrufer kennt"


def _launching(tmp_path: Path, body: str) -> ProviderProcess:
    """A server whose launcher runs `body` instead of the fake."""
    root, cache = closed_provider(tmp_path)
    entrypoint = provider_launcher(root)
    entrypoint.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    return ProviderProcess(
        ProviderLaunch(executable=entrypoint, arguments=("--cache", str(cache)))
    )


def _serving(tmp_path: Path) -> QwenSpeaking:
    root, cache = closed_provider(tmp_path)
    return QwenSpeaking(
        ProviderProcess(
            ProviderLaunch(
                executable=provider_launcher(root), arguments=("--cache", str(cache))
            )
        ),
        QwenSpeaker.RYAN,
    )


def _ended_records(
    caplog: pytest.LogCaptureFixture, phase: str | None = None
) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.getMessage() == "provider.exited" and phase in (None, record.phase)
    ]


def _failure_text(error: BaseException) -> str:
    """The text a stopping layer's `exc_info` renders beyond the message."""
    return "\n".join(getattr(error, "__notes__", []))


def _logged_everywhere(caplog: pytest.LogCaptureFixture) -> str:
    """Every message, rendered exception and extra field the records carry."""
    return "\n".join(
        f"{record.getMessage()} {record.exc_text or ''} {vars(record)}"
        for record in caplog.records
    )


def _failing_start_after_writing(stderr_bytes: int) -> str:
    return (
        "import sys\n"
        "sys.stderr.write('ROOT CAUSE: engine core failed\\n')\n"
        f"sys.stderr.write('Traceback line\\n' * ({stderr_bytes} // 15))\n"
        "sys.stderr.flush()\n"
        "raise SystemExit(1)"
    )


def test_the_launcher_name_is_a_script_the_qwen_project_declares() -> None:
    declared = tomllib.loads(
        (Path(__file__).parents[1] / "providers/qwen/pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]["scripts"]

    entrypoint = provider_entrypoint(Path("providers"), ProviderId.QWEN)

    assert entrypoint is not None
    assert entrypoint.name in declared


def test_a_launcher_failing_to_import_logs_its_exit_status_and_notes_the_module(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server = _launching(tmp_path, "import module_that_is_not_installed")

    with caplog.at_level(logging.WARNING), pytest.raises(ProviderFailedError) as failed:
        server.load()
    server.close()

    (ended,) = _ended_records(caplog, "before_ready")
    assert ended.exit_status == 1
    assert ended.reason == ProviderStartFailure.EXITED
    assert (
        "ModuleNotFoundError: No module named 'module_that_is_not_installed'"
        in _failure_text(failed.value)
    )
    assert "module_that_is_not_installed" not in _logged_everywhere(caplog)


def test_a_root_cause_printed_before_a_long_traceback_reaches_the_note(
    tmp_path: Path,
) -> None:
    server = _launching(tmp_path, _failing_start_after_writing(20 * 1024))

    with pytest.raises(ProviderFailedError) as failed:
        server.load()
    server.close()

    assert "ROOT CAUSE: engine core failed" in _failure_text(failed.value)


def test_a_server_writing_without_end_cannot_grow_the_note_without_limit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    written = 1024 * 1024
    server = _launching(tmp_path, _failing_start_after_writing(written))

    with caplog.at_level(logging.WARNING), pytest.raises(ProviderFailedError) as failed:
        server.load()
    server.close()

    (ended,) = _ended_records(caplog, "before_ready")
    assert ended.reason == ProviderStartFailure.STDERR_OVERSIZED
    assert len(_failure_text(failed.value)) < written // 4
    assert "ROOT CAUSE" not in _failure_text(failed.value)


def test_a_server_stopped_by_the_service_before_it_answers_logs_a_failed_start_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root, cache = closed_provider(tmp_path, "held-ready")
    server = ProviderProcess(
        ProviderLaunch(
            executable=provider_launcher(root), arguments=("--cache", str(cache))
        )
    )
    loading = threading.Thread(target=_load_quietly, args=(server,))

    with caplog.at_level(logging.WARNING):
        loading.start()
        wait_for_starts(cache, 1)
        server.close()
        loading.join(timeout=5)

    (ended,) = _ended_records(caplog, "before_ready")
    assert ended.reason == ProviderStartFailure.START_FAILED


def _load_quietly(server: ProviderProcess) -> None:
    with contextlib.suppress(ProviderFailedError):
        server.load()


def test_a_ready_server_logs_no_startup_record_and_its_stderr_reaches_no_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    voice = _serving(tmp_path)

    with caplog.at_level(logging.DEBUG):
        voice.load()
        assert list(voice.pcm_chunks(_SENTENCE, "de")) == [b"\x00\x01"]
        voice.close()

    assert _ended_records(caplog) == []
    assert _SENTENCE not in _logged_everywhere(caplog)
    assert "server noise before ready" not in _logged_everywhere(caplog)


def test_a_server_writing_much_after_ready_never_blocks_on_its_stderr(
    tmp_path: Path,
) -> None:
    voice = _serving(tmp_path)
    voice.load()
    spoken: list[bytes] = []
    speaking = threading.Thread(
        target=lambda: spoken.extend(voice.pcm_chunks("chatty", "de")), daemon=True
    )

    speaking.start()
    speaking.join(timeout=10)

    assert spoken == [b"\x00\x01"]
    assert voice.ready is True
    voice.close()


def test_a_server_that_dies_after_ready_logs_its_exit_status_and_no_sentence(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    voice = _serving(tmp_path)

    with caplog.at_level(logging.DEBUG):
        voice.load()
        with pytest.raises(ProviderSynthesisFailedError):
            list(voice.pcm_chunks("midstream-exit", "de"))
        voice.close()

    assert _SENTENCE not in _logged_everywhere(caplog)
    (ended,) = _ended_records(caplog, "after_ready")
    assert ended.exit_status == 1
