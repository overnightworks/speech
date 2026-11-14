"""A provider that fails leaves its cause in the service log, without the sentence."""

from __future__ import annotations

import logging
import os
import re
import signal
import sys
import time
import tomllib
from pathlib import Path

import pytest
from speech_provider_contract import QWEN_MODEL_ID, QWEN_SAMPLE_RATE

from speech_service.provider_process import (
    ENDED_AFTER_READY,
    ENDED_BEFORE_READY,
    SYNTHESIS_FAILED,
    ProviderFailedError,
    ProviderId,
    ProviderLaunch,
    ProviderProcess,
    ProviderProtocolError,
    ProviderSynthesisFailedError,
    provider_entrypoint,
)
from tests.support.worker_support import closed_provider, provider_worker

_ENTRYPOINT = """#!{python}
{body}
"""

_CONTRACT_BODY = """import os
import sys
from pathlib import Path

from speech_provider_contract import ProviderFunctions, serve_provider

cache = sys.argv[sys.argv.index("--cache") + 1]
protocol_stdout = os.dup(1)


def load(device, cache):
    if {fail_load!r}:
        raise RuntimeError("model weights unreadable")
    return object()


def synthesize(model, text, language, instructions):
    print(text, file=sys.stderr, flush=True)
    Path(cache).mkdir(exist_ok=True)
    Path(cache, "stderr-target").write_text(os.readlink("/proc/self/fd/2"))
    yield b"\\x00\\x01"


raise SystemExit(serve_provider(
    device="cpu", cache=cache, protocol_stdout=protocol_stdout,
    sample_rate=24000, functions=ProviderFunctions(load, synthesize),
))
"""


_FAILING_BODY = """import errno
import os
import sys
from pathlib import Path

from speech_provider_contract import ProviderFunctions, serve_provider

cache = sys.argv[sys.argv.index("--cache") + 1]
protocol_stdout = os.dup(1)


class AcceleratorError(RuntimeError):
    pass


class SentenceCode(int):
    def __new__(cls, value, text):
        code = super().__new__(cls, value)
        code.text = text
        return code

    def __str__(self):
        return self.text

    __repr__ = __str__

    def __format__(self, _specification):
        return self.text


def coded(code):
    error = AcceleratorError()
    error.error_code = code
    return error


class PropertyCodeError(RuntimeError):
    def __init__(self, text):
        super().__init__()
        self.text = text

    @property
    def error_code(self):
        Path(cache, "property-ran").write_text("ran")
        return self.text


def synthesize(model, text, language, instructions):
{synthesis}
    yield b"\\x00\\x01"


raise SystemExit(serve_provider(
    device="cpu", cache=cache, protocol_stdout=protocol_stdout,
    sample_rate=24000,
    functions=ProviderFunctions(lambda _device, _cache: object(), synthesize),
))
"""
_STAGE_BODY = """import os
import subprocess
import sys
from pathlib import Path

from speech_provider_contract import ProviderFunctions, serve_provider

cache = Path(sys.argv[sys.argv.index("--cache") + 1])
protocol_stdout = os.dup(1)
stage = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
cache.mkdir(exist_ok=True)
cache.joinpath("worker").write_text(str(os.getpid()))
cache.joinpath("stage").write_text(str(stage.pid))

raise SystemExit(serve_provider(
    device="cpu", cache=str(cache), protocol_stdout=protocol_stdout,
    sample_rate=24000,
    functions=ProviderFunctions(lambda _device, _cache: object(), lambda *_: ()),
))
"""
_SENTENCE = "Ein Satz, den nur der Aufrufer kennt"


def _provider(tmp_path: Path, body: str) -> ProviderProcess:
    root, cache = closed_provider(tmp_path)
    entrypoint = provider_worker(root)
    entrypoint.write_text(
        _ENTRYPOINT.format(python=sys.executable, body=body), encoding="utf-8"
    )
    return ProviderProcess(
        ProviderLaunch(
            model_name=QWEN_MODEL_ID,
            sample_rate=QWEN_SAMPLE_RATE,
            executable=entrypoint,
            arguments=("--device", "cpu", "--cache", str(cache)),
        )
    )


def test_the_derived_worker_name_is_the_script_the_provider_declares() -> None:
    declared = tomllib.loads(
        (Path(__file__).parents[1] / "providers/qwen/pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]["scripts"]

    entrypoint = provider_entrypoint(Path("providers"), ProviderId.QWEN)

    assert entrypoint is not None
    assert entrypoint.name in declared


def test_provider_failing_to_import_logs_the_module_and_its_own_exit_status(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine = _provider(tmp_path, "import module_that_is_not_installed")

    with caplog.at_level(logging.ERROR), pytest.raises(ProviderFailedError):
        engine.load()
    engine.close()

    assert "exit status 1," in caplog.text
    assert (
        "ModuleNotFoundError: No module named 'module_that_is_not_installed'"
        in caplog.text
    )


def test_provider_failing_to_load_the_model_logs_its_cause_and_a_reaped_status(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine = _provider(tmp_path, _CONTRACT_BODY.format(fail_load=True))

    with caplog.at_level(logging.ERROR), pytest.raises(ProviderFailedError):
        engine.load()
    engine.close()

    assert caplog.text.count(ENDED_BEFORE_READY) == 1
    assert "RuntimeError: model weights unreadable" in caplog.text
    # Whether the service's stop or the provider's own exit comes first is a
    # race; either way the reaped provider has a status.
    assert re.search(r"exit status -?\d+,", caplog.text)


def _failing_start_after_writing(stderr_bytes: int) -> str:
    return (
        "import sys\n"
        "sys.stderr.write('ROOT CAUSE: engine core failed\\n')\n"
        f"sys.stderr.write('Traceback line\\n' * ({stderr_bytes} // 15))\n"
        "sys.stderr.flush()\n"
        "raise SystemExit(1)"
    )


def test_a_root_cause_printed_before_a_long_traceback_reaches_the_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine = _provider(tmp_path, _failing_start_after_writing(20 * 1024))

    with caplog.at_level(logging.ERROR), pytest.raises(ProviderFailedError):
        engine.load()
    engine.close()

    assert "ROOT CAUSE: engine core failed" in caplog.text


def test_a_provider_writing_without_end_cannot_grow_the_log_without_limit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    written = 1024 * 1024
    engine = _provider(tmp_path, _failing_start_after_writing(written))

    with caplog.at_level(logging.ERROR), pytest.raises(ProviderFailedError):
        engine.load()
    engine.close()

    assert len(caplog.text) < written // 4
    assert "ROOT CAUSE" not in caplog.text


def test_ready_provider_logs_no_startup_record_and_its_stderr_reaches_no_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine = _provider(tmp_path, _CONTRACT_BODY.format(fail_load=False))

    with caplog.at_level(logging.DEBUG):
        engine.load()
        assert list(engine.pcm_chunks("caller secret words", "de")) == [b"\x00\x01"]
        engine.close()

    assert ENDED_BEFORE_READY not in caplog.text
    assert ENDED_AFTER_READY not in caplog.text
    assert "caller secret words" not in caplog.text
    assert (tmp_path / "cache/stderr-target").read_text() == os.devnull


def _failed_synthesis_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, synthesis: str
) -> str:
    engine = _provider(tmp_path, _FAILING_BODY.format(synthesis=synthesis))
    with caplog.at_level(logging.DEBUG):
        engine.load()
        with pytest.raises(ProviderSynthesisFailedError):
            list(engine.pcm_chunks(_SENTENCE, "de"))
        engine.close()
    _assert_no_sentence(caplog)
    (record,) = (
        record for record in caplog.records if SYNTHESIS_FAILED in record.getMessage()
    )
    return record.getMessage()


def _assert_no_sentence(caplog: pytest.LogCaptureFixture) -> None:
    assert _SENTENCE not in caplog.text
    assert all(_SENTENCE not in (record.exc_text or "") for record in caplog.records)


def test_a_failed_synthesis_logs_its_errno_and_frame_and_an_ended_record(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    message = _failed_synthesis_log(
        tmp_path,
        caplog,
        "    raise OSError(errno.EROFS, text, '/' + text)",
    )

    assert message.startswith(f"{SYNTHESIS_FAILED}, request 1, cause: ")
    assert "builtins.OSError[EROFS]" in message
    assert ", in synthesize\n" in message
    assert ENDED_AFTER_READY in caplog.text


def test_a_failed_synthesis_with_the_sentence_as_errno_names_no_errno(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    message = _failed_synthesis_log(tmp_path, caplog, "    raise OSError(text, 'x')")

    assert "builtins.OSError" in message
    assert "builtins.OSError[" not in message


@pytest.mark.parametrize(
    ("raised", "rendered"),
    [
        ("coded(209)", " error_code 209"),
        ("coded(text)", None),
        ("coded(SentenceCode(209, text))", None),
        ("PropertyCodeError(text)", None),
    ],
    ids=["int", "sentence", "int-subclass", "property"],
)
def test_a_failed_synthesis_renders_only_an_exact_int_error_code(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    raised: str,
    rendered: str | None,
) -> None:
    message = _failed_synthesis_log(tmp_path, caplog, f"    raise {raised}")

    if rendered is None:
        assert "error_code" not in message
    else:
        assert rendered in message
    assert not (tmp_path / "cache/property-ran").exists()


def test_a_failed_synthesis_leaves_its_notes_unrendered(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    message = _failed_synthesis_log(
        tmp_path,
        caplog,
        "    error = RuntimeError()\n    error.add_note(text)\n    raise error",
    )

    assert "builtins.RuntimeError" in message


def test_a_failed_synthesis_names_its_chain_from_the_root_cause_with_its_frames(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    message = _failed_synthesis_log(
        tmp_path,
        caplog,
        "    try:\n        raise KeyError(text)\n    except KeyError as error:\n"
        "        raise AcceleratorError() from error",
    )

    assert "cause: builtins.KeyError -> __main__.AcceleratorError\n" in message
    assert "\nbuiltins.KeyError\n  File " in message
    assert "    raise KeyError(text)\n" in message


def test_a_failed_synthesis_escapes_control_and_non_ascii_source_characters(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    message = _failed_synthesis_log(
        tmp_path,
        caplog,
        "    raise RuntimeError()  # tab\there, feed\x0chere, \xe9",
    )

    assert "# tab\\there, feed\\x0chere, \\xe9" in message


def test_a_failure_during_a_cancel_drain_logs_its_cause(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine = _provider(
        tmp_path,
        _FAILING_BODY.format(
            synthesis="    try:\n        while True:\n            yield b'\\x00\\x01'\n"
            "    except GeneratorExit:\n        raise OSError(errno.EIO, text)"
        ),
    )
    with caplog.at_level(logging.DEBUG):
        engine.load()
        chunks = engine.pcm_chunks(_SENTENCE, "de")
        assert next(chunks) == b"\x00\x01"
        chunks.close()
        engine.close()

    _assert_no_sentence(caplog)
    cause = "builtins.GeneratorExit -> builtins.OSError[EIO]"
    assert f"{SYNTHESIS_FAILED}, request 1, cause: {cause}" in caplog.text
    assert engine.ready is False


def test_a_provider_that_dies_after_ready_logs_its_exit_status(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine = _provider(tmp_path, _FAILING_BODY.format(synthesis="    os._exit(3)"))
    with caplog.at_level(logging.DEBUG):
        engine.load()
        with pytest.raises(ProviderProtocolError):
            list(engine.pcm_chunks(_SENTENCE, "de"))
        engine.close()

    _assert_no_sentence(caplog)
    assert caplog.text.count(f"{ENDED_AFTER_READY}, exit status 3") == 1


def _has_ended(pid: int) -> bool:
    """Whether `pid` is gone or a zombie that its adopter has yet to reap."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            status = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        except FileNotFoundError, ProcessLookupError:
            return True
        if status.rpartition(")")[2].split()[0] == "Z":
            return True
        time.sleep(0.01)
    return False


def test_a_stage_process_that_outlives_its_worker_ends_with_the_service_stop(
    tmp_path: Path,
) -> None:
    engine = _provider(tmp_path, _STAGE_BODY)
    engine.load()
    cache = tmp_path / "cache"
    worker = int((cache / "worker").read_text())
    stage = int((cache / "stage").read_text())

    os.kill(worker, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while engine.ready and time.monotonic() < deadline:
        time.sleep(0.01)
    engine.close()

    assert not engine.ready
    assert _has_ended(stage)
