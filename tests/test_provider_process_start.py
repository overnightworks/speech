"""A provider that ends before it is ready leaves its cause in the service log."""

from __future__ import annotations

import logging
import os
import re
import sys
import tomllib
from pathlib import Path

import pytest
from speech_provider_contract import QWEN_MODEL_ID, QWEN_SAMPLE_RATE

from speech_service.provider_process import (
    ENDED_BEFORE_READY,
    ProviderFailedError,
    ProviderId,
    ProviderLaunch,
    ProviderProcess,
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


def synthesize(model, text, language):
    print(text, file=sys.stderr, flush=True)
    Path(cache).mkdir(exist_ok=True)
    Path(cache, "stderr-target").write_text(os.readlink("/proc/self/fd/2"))
    yield b"\\x00\\x01"


raise SystemExit(serve_provider(
    device="cpu", cache=cache, protocol_stdout=protocol_stdout,
    sample_rate=24000, functions=ProviderFunctions(load, synthesize),
))
"""


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


def test_ready_provider_logs_no_startup_record_and_its_stderr_reaches_no_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine = _provider(tmp_path, _CONTRACT_BODY.format(fail_load=False))

    with caplog.at_level(logging.DEBUG):
        engine.load()
        assert list(engine.pcm_chunks("caller secret words", "de")) == [b"\x00\x01"]
        engine.close()

    assert ENDED_BEFORE_READY not in caplog.text
    assert "caller secret words" not in caplog.text
    assert (tmp_path / "cache/stderr-target").read_text() == os.devnull
