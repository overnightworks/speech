"""The proof's log check rejects a service log that holds spoken text."""

from __future__ import annotations

import functools
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from shipped_catalogue import CATALOGUE, ON_DEMAND_VOICES, PIPER_VOICES, first_language
from speech_client.outcomes import Unreachable
from speech_client.wire import Health, VoiceId
from speech_service.config import Settings
from speech_service.hearing_models import HearingModel
from speech_service.onnxruntime_telemetry import ONNXRUNTIME_TELEMETRY_SWITCH
from tests.support.clock_support import FakeClock
from tests.support.script_support import script_module
from tests.support.speaking_health_support import health_offering

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextlib import AbstractContextManager
    from types import ModuleType
    from typing import IO

    from speech_client.health import HealthOutcome

PROVE = script_module("prove_speech")
HEARING = script_module("prove_hearing")
LINES = script_module("voice_lines")
PIPE_BYTES = 65536


@pytest.mark.parametrize(
    ("log", "texts", "leaked"),
    [
        ("INFO started\nINFO text: Hallo Welt", ["Hallo Welt"], ["Hallo Welt"]),
        ("INFO started\nINFO speaking done", ["Hallo Welt", "Die Kamera"], []),
        ("INFO started", ["", "  "], []),
    ],
)
def test_the_log_check_reports_only_the_texts_the_log_holds(
    log: str, texts: list[str], leaked: list[str]
) -> None:
    assert PROVE.spoken_in_log(log, texts) == leaked


FLEET_BUDGET_SECONDS = 1830


@pytest.fixture
def restored_terminate_handler() -> Iterator[None]:
    """Put the TERM handler back after a test that lets the proof install its own."""
    previous = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, previous)


@pytest.fixture
def terminate_handler() -> Iterator[None]:
    """Install the proof's TERM handler for one test and put the old one back."""
    previous = signal.getsignal(signal.SIGTERM)
    PROVE.exit_on_terminate()
    yield
    signal.signal(signal.SIGTERM, previous)


FAKE_PROBE_RUN_STATUS = 5
HEARING_MODEL = next(iter(HearingModel)).value
VOICE = next(iter(VoiceId)).value


def run_returning(
    status: int,
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    """A fake for the re-exec: the `probe-run` command ends with this status."""

    def run(
        command: list[str],
        env: dict[str, str],  # noqa: ARG001 - the run signature
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        assert not check
        return subprocess.CompletedProcess(command, status)

    return run


def test_the_proof_runs_itself_through_probe_run_with_its_own_cap() -> None:
    command = PROVE.fleet_command(["--help"])

    assert command[:4] == [
        PROVE.PROBE_RUN,
        "-k",
        str(PROVE.KILL_GRACE_SECONDS),
        str(PROVE.CAP_SECONDS),
    ]
    assert command[4] == sys.executable
    assert Path(command[5]) == Path(PROVE.__file__)
    assert command[6:] == ["--help"]


def test_another_script_runs_under_the_same_fleet_form() -> None:
    other = Path("/scripts/other_proof.py")

    command = PROVE.fleet_command(["--device", "cpu"], other)

    assert Path(command[5]) == other
    assert command[6:] == ["--device", "cpu"]


def test_the_cap_and_kill_grace_fit_the_fleet_budget() -> None:
    assert PROVE.CAP_SECONDS + PROVE.KILL_GRACE_SECONDS <= FLEET_BUDGET_SECONDS


def test_a_run_outside_probe_run_hands_over_to_the_fleet_form() -> None:
    started: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def run(
        command: list[str], env: dict[str, str], *, check: bool
    ) -> subprocess.CompletedProcess[bytes]:
        assert not check
        started.append(command)
        environments.append(env)
        return subprocess.CompletedProcess(command, 0)

    other = Path("/scripts/other_proof.py")

    status = PROVE.run_under_probe_run(
        ["--help"], run, other, which=lambda _: "/bin/probe-run", environment={"X": "1"}
    )

    assert status == 0
    assert started == [PROVE.fleet_command(["--help"], other)]
    assert PROVE.UNDER_PROBE_RUN_ENV in environments[0]
    assert environments[0]["X"] == "1"


def test_a_missing_probe_run_fails_the_run_naming_it() -> None:
    run = run_returning(0)

    with pytest.raises(SystemExit) as failure:
        PROVE.run_under_probe_run([], run, which=lambda _: None)

    assert PROVE.PROBE_RUN in str(failure.value)


def test_a_failed_proof_under_probe_run_keeps_its_own_exit_status() -> None:
    status = 3

    assert (
        PROVE.run_under_probe_run(
            [], run_returning(status), which=lambda _: "/bin/probe-run"
        )
        == status
    )


def health_without_voices(*, ready: bool) -> Health:
    return Health.model_validate(
        {
            "speaking": {"ready": ready, "voices": {}},
            "hearing": {"model": "none", "ready": False},
            "card_memory_mb": 7,
        }
    )


def test_a_wait_returns_the_first_health_that_shows_speaking_ready() -> None:
    answers = iter(
        [health_without_voices(ready=False), health_without_voices(ready=True)]
    )
    clock = FakeClock()

    health = PROVE.wait_for_speaking_ready(
        "http://127.0.0.1:1",
        lambda: None,
        fetch=lambda *_: next(answers),
        clock=clock.now,
        sleep=clock.jump,
    )

    assert health.speaking.ready


def test_a_health_that_does_not_offer_the_voice_ends_the_proof() -> None:
    health = health_without_voices(ready=True)

    with pytest.raises(SystemExit, match="does not offer"):
        PROVE.require_speaking(health, PIPER_VOICES[0])


def test_a_health_that_offers_the_voice_lets_the_proof_go_on() -> None:
    voice = PIPER_VOICES[0]
    health = health_offering(voice, CATALOGUE[voice].languages)

    PROVE.require_speaking(health, voice)


def a_server_whose_log_holds(text: str) -> Callable[..., Mock]:
    """A fake start whose service has logged `text` and exits at once."""

    def start(*_: object, stdout: IO[str], **__: object) -> Mock:
        stdout.write(text)
        stdout.flush()
        return Mock(returncode=0)

    return start


def speech_server_logging(tmp_path: Path, text: str) -> AbstractContextManager[object]:
    return PROVE.speech_server(
        "http://127.0.0.1:1",
        {},
        tmp_path,
        start=a_server_whose_log_holds(text),
        wait_ready=lambda *_: health_without_voices(ready=True),
    )


def test_a_passed_proof_whose_service_log_holds_a_spoken_text_fails(
    tmp_path: Path,
) -> None:
    running = speech_server_logging(tmp_path, PROVE.SENTENCE)

    with pytest.raises(SystemExit, match="holds 1 spoken text"), running:
        pass


def raise_the_proofs_own_error() -> None:
    message = "the proof's own error"
    raise RuntimeError(message)


def test_a_failed_proof_reports_a_spoken_text_in_the_log_without_hiding_its_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    running = speech_server_logging(tmp_path, PROVE.SENTENCE)

    with pytest.raises(RuntimeError, match="the proof's own error"), running:
        raise_the_proofs_own_error()

    assert "holds 1 spoken text" in capsys.readouterr().out


def test_a_child_that_ignores_term_is_killed_after_the_grace() -> None:
    ignores_term = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        " print('armed', flush=True); time.sleep(600)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", ignores_term], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "armed"

        PROVE.stop_process(child, grace_seconds=0.2)

        assert child.returncode == -signal.SIGKILL
    finally:
        child.kill()
        child.wait()
        child.stdout.close()


def test_the_default_wait_ends_at_once_when_the_server_exited(tmp_path: Path) -> None:
    def never_waits(_: float) -> None:
        pytest.fail("the wait went on after the server exited")

    server = Mock(stdout=None, returncode=3)
    server.poll.return_value = 3
    running = PROVE.speech_server(
        "http://127.0.0.1:1",
        {},
        tmp_path,
        start=lambda *_, **__: server,
        wait_ready=functools.partial(
            PROVE.wait_for_speaking_ready,
            fetch=lambda *_: health_without_voices(ready=False),
            sleep=never_waits,
        ),
    )

    with pytest.raises(SystemExit, match="exited with status 3"), running:
        pass


@pytest.mark.usefixtures("terminate_handler")
def test_term_stops_the_speech_server_through_its_teardown(tmp_path: Path) -> None:
    server = Mock(stdout=None, returncode=0)
    running = PROVE.speech_server(
        "http://127.0.0.1:1",
        {},
        tmp_path,
        start=lambda *_, **__: server,
        wait_ready=lambda *_: health_without_voices(ready=True),
    )

    def term_while_running() -> None:
        with running:
            os.kill(os.getpid(), signal.SIGTERM)

    with pytest.raises(SystemExit):
        term_while_running()

    server.terminate.assert_called_once_with()


@pytest.mark.usefixtures("restored_terminate_handler")
def test_term_during_the_proof_under_probe_run_runs_its_teardown() -> None:
    torn_down: list[bool] = []

    def proof_body() -> None:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            torn_down.append(True)

    with pytest.raises(SystemExit):
        PROVE.prove_under_probe_run(proof_body)

    assert torn_down == [True]


@pytest.mark.usefixtures("restored_terminate_handler")
@pytest.mark.parametrize(
    ("before", "expected"),
    [(None, "1"), ("0", "0")],
    ids=["unset_is_switched_off", "operator_value_stays"],
)
def test_the_proof_body_runs_with_onnxruntime_telemetry_off_unless_named(
    monkeypatch: pytest.MonkeyPatch, before: str | None, expected: str
) -> None:
    if before is None:
        monkeypatch.delenv(ONNXRUNTIME_TELEMETRY_SWITCH, raising=False)
    else:
        monkeypatch.setenv(ONNXRUNTIME_TELEMETRY_SWITCH, before)
    seen: list[str | None] = []

    PROVE.prove_under_probe_run(
        lambda: seen.append(os.environ.get(ONNXRUNTIME_TELEMETRY_SWITCH))
    )

    assert seen == [expected]


def test_an_entry_outside_probe_run_exits_with_the_status_of_its_reexec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["proof"])
    monkeypatch.delenv(PROVE.UNDER_PROBE_RUN_ENV, raising=False)
    status = 7
    proved: list[bool] = []

    with pytest.raises(SystemExit) as exit_request:
        PROVE.enter_proof(lambda _: status, lambda _arguments: proved.append(True))

    assert exit_request.value.code == status
    assert proved == []


@pytest.fixture
def fake_probe_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `probe-run` first on PATH that records its command line and exits 5."""
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    record = tmp_path / "probe-run-argv"
    fake = bin_directory / PROVE.PROBE_RUN
    fake.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$@" > "{record}"\nexit {FAKE_PROBE_RUN_STATUS}\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_directory}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv(PROVE.UNDER_PROBE_RUN_ENV, raising=False)
    return record


@pytest.mark.parametrize(
    ("script", "arguments"),
    [
        pytest.param(
            HEARING,
            [
                "--device",
                "cpu",
                "--hearing-model",
                HEARING_MODEL,
                "--record",
                "r.json",
                "--english-voice-directory",
                "voices",
            ],
            id="prove_hearing",
        ),
        pytest.param(
            LINES,
            ["--voice", VOICE, "--hearing-model", HEARING_MODEL, "--device", "cpu"],
            id="voice_lines",
        ),
    ],
)
def test_an_entry_outside_probe_run_reexecutes_its_own_script_with_its_arguments(
    fake_probe_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: ModuleType,
    arguments: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [script.__file__, *arguments])

    with pytest.raises(SystemExit) as exit_request:
        script.main()

    assert exit_request.value.code == FAKE_PROBE_RUN_STATUS
    command = fake_probe_run.read_text(encoding="utf-8").splitlines()
    assert command[-len(arguments) - 2 :] == [
        sys.executable,
        str(Path(script.__file__).resolve()),
        *arguments,
    ]


@pytest.mark.usefixtures("restored_terminate_handler")
def test_an_entry_inside_probe_run_runs_the_proof_not_the_reexec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["proof"])
    monkeypatch.setenv(PROVE.UNDER_PROBE_RUN_ENV, "1")
    proved: list[bool] = []

    PROVE.enter_proof(
        Mock(side_effect=AssertionError), lambda _arguments: proved.append(True)
    )

    assert proved == [True]


def test_a_service_ends_when_the_thread_that_started_it_ends() -> None:
    sleeps_until_told = "import time; print('up', flush=True); time.sleep(600)"
    started: list[subprocess.Popen[str]] = []

    def start_and_wait_until_up() -> None:
        child = PROVE.start_with_parent_death(
            [sys.executable, "-c", sleeps_until_told],
            stdout=subprocess.PIPE,
            text=True,
        )
        started.append(child)
        child.stdout.readline()

    thread = threading.Thread(target=start_and_wait_until_up)
    thread.start()
    thread.join()
    child = started[0]
    try:
        assert child.wait(timeout=30) == -signal.SIGTERM
    finally:
        child.kill()
        child.wait()
        child.stdout.close()


FAKE_SERVICE = """
import os, sys
for number in range(LINES):
    print("log line", number, "x" * 100, flush=True)
print("last line", flush=True)
with open(os.environ["SERVICE_LOGGED_FIFO"], "w") as logged:
    logged.write("done")
import threading
threading.Event().wait()
"""


def test_the_default_start_never_blocks_a_service_that_logs_past_the_pipe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = tmp_path / "packages" / "speech_service"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(
        FAKE_SERVICE.replace("LINES", str(PIPE_BYTES // 100 * 2))
    )
    logged = tmp_path / "logged"
    os.mkfifo(logged)
    env = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": str(tmp_path / "packages"),
        "SERVICE_LOGGED_FIFO": str(logged),
    }

    def wait_until_logged(*_: object) -> dict[str, object]:
        assert logged.read_text() == "done"
        return {}

    with PROVE.speech_server(
        "http://127.0.0.1:1", env, tmp_path, wait_ready=wait_until_logged
    ):
        pass

    output = capsys.readouterr().out
    assert "last line" in output
    assert "server_exit" in output


@pytest.mark.parametrize(
    ("health", "detail_in_message"),
    [
        (health_without_voices(ready=False), True),
        (None, False),
    ],
    ids=["the last health is named", "no health ever answered"],
)
def test_a_wait_that_runs_out_names_the_last_health_it_saw(
    health: Health | None,
    *,
    detail_in_message: bool,
) -> None:
    clock = FakeClock()

    def answer(*_: object) -> HealthOutcome:
        return Unreachable() if health is None else health

    with pytest.raises(SystemExit, match="not ready within") as failure:
        PROVE.wait_for_speaking_ready(
            "http://127.0.0.1:1",
            lambda: None,
            fetch=answer,
            clock=clock.now,
            sleep=clock.jump,
        )

    assert ("card_memory_mb" in str(failure.value)) is detail_in_message


def test_a_child_whose_parent_is_not_its_starter_never_runs(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "ran"

    child = PROVE.start_with_parent_death(
        [sys.executable, "-c", f"open({str(marker)!r}, 'w')"],
        parent_pid=1,
    )

    assert child.wait(timeout=30) == 1
    assert not marker.exists()


def test_every_piper_voice_has_a_proof_sentence_in_its_language() -> None:
    assert all(first_language(voice) in PROVE.SENTENCES for voice in PIPER_VOICES)


def test_every_piper_voice_has_a_marks_text_in_its_language() -> None:
    assert all(first_language(voice) in PROVE.MARKED_TEXTS for voice in PIPER_VOICES)


@pytest.mark.parametrize(
    "offered",
    [{*PIPER_VOICES}, {*PIPER_VOICES, ON_DEMAND_VOICES[0]}, {ON_DEMAND_VOICES[0]}],
    ids=["the Piper voices", "with the GPU voice", "the GPU voice alone"],
)
def test_the_offer_the_proof_sets_is_the_offer_the_service_reads(
    offered: set[VoiceId], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPEECH_VOICES", PROVE.offer_setting(offered))

    assert Settings().voices == offered
