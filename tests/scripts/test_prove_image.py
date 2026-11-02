"""The image proof's own logic: what it admits, what it refuses, and how it can fail."""

from __future__ import annotations

import re
import subprocess
from typing import TYPE_CHECKING

import jwt
import pytest

from speech_runtime_client.wire import RefusalCode, refusal
from speech_runtime_service.config import load_caller_config
from speech_runtime_service.pcm import wav_header
from tests.support.script_support import script_module

if TYPE_CHECKING:
    from pathlib import Path

    from proof_identity import Minting

PROVE = script_module("prove_image")
SAMPLE_RATE = 24_000
SILENCE = bytes(64)


@pytest.fixture
def config() -> PROVE.CallerConfig:
    return load_caller_config(PROVE.BAKED_CONFIG)


def test_the_baked_caller_names_a_voice_it_may_and_one_it_may_not_speak(
    config: PROVE.CallerConfig,
) -> None:
    baked = PROVE.baked_caller(config)

    assert baked.voice in config.apps[baked.app].voices
    assert baked.other_voice not in config.apps[baked.app].voices


def test_tokens_minted_for_the_baked_caller_carry_its_identity(
    config: PROVE.CallerConfig, minting: Minting
) -> None:
    baked = PROVE.baked_caller(config)
    pair = PROVE.mint_pair(PROVE.minting_for(config, baked, minting))

    workload = jwt.decode(pair.workload, options={"verify_signature": False})
    person = jwt.decode(pair.person, options={"verify_signature": False}, audience=None)
    assert workload["iss"] == config.identity.issuer
    assert workload["azp"] == config.apps[baked.app].workload_client
    assert config.identity.workload_role in workload["realm_access"]["roles"]
    assert config.identity.service_group in person[config.identity.group_claim]
    assert person["aud"] == config.identity.audience


def marked_wav() -> bytes:
    return wav_header(SAMPLE_RATE) + SILENCE


@pytest.mark.parametrize(
    "wav",
    [
        pytest.param(PROVE.without_mark(marked_wav()), id="no mark"),
        pytest.param(b"not audio at all", id="not a wav"),
        pytest.param(wav_header(SAMPLE_RATE), id="no audio"),
        pytest.param(wav_header(SAMPLE_RATE)[:-8] + SILENCE, id="no data chunk"),
        pytest.param(
            wav_header(SAMPLE_RATE, 0)
            + b"JUNK"
            + len(SILENCE).to_bytes(4, "little")
            + SILENCE,
            id="empty data chunk before another chunk",
        ),
    ],
)
def test_a_wav_without_the_mark_or_the_audio_fails_the_proof(wav: bytes) -> None:
    with pytest.raises(SystemExit):
        PROVE.require_marked_wav(wav)


def test_a_marked_wav_with_audio_passes_and_loses_its_mark_when_cut() -> None:
    PROVE.require_marked_wav(marked_wav())

    assert PROVE.ai_mark_of(PROVE.without_mark(marked_wav())) is None


def test_the_proof_fails_when_an_action_that_must_fail_does_not() -> None:
    PROVE.require_fails(
        lambda: PROVE.require_marked_wav(b""), "an empty answer", naming="not a WAV"
    )

    with pytest.raises(SystemExit):
        PROVE.require_fails(lambda: None, "nothing", naming="")


def test_the_proof_fails_when_an_action_fails_for_another_reason() -> None:
    def docker_is_down() -> None:
        message = "docker run failed"
        raise SystemExit(message)

    with pytest.raises(SystemExit, match="not as ConnectionRefusedError"):
        PROVE.require_fails(
            docker_is_down, "a loopback bind", naming="ConnectionRefusedError"
        )


def test_a_refusal_passes_only_as_the_expected_code_with_its_status() -> None:
    body = refusal(RefusalCode.FORBIDDEN).model_dump_json().encode()
    status = refusal(RefusalCode.FORBIDDEN).status

    reply = PROVE.Reply(status, body)

    PROVE.require_refusal(reply, RefusalCode.FORBIDDEN)
    with pytest.raises(SystemExit):
        PROVE.require_refusal(reply, RefusalCode.UNAUTHORIZED)


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(PROVE.Reply(200, marked_wav()), id="audio"),
        pytest.param(
            PROVE.Reply(200, refusal(RefusalCode.BUSY).model_dump_json().encode()),
            id="status of another code",
        ),
    ],
)
def test_an_answer_that_is_no_matching_refusal_fails_the_proof(
    reply: PROVE.Reply,
) -> None:
    with pytest.raises(SystemExit):
        PROVE.require_refusal(reply, RefusalCode.BUSY)


def test_locked_names_come_from_the_export_for_this_platform_and_its_projects() -> None:
    export = """\
# a comment
-e ./packages/client
Some_Package==1.0 ; python_version >= '3.14'
windows-only==1.0 ; sys_platform == 'win32'
webauth @ git+https://example.invalid/webauth@abc
    # via something
"""

    names = PROVE.locked_distributions(export, lambda path: f"own-{path[-6:]}")

    assert names == {"own-client", "some-package", "webauth"}


def clean_report() -> PROVE.ImageReport:
    return PROVE.ImageReport(
        distributions=["Some_Package"],
        user=1000,
        application_files=["deploy/speech.toml"],
        application_files_with_private_keys=[],
        package_stray_files=[],
        package_stray_files_with_private_keys=[],
        voice_hashes={"voice.onnx": "abc"},
        executables={"git": None},
        nvidia_directories=[],
    )


@pytest.mark.parametrize(
    ("change", "fault"),
    [
        ({"distributions": ["Some_Package", "nvidia-cublas-cu12"]}, "extra"),
        ({"distributions": []}, "missing"),
        ({"user": 0}, "root"),
        ({"application_files": ["deploy/.git/config"]}, "holds"),
        ({"application_files": ["tests/test_x.py"]}, "holds"),
        ({"application_files": ["deploy/token.key"]}, "holds"),
        ({"application_files_with_private_keys": ["a.txt"]}, "private key"),
        ({"executables": {"git": "/usr/bin/git"}}, "carries git"),
        ({"nvidia_directories": ["nvidia"]}, "holds nvidia"),
        (
            {"package_stray_files": ["speech_runtime_service/local.txt"]},
            "among the packages",
        ),
        (
            {"package_stray_files_with_private_keys": ["speech_runtime_x/a.key"]},
            "private key",
        ),
        ({"voice_hashes": {"voice.onnx": "other"}}, "pinned bytes"),
    ],
)
def test_an_image_that_differs_from_the_lock_is_named_by_its_fault(
    change: dict[str, object], fault: str
) -> None:
    report = PROVE.ImageReport(**{**vars(clean_report()), **change})

    faults = PROVE.image_faults(report, {"some-package"}, {"voice.onnx": "abc"})

    assert any(fault in line for line in faults)


def test_an_image_as_locked_has_no_fault() -> None:
    assert not PROVE.image_faults(
        clean_report(), {"some-package"}, {"voice.onnx": "abc"}
    )


def test_a_log_with_spoken_text_or_a_token_is_named() -> None:
    log = "INFO ok\nspoke: Hallo Welt\nheader abc.def.ghi"

    faults = PROVE.log_faults(log, ["Hallo Welt"], ["abc.def.ghi"])
    assert [fault.split()[0] for fault in faults] == ["spoken", "a"]
    assert not PROVE.log_faults("INFO ok", ["Hallo Welt"], ["abc.def.ghi"])


def test_only_the_external_networks_of_a_rendered_compose_file_are_named() -> None:
    rendered = {
        "networks": {
            "a": {"name": "outside", "external": True},
            "b": {"name": "inside"},
        }
    }

    assert PROVE.external_networks(rendered) == ["outside"]


def test_an_nvidia_distribution_in_the_image_is_named_even_when_the_lock_lists_it() -> (
    None
):
    report = PROVE.ImageReport(
        **{**vars(clean_report()), "distributions": ["Some_Package", "nvidia-nccl"]}
    )

    faults = PROVE.image_faults(
        report, {"some-package", "nvidia-nccl"}, {"voice.onnx": "abc"}
    )

    assert faults == ["holds nvidia-nccl"]


class FakeDocker:
    """The process boundary: every command ends with the code its name is given."""

    def __init__(self, failing: set[str]) -> None:
        """Fail the commands named in `failing`."""
        self.failing = failing
        self.commands: list[str] = []

    def __call__(
        self, command: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        """Answer one command with the code its name was given."""
        name = next(word for word in ("image", "down", "logs") if word in command)
        self.commands.append(name)
        code = 1 if name in self.failing else 0
        return subprocess.CompletedProcess(command, code, "", f"{name} broke")


@pytest.mark.parametrize(
    ("failing", "left"),
    [
        pytest.param(set(), [], id="clean"),
        pytest.param({"down"}, ["compose down"], id="compose down fails"),
        pytest.param({"image"}, ["docker image rm"], id="image removal fails"),
        pytest.param({"down", "image"}, ["compose down", "docker image rm"], id="both"),
    ],
)
def test_a_teardown_tries_both_removals_and_names_each_that_failed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing: set[str],
    left: list[str],
) -> None:
    fake = FakeDocker(failing)
    monkeypatch.setattr(PROVE.subprocess, "run", fake)

    result = PROVE.teardown(PROVE.Compose((PROVE.COMPOSE_FILE,), {}), failed=False)

    assert fake.commands == ["down", "image"]
    assert [line.split(" failed")[0] for line in result] == left
    assert ("torn_down" in capsys.readouterr().out) is (not left)


def test_every_job_the_quality_gate_waits_for_is_named_in_its_needs() -> None:
    workflow = (PROVE.REPOSITORY / ".github" / "workflows" / "ci.yml").read_text()
    jobs, _, gate = workflow.partition("\n  quality:\n")
    jobs = jobs.partition("\njobs:\n")[2]
    needs = set(re.findall(r"^      - (\S+)$", gate.split("if:")[0], re.MULTILINE))

    assert needs == set(re.findall(r"^  ([a-z-]+):$", jobs, re.MULTILINE))


def test_the_build_context_admits_only_files_the_image_copies() -> None:
    admitted = [
        line.removeprefix("!")
        for line in (PROVE.REPOSITORY / ".dockerignore").read_text().splitlines()
        if line.startswith("!")
    ]
    dockerfile = (PROVE.REPOSITORY / "Dockerfile").read_text()
    copied = {
        source
        for line in re.findall(r"^COPY (?!--from)(.+)$", dockerfile, re.MULTILINE)
        for source in line.split()[:-1]
    }

    assert all(path.partition("/**/")[0] in copied for path in admitted)


def test_the_source_trees_admit_python_source_and_py_typed_and_nothing_else() -> None:
    lines = (PROVE.REPOSITORY / ".dockerignore").read_text().splitlines()
    trees = [
        line.removeprefix("!") for line in lines if "/src" in line or line == "!src"
    ]

    assert trees
    assert all(tree.endswith(("/*.py", "/py.typed")) for tree in trees)


def test_the_exclusions_of_nested_repositories_and_caches_follow_the_admissions() -> (
    None
):
    lines = [
        line
        for line in (PROVE.REPOSITORY / ".dockerignore").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    last_admission = max(i for i, line in enumerate(lines) if line.startswith("!"))

    assert {"**/.git", "**/.venv", "**/__pycache__"} <= set(lines[last_admission + 1 :])


def test_a_failed_removal_of_the_temporary_files_fails_the_proof_loudly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class StoppedCompose:
        def __init__(self, *_: object) -> None:
            pass

        def run(self, *_: object, **__: object) -> object:
            message = "stopped"
            raise SystemExit(message)

    directory = tmp_path / "proof"
    stuck = directory / "stuck"
    stuck.mkdir(parents=True)
    (stuck / "file").write_text("x")
    stuck.chmod(0o500)
    monkeypatch.setattr(PROVE, "wait_for_load", lambda: None)
    monkeypatch.setattr(PROVE.tempfile, "mkdtemp", lambda **_: str(directory))
    monkeypatch.setattr(PROVE, "Compose", StoppedCompose)
    monkeypatch.setattr(PROVE, "teardown", lambda *_, **__: [])
    try:
        with pytest.raises(PermissionError):
            PROVE._prove()  # noqa: SLF001 - the whole run is the entry point under test
    finally:
        stuck.chmod(0o700)

    assert "PROOF_OK" not in capsys.readouterr().out
