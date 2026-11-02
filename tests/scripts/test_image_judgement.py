"""The image proof's judgement: what it admits, what it refuses, and how it can fail."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jwt
import pytest

import image_judgement as judged
from image_stack import REPOSITORY, Reply
from proof_identity import mint_pair
from speech_runtime_client.wire import RefusalCode, refusal
from speech_runtime_service.config import CallerConfig, load_caller_config
from speech_runtime_service.pcm import wav_header

if TYPE_CHECKING:
    from proof_identity import Minting

BAKED_CONFIG = REPOSITORY / "deploy" / "speech.toml"
SAMPLE_RATE = 24_000
SILENCE = bytes(64)


@pytest.fixture
def config() -> CallerConfig:
    return load_caller_config(BAKED_CONFIG)


def test_the_baked_caller_names_a_voice_it_may_and_one_it_may_not_speak(
    config: CallerConfig,
) -> None:
    baked = judged.baked_caller(config)

    assert baked.voice in config.apps[baked.app].voices
    assert baked.other_voice not in config.apps[baked.app].voices


def test_tokens_minted_for_the_baked_caller_carry_its_identity(
    config: CallerConfig, minting: Minting
) -> None:
    baked = judged.baked_caller(config)
    pair = mint_pair(judged.minting_for(config, baked, minting))

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
        pytest.param(judged.without_mark(marked_wav()), id="no mark"),
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
        judged.require_marked_wav(wav)


def test_a_marked_wav_with_audio_passes_and_loses_its_mark_when_cut() -> None:
    judged.require_marked_wav(marked_wav())

    assert judged.ai_mark_of(judged.without_mark(marked_wav())) is None


def test_the_proof_fails_when_an_action_that_must_fail_does_not() -> None:
    judged.require_fails(
        lambda: judged.require_marked_wav(b""), "an empty answer", naming="not a WAV"
    )

    with pytest.raises(SystemExit):
        judged.require_fails(lambda: None, "nothing", naming="")


def test_the_proof_fails_when_an_action_fails_for_another_reason() -> None:
    def docker_is_down() -> None:
        message = "docker run failed"
        raise SystemExit(message)

    with pytest.raises(SystemExit, match="not as ConnectionRefusedError"):
        judged.require_fails(
            docker_is_down, "a loopback bind", naming="ConnectionRefusedError"
        )


def test_a_refusal_passes_only_as_the_expected_code_with_its_status() -> None:
    body = refusal(RefusalCode.FORBIDDEN).model_dump_json().encode()
    status = refusal(RefusalCode.FORBIDDEN).status

    reply = Reply(status, body)

    judged.require_refusal(reply, RefusalCode.FORBIDDEN)
    with pytest.raises(SystemExit):
        judged.require_refusal(reply, RefusalCode.UNAUTHORIZED)


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(Reply(200, marked_wav()), id="audio"),
        pytest.param(
            Reply(200, refusal(RefusalCode.BUSY).model_dump_json().encode()),
            id="status of another code",
        ),
    ],
)
def test_an_answer_that_is_no_matching_refusal_fails_the_proof(
    reply: Reply,
) -> None:
    with pytest.raises(SystemExit):
        judged.require_refusal(reply, RefusalCode.BUSY)


def test_locked_names_come_from_the_export_for_this_platform_and_its_projects() -> None:
    export = """\
# a comment
-e ./packages/client
Some_Package==1.0 ; python_version >= '3.14'
windows-only==1.0 ; sys_platform == 'win32'
webauth @ git+https://example.invalid/webauth@abc
    # via something
"""

    names = judged.locked_distributions(export, lambda path: f"own-{path[-6:]}")

    assert names == {"own-client", "some-package", "webauth"}


def clean_report() -> judged.ImageReport:
    return judged.ImageReport(
        distributions=["Some_Package"],
        user=1000,
        application_files=["deploy/speech.toml"],
        application_files_with_private_keys=[],
        package_stray_files=[],
        package_files_with_private_keys=[],
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
            {"package_files_with_private_keys": ["speech_runtime_x/a.key"]},
            "private key",
        ),
        ({"voice_hashes": {"voice.onnx": "other"}}, "pinned bytes"),
    ],
)
def test_an_image_that_differs_from_the_lock_is_named_by_its_fault(
    change: dict[str, object], fault: str
) -> None:
    report = judged.ImageReport(**{**vars(clean_report()), **change})

    faults = judged.image_faults(report, {"some-package"}, {"voice.onnx": "abc"})

    assert any(fault in line for line in faults)


def test_an_image_as_locked_has_no_fault() -> None:
    assert not judged.image_faults(
        clean_report(), {"some-package"}, {"voice.onnx": "abc"}
    )


def test_a_log_with_spoken_text_or_a_token_is_named() -> None:
    log = "INFO ok\nspoke: Hallo Welt\nheader abc.def.ghi"

    faults = judged.log_faults(log, ["Hallo Welt"], ["abc.def.ghi"])
    assert [fault.split()[0] for fault in faults] == ["spoken", "a"]
    assert not judged.log_faults("INFO ok", ["Hallo Welt"], ["abc.def.ghi"])


def test_only_the_external_networks_of_a_rendered_compose_file_are_named() -> None:
    rendered = {
        "networks": {
            "a": {"name": "outside", "external": True},
            "b": {"name": "inside"},
        }
    }

    assert judged.external_networks(rendered) == ["outside"]


def test_an_nvidia_distribution_in_the_image_is_named_even_when_the_lock_lists_it() -> (
    None
):
    report = judged.ImageReport(
        **{**vars(clean_report()), "distributions": ["Some_Package", "nvidia-nccl"]}
    )

    faults = judged.image_faults(
        report, {"some-package", "nvidia-nccl"}, {"voice.onnx": "abc"}
    )

    assert faults == ["holds nvidia-nccl"]
