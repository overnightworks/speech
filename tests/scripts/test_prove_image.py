"""The image proof's own logic: what it admits, what it refuses, and how it can fail."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jwt
import pytest

from speech_runtime_client.wire import RefusalCode, refusal
from speech_runtime_service.config import load_caller_config
from speech_runtime_service.pcm import wav_header
from tests.support.script_support import script_module

if TYPE_CHECKING:
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
    ],
)
def test_a_wav_without_the_mark_or_the_audio_fails_the_proof(wav: bytes) -> None:
    with pytest.raises(SystemExit):
        PROVE.require_marked_wav(wav)


def test_a_marked_wav_with_audio_passes_and_loses_its_mark_when_cut() -> None:
    PROVE.require_marked_wav(marked_wav())

    assert PROVE.ai_mark_of(PROVE.without_mark(marked_wav())) is None


def test_the_proof_fails_when_an_action_that_must_fail_does_not() -> None:
    PROVE.require_fails(lambda: PROVE.require_marked_wav(b""), "an empty answer")

    with pytest.raises(SystemExit):
        PROVE.require_fails(lambda: None, "nothing")


def test_a_refusal_passes_only_as_the_expected_code_with_its_status() -> None:
    body = refusal(RefusalCode.FORBIDDEN).model_dump_json().encode()
    status = refusal(RefusalCode.FORBIDDEN).status

    PROVE.require_refusal(PROVE.Reply(status, body), RefusalCode.FORBIDDEN)
    with pytest.raises(SystemExit):
        PROVE.require_refusal(PROVE.Reply(status, body), RefusalCode.UNAUTHORIZED)


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
        voice_hashes={"voice.onnx": "abc"},
        executables={"git": None},
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
