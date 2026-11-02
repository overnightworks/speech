"""The image proof's judgement: what it admits, what it refuses, and how it can fail."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jwt
import pytest

import image_judgement as judged
from image_stack import REPOSITORY, LiveOutcome, Reply
from proof_identity import mint_pair
from speech_runtime_client.wire import RefusalCode, Transcription, refusal
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


@pytest.fixture(params=[False, True], ids=["deaf app", "hearing app"])
def app(request: pytest.FixtureRequest, config: CallerConfig) -> str:
    return judged.app_that(config, hears=request.param)


def test_the_baked_table_names_an_app_that_hears_and_one_that_does_not(
    config: CallerConfig,
) -> None:
    hearing = judged.app_that(config, hears=True)
    deaf = judged.app_that(config, hears=False)

    assert config.apps[hearing].hear
    assert not config.apps[deaf].hear


def test_a_table_with_no_app_of_the_wanted_kind_fails_the_proof() -> None:
    only_deaf = CallerConfig.model_validate(
        {
            "identity": {
                "issuer": "i",
                "jwks_url": "https://k.invalid/certs",
                "audience": "a",
                "workload_role": "r",
                "group_claim": "g",
                "service_group": "/s",
            },
            "apps": {"one": {"workload_client": "c", "voices": ["piper"]}},
        }
    )

    with pytest.raises(SystemExit, match="hear=True"):
        judged.app_that(only_deaf, hears=True)


def test_the_baked_caller_names_a_voice_it_may_and_one_it_may_not_speak(
    config: CallerConfig, app: str
) -> None:
    baked = judged.baked_caller(config, app)

    assert baked.hear is config.apps[app].hear
    assert baked.voice in config.apps[baked.app].voices
    assert baked.other_voice not in config.apps[baked.app].voices


def test_tokens_minted_for_the_baked_caller_carry_its_identity(
    config: CallerConfig, minting: Minting, app: str
) -> None:
    baked = judged.baked_caller(config, app)
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
        hearing_model_directory="/opt/hearing/snapshots/revision",
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
        ({"hearing_model_directory": None}, "hearing model"),
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

    faults = judged.log_faults(log, ["Hallo Welt"], [], ["abc.def.ghi"])
    assert [fault.split()[0] for fault in faults] == ["spoken", "a"]
    assert not judged.log_faults("INFO ok", ["Hallo Welt"], ["gehört"], ["abc.def.ghi"])


def test_a_log_with_heard_text_is_named_and_fails_the_proof() -> None:
    log = "INFO ok\ntranscript: Guten Tag zusammen"

    assert judged.log_faults(log, [], ["Guten Tag zusammen"], [])[0].startswith(
        "heard text"
    )
    with pytest.raises(SystemExit, match="heard text"):
        judged.require_log_clean(log, [], ["Guten Tag zusammen"], [])
    judged.require_log_clean("INFO ok", [], ["Guten Tag zusammen"], [])


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


SENTENCE = "Die Kamera zeigt den Vortragenden, während die Folie vorbereitet wird."


def transcription(text: str, status: int = 200) -> Reply:
    return Reply(status, Transcription(text=text).model_dump_json().encode())


@pytest.mark.parametrize(
    ("heard", "words"),
    [
        pytest.param("die kamera zeigt", ["kamera", "zeigt"], id="case does not count"),
        pytest.param("Die den wird", [], id="short words match by accident"),
        pytest.param("Guten Tag zusammen", [], id="another text"),
        pytest.param("", [], id="nothing"),
    ],
)
def test_the_words_of_the_sentence_a_text_holds_leave_out_the_short_ones(
    heard: str, words: list[str]
) -> None:
    assert judged.heard_words(heard, SENTENCE) == words


def test_a_text_without_a_word_of_the_sentence_fails_the_proof() -> None:
    judged.require_heard("Die Kamera", SENTENCE)

    with pytest.raises(SystemExit, match="no word of the spoken sentence"):
        judged.require_heard("Guten Tag zusammen", SENTENCE)


@pytest.mark.parametrize(
    ("reply", "fault"),
    [
        pytest.param(transcription("Guten Tag zusammen"), "no word", id="wrong text"),
        pytest.param(Reply(403, b"{}"), "answered 403", id="a refusal"),
        pytest.param(Reply(500, b"oops"), "answered 500", id="a server error"),
    ],
)
def test_a_recording_answer_that_is_no_transcription_of_the_sentence_fails(
    reply: Reply, fault: str
) -> None:
    with pytest.raises(SystemExit, match=fault):
        judged.require_transcript(reply, SENTENCE)


def test_a_transcription_with_a_word_of_the_sentence_gives_its_text() -> None:
    assert judged.require_transcript(transcription("die Kamera"), SENTENCE) == (
        "die Kamera"
    )


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(transcription("anything"), id="text"),
        pytest.param(
            Reply(413, refusal(RefusalCode.TOO_LARGE).model_dump_json().encode()),
            id="too large",
        ),
    ],
)
def test_a_recording_at_the_limit_may_get_text_or_too_large(reply: Reply) -> None:
    judged.require_limit_answer(reply)


@pytest.mark.parametrize(
    ("reply", "fault"),
    [
        pytest.param(Reply(500, b"internal server error"), "no wire refusal", id="500"),
        pytest.param(
            Reply(429, refusal(RefusalCode.BUSY).model_dump_json().encode()),
            "expected",
            id="another refusal",
        ),
    ],
)
def test_a_recording_at_the_limit_that_got_anything_else_fails_the_proof(
    reply: Reply, fault: str
) -> None:
    with pytest.raises(SystemExit, match=fault):
        judged.require_limit_answer(reply)


@pytest.mark.parametrize(
    ("usage", "fault"),
    [
        pytest.param(judged.TmpUsage(1, 100), "spool is left", id="a spool is left"),
        pytest.param(judged.TmpUsage(0, 9), "smaller than one recording", id="small"),
    ],
)
def test_a_tmp_that_cannot_hold_a_recording_or_keeps_one_fails_the_proof(
    usage: judged.TmpUsage, fault: str
) -> None:
    with pytest.raises(SystemExit, match=fault):
        judged.require_tmp_clean(usage, 10)


def test_an_empty_tmp_that_holds_a_recording_passes() -> None:
    judged.require_tmp_clean(judged.TmpUsage(0, 10), 10)


@pytest.mark.parametrize(
    ("outcome", "fault"),
    [
        pytest.param(
            LiveOutcome(None, 0, "InvalidStatus"), "InvalidStatus", id="error"
        ),
        pytest.param(LiveOutcome("Guten Tag zusammen", 2, None), "no word", id="text"),
    ],
)
def test_a_live_turn_without_a_final_of_the_sentence_fails_the_proof(
    outcome: LiveOutcome, fault: str
) -> None:
    with pytest.raises(SystemExit, match=fault):
        judged.require_final(outcome, SENTENCE)


def test_a_live_turn_with_a_final_of_the_sentence_passes() -> None:
    judged.require_final(LiveOutcome("die Kamera", 0, None), SENTENCE)


@pytest.mark.parametrize(
    ("hearing", "passes", "fault"),
    [
        pytest.param({"model": None, "ready": False}, True, "not ready", id="no model"),
        pytest.param(
            {"model": "small", "ready": False}, True, "not ready", id="loading"
        ),
        pytest.param(
            {"model": "large-v3", "ready": True}, True, "not ready", id="another"
        ),
        pytest.param(
            {"model": "small", "ready": True}, False, "healthcheck", id="check"
        ),
    ],
)
def test_hearing_that_is_not_ready_with_the_model_or_unchecked_fails_the_proof(
    hearing: dict[str, object], *, passes: bool, fault: str
) -> None:
    with pytest.raises(SystemExit, match=fault):
        judged.require_hearing_ready(
            {"hearing": hearing}, "small", healthcheck_passes=passes
        )


def test_hearing_ready_with_the_model_and_a_passing_healthcheck_passes() -> None:
    judged.require_hearing_ready(
        {"hearing": {"model": "small", "ready": True}}, "small", healthcheck_passes=True
    )


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("30s", 30), ("1m30s", 90), ("2m", 120), ("1h", 3600)],
)
def test_a_rendered_compose_duration_is_read_in_seconds(
    text: str, seconds: int
) -> None:
    assert judged.duration_seconds(text) == seconds


STARTED = "2026-11-02T20:00:00.123456789Z"


def check(ended: str, exit_code: int) -> dict[str, object]:
    return {"End": ended, "ExitCode": exit_code}


def test_the_seconds_to_healthy_end_at_the_first_passing_check_after_the_start() -> (
    None
):
    log = [
        check("2026-11-02T19:59:00Z", 0),
        check("2026-11-02T20:00:03Z", 1),
        check("2026-11-02T20:00:07.623456789+00:00", 0),
        check("2026-11-02T20:00:30Z", 0),
    ]

    assert judged.seconds_to_healthy(STARTED, log) == pytest.approx(7.5)


@pytest.mark.parametrize(
    "log",
    [
        pytest.param([], id="no check"),
        pytest.param([check("2026-11-02T19:59:00Z", 0)], id="only before the start"),
        pytest.param([check("2026-11-02T20:00:03Z", 1)], id="only failing"),
    ],
)
def test_a_start_with_no_passing_check_after_it_fails_the_proof(
    log: list[dict[str, object]],
) -> None:
    with pytest.raises(SystemExit, match="no healthcheck passed"):
        judged.seconds_to_healthy(STARTED, log)


def test_a_service_that_was_healthy_after_its_start_period_fails_the_proof() -> None:
    judged.require_ready_within(29.9, "30s")

    with pytest.raises(SystemExit, match="past 30s"):
        judged.require_ready_within(30.1, "30s")
