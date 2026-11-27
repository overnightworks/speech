"""Qwen speaks through its server's HTTP and fails closed on anything else."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest
from speech_provider_contract import QWEN_MODEL_ID, QWEN_SAMPLE_RATE, QwenSpeaker

from speech_client.wire import Language
from speech_service.failures import failure_fields, payload_free_chain
from speech_service.provider_process import (
    ProviderNotReadyError,
    ProviderSynthesisFailedError,
    UnsupportedLanguageError,
)
from tests.support.launcher_support import (
    SERVER_WAIT_SECONDS,
    assert_reaped,
    closed_provider,
    observed,
    qwen_voice,
    received_requests,
)
from tests.support.poll_support import wait_until

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from speech_service.speaking import QwenSpeaking


@pytest.fixture
def speaking(tmp_path: Path) -> Iterator[Callable[[str], QwenSpeaking]]:
    """Build a loaded Qwen voice over the fake server; close each at the end."""
    started: list[QwenSpeaking] = []

    def build(behavior: str = "normal") -> QwenSpeaking:
        root, cache = closed_provider(tmp_path, behavior)
        voice = qwen_voice(root, cache, QwenSpeaker.SOHEE, (Language.DE,))
        started.append(voice)
        voice.load()
        return voice

    yield build
    for voice in started:
        voice.close()


def test_a_loaded_voice_names_the_pinned_model_and_rate_and_is_ready(
    speaking: Callable[[str], QwenSpeaking],
) -> None:
    voice = speaking("normal")

    assert voice.ready is True
    assert voice.model_name == QWEN_MODEL_ID
    assert voice.sample_rate == QWEN_SAMPLE_RATE


@pytest.mark.parametrize(
    ("language", "language_name"),
    [("de", "German"), ("en", "English"), ("de-DE", "German"), ("EN_us", "English")],
)
def test_the_request_names_the_voice_and_the_language_of_the_server(
    tmp_path: Path,
    speaking: Callable[[str], QwenSpeaking],
    language: str,
    language_name: str,
) -> None:
    voice = speaking("normal")

    assert list(voice.pcm_chunks("Hallo", language)) == [b"\x00\x01"]

    *_, request = received_requests(tmp_path / "cache")
    assert request["input"] == "Hallo"
    assert request["voice"] == "Sohee"
    assert request["language"] == language_name
    assert request["response_format"] == "pcm"
    assert request["stream"] is True
    assert "instructions" not in request


def test_named_instructions_reach_the_server_for_that_utterance_only(
    tmp_path: Path, speaking: Callable[[str], QwenSpeaking]
) -> None:
    voice = speaking("normal")

    assert list(voice.pcm_chunks("Hallo", "de", "Sprich zornig"))
    assert list(voice.pcm_chunks("Hallo", "de"))

    *_, with_tone, without_tone = received_requests(tmp_path / "cache")
    assert with_tone["instructions"] == "Sprich zornig"
    assert "instructions" not in without_tone


def test_an_odd_remainder_at_the_end_fails_after_its_whole_samples_and_the_server(
    speaking: Callable[[str], QwenSpeaking],
) -> None:
    voice = speaking("normal")
    chunks = voice.pcm_chunks("odd", "de")

    assert next(chunks) == b"\x00\x01"
    with pytest.raises(ProviderSynthesisFailedError):
        next(chunks)

    assert voice.ready is False


def test_a_sample_split_across_two_chunks_comes_out_whole(
    speaking: Callable[[str], QwenSpeaking],
) -> None:
    voice = speaking("normal")

    assert list(voice.pcm_chunks("split-sample", "de")) == [b"\x00\x01"]
    assert voice.ready is True


def test_a_refusal_fails_the_synthesis_and_the_server_and_logs_only_its_status(
    tmp_path: Path,
    speaking: Callable[[str], QwenSpeaking],
    caplog: pytest.LogCaptureFixture,
) -> None:
    voice = speaking("normal")
    _, _, pid, _ = observed(tmp_path / "cache")

    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(RuntimeError, match="synthesis failed"),
    ):
        list(voice.pcm_chunks("refused", "de"))
    voice.close()

    (failed,) = (r for r in caplog.records if r.getMessage() == "synthesis.failed")
    assert failed.status == HTTPStatus.INTERNAL_SERVER_ERROR
    assert voice.ready is False
    assert "refused" not in "\n".join(
        f"{r.getMessage()} {r.exc_text or ''} {vars(r)}" for r in caplog.records
    )
    assert_reaped(pid)


def test_a_sample_rate_other_than_qwens_fails_the_load(
    speaking: Callable[[str], QwenSpeaking],
) -> None:
    with pytest.raises(ProviderSynthesisFailedError):
        speaking("other-rate")


def test_a_server_that_ends_mid_stream_fails_the_synthesis_and_logs_no_server_text(
    speaking: Callable[[str], QwenSpeaking], caplog: pytest.LogCaptureFixture
) -> None:
    voice = speaking("normal")
    chunks = voice.pcm_chunks("midstream-exit", "de")

    assert next(chunks) == b"\x00\x01"
    with pytest.raises(ProviderSynthesisFailedError) as raised:
        list(chunks)
    transport_text = str(raised.value.__context__)
    logging.getLogger("speaking").error(
        "speaking.failed",
        extra=failure_fields(raised.value),
        exc_info=payload_free_chain(raised.value),
    )

    assert voice.ready is False
    assert transport_text
    assert transport_text not in caplog.text


def test_closing_the_stream_early_hangs_up_and_leaves_the_server_ready(
    tmp_path: Path, speaking: Callable[[str], QwenSpeaking]
) -> None:
    voice = speaking("normal")
    chunks = voice.pcm_chunks("wait-disconnect", "de")
    assert next(chunks) == b"\x00\x01"

    chunks.close()

    observed_hang_up = tmp_path / "cache/disconnect-observed"
    wait_until(
        observed_hang_up.exists,
        within=SERVER_WAIT_SECONDS,
        waited_for="the server never observed the hang-up",
    )
    assert voice.ready is True
    assert list(voice.pcm_chunks("Hallo", "de")) == [b"\x00\x01"]


def test_an_unsupported_language_is_refused_before_any_request(
    tmp_path: Path, speaking: Callable[[str], QwenSpeaking]
) -> None:
    voice = speaking("normal")
    warm_up_requests = received_requests(tmp_path / "cache")

    with pytest.raises(UnsupportedLanguageError):
        list(voice.pcm_chunks("hello", "fr"))

    assert received_requests(tmp_path / "cache") == warm_up_requests
    assert voice.ready is True


def test_a_voice_that_is_not_loaded_refuses_its_request_as_not_ready(
    tmp_path: Path,
) -> None:
    root, cache = closed_provider(tmp_path)
    voice = qwen_voice(root, cache, QwenSpeaker.RYAN, (Language.DE,))

    with pytest.raises(ProviderNotReadyError):
        list(voice.pcm_chunks("hello", "de"))
