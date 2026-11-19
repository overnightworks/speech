"""Qwen speaks through its server's HTTP and fails closed on anything else."""

from __future__ import annotations

import logging
import time
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest
from speech_provider_contract import QWEN_MODEL_ID, QWEN_SAMPLE_RATE, QwenSpeaker

from speech_service.provider_process import (
    ProviderLaunch,
    ProviderNotReadyError,
    ProviderProcess,
    ProviderSynthesisFailedError,
    UnsupportedLanguageError,
)
from speech_service.speaking import QwenSpeaking
from tests.support.launcher_support import (
    SERVER_WAIT_SECONDS,
    assert_reaped,
    closed_provider,
    observed,
    provider_launcher,
    received_requests,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

_SENTENCE = "Ein Satz, den nur der Aufrufer kennt"


@pytest.fixture
def speaking(tmp_path: Path) -> Iterator[Callable[[str], QwenSpeaking]]:
    """Build a loaded Qwen voice over the fake server; close each at the end."""
    started: list[QwenSpeaking] = []

    def build(behavior: str = "normal") -> QwenSpeaking:
        root, cache = closed_provider(tmp_path, behavior)
        voice = QwenSpeaking(
            ProviderProcess(
                ProviderLaunch(
                    executable=provider_launcher(root),
                    arguments=("--cache", str(cache)),
                )
            ),
            QwenSpeaker.SOHEE,
        )
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

    (request,) = received_requests(tmp_path / "cache")
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

    with_tone, without_tone = received_requests(tmp_path / "cache")
    assert with_tone["instructions"] == "Sprich zornig"
    assert "instructions" not in without_tone


def test_a_chunk_reaches_the_caller_before_the_server_sends_the_next(
    tmp_path: Path, speaking: Callable[[str], QwenSpeaking]
) -> None:
    chunks = speaking("normal").pcm_chunks("gate", "de")

    assert next(chunks) == b"\x00\x01"

    (tmp_path / "cache/release-second").touch()
    assert list(chunks) == [b"\x02\x03"]


def test_an_odd_remainder_at_the_end_fails_after_its_whole_samples_and_the_server(
    speaking: Callable[[str], QwenSpeaking],
) -> None:
    voice = speaking("normal")
    chunks = voice.pcm_chunks("odd", "de")

    assert next(chunks) == b"\x00\x01"
    with pytest.raises(ProviderSynthesisFailedError):
        next(chunks)

    assert voice.ready is False


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


@pytest.mark.parametrize(
    ("behavior", "failing"), [("other-rate", True), ("normal", False)]
)
def test_a_sample_rate_other_than_qwens_fails_the_synthesis(
    speaking: Callable[[str], QwenSpeaking], behavior: str, *, failing: bool
) -> None:
    voice = speaking(behavior)

    if failing:
        with pytest.raises(ProviderSynthesisFailedError):
            list(voice.pcm_chunks("Hallo", "de"))
        assert voice.ready is False
    else:
        assert list(voice.pcm_chunks("Hallo", "de")) == [b"\x00\x01"]
        assert voice.ready is True


def test_a_server_that_ends_mid_stream_fails_the_synthesis_and_the_engine(
    speaking: Callable[[str], QwenSpeaking],
) -> None:
    voice = speaking("normal")
    chunks = voice.pcm_chunks("midstream-exit", "de")

    assert next(chunks) == b"\x00\x01"
    with pytest.raises(ProviderSynthesisFailedError):
        list(chunks)

    assert voice.ready is False


def test_closing_the_stream_early_hangs_up_and_leaves_the_server_ready(
    tmp_path: Path, speaking: Callable[[str], QwenSpeaking]
) -> None:
    voice = speaking("normal")
    chunks = voice.pcm_chunks("wait-disconnect", "de")
    assert next(chunks) == b"\x00\x01"

    chunks.close()

    observed_hang_up = tmp_path / "cache/disconnect-observed"
    deadline = time.monotonic() + SERVER_WAIT_SECONDS
    while not observed_hang_up.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert observed_hang_up.exists()
    assert voice.ready is True
    assert list(voice.pcm_chunks("Hallo", "de")) == [b"\x00\x01"]


def test_an_unsupported_language_is_refused_before_any_request(
    tmp_path: Path, speaking: Callable[[str], QwenSpeaking]
) -> None:
    voice = speaking("normal")

    with pytest.raises(UnsupportedLanguageError):
        list(voice.pcm_chunks("hello", "fr"))

    assert received_requests(tmp_path / "cache") == []
    assert voice.ready is True


def test_a_voice_that_is_not_loaded_refuses_its_request_as_not_ready(
    tmp_path: Path,
) -> None:
    root, cache = closed_provider(tmp_path)
    voice = QwenSpeaking(
        ProviderProcess(
            ProviderLaunch(
                executable=provider_launcher(root), arguments=("--cache", str(cache))
            )
        ),
        QwenSpeaker.RYAN,
    )

    with pytest.raises(ProviderNotReadyError):
        list(voice.pcm_chunks("hello", "de"))


def test_the_sentence_and_the_server_output_reach_no_log(
    speaking: Callable[[str], QwenSpeaking],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        voice = speaking("normal")
        assert list(voice.pcm_chunks(_SENTENCE, "de"))
        voice.close()

    logged = "\n".join(
        f"{r.getMessage()} {r.exc_text or ''} {vars(r)}" for r in caplog.records
    )
    assert _SENTENCE not in logged
    assert "server noise before ready" not in logged
    assert not [r for r in caplog.records if r.getMessage() == "provider.exited"]
